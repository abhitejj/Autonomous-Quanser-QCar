"""
taxi_mission.py

Mission and control logic for the autonomous taxi: route planning on the SDCS
roadmap, the pickup/drop-off state machine, speed shaping, and the speed and
steering controllers.

Imports only numpy and Quanser's pure-maths helpers, never the device layer, so
all of it can be exercised on a development machine without a car attached.
Everything that touches the QCar, the GPS, the EKF or the YOLO stream lives in
taxi_control.py.
"""

import numpy as np

from pal.utilities.math import wrap_to_pi

HUB_NODE = 10

# ===== Controller gains
K_P = 0.1        # speed controller proportional gain
K_I = 1.0        # speed controller integral gain
K_STANLEY = 1.0  # steering controller cross-track gain
# Floor on the speed used in Stanley's cross-track term atan(k*e/v). At and
# above this the behaviour is identical to the demo controller; below it the
# gain would otherwise grow without bound and saturate the steering, which the
# demo never sees because it does not stop at nodes and the taxi does.
STANLEY_MIN_SPEED = 0.2  # m/s

# ===== Speed shaping
# Carried over from the virtual taxi. Distances are in metres, speeds in m/s and
# steering in radians on both platforms, so these transfer unchanged. Note the
# steering thresholds sit inside the hardware limit of pi/6 (0.524 rad), so a
# sharp curve is still reachable.
CRUISE_SPEED = 0.6              # m/s   nominal cruise, as the demo controller
DECEL_RADIUS = 0.6              # m     distance at which terminal braking starts
STOP_RADIUS = 0.15              # m     treated as arrived
MIN_SPEED = 0.12                # m/s   creep floor while decelerating
ROUTE_START_HOLD = 0.55         # s     stand still when a new route is loaded
ROUTE_START_RAMP = 2.20         # s     then ramp up to cruise over this long
# Curve slowdown came from the virtual taxi, which used Pure Pursuit. Stanley's
# cross-track term is atan(k*e/v), so slowing down raises its gain: the car
# saturated its steering on exactly the corners it had slowed for. The demo
# controller that tracks this map cleanly holds a constant speed, so this is off
# by default.
CURVE_SLOWDOWN = False
CURVE_STEER_THRESHOLD = 0.24    # rad   beyond this, treat as a curve
SHARP_CURVE_STEER_THRESHOLD = 0.42  # rad
CURVE_SPEED_CAP = 0.32          # m/s
SHARP_CURVE_SPEED_CAP = 0.24    # m/s
POST_TURN_SPEED_CAP = 0.28      # m/s   held briefly after leaving a curve
POST_TURN_RECOVERY_TIME = 1.25  # s

# ===== Mission
DEFAULT_PICKUP_NODE = 4
DEFAULT_DROPOFF_NODE = 22
IDLE_DWELL = 2.0        # s  pause at the hub between rides
BOARDING_DWELL = 3.0    # s  stopped time at pickup and drop-off
PICKUP_RADIUS = 0.35    # m  close enough to the pickup node to let a fare board
ARRIVAL_RADIUS = 0.20   # m  close enough to a drop-off or the hub

# Node pools for a future second taxi. These are the two non-conflicting sets
# vehicle_control.py uses, filtered to nodes a taxi can sensibly stop at.
PICKUP_POOLS = {
    0: [0, 2, 4, 14, 20],
    1: [3, 5, 8, 21, 23],
}


def sanitize_path(path, minStep=1e-3):
    """Remove duplicate and backtracking waypoints from a generated path.

    generate_path stitches each edge's waypoints together with the pose of the
    node that follows it. When an edge's last waypoint overshoots that node by a
    few millimetres the join doubles back on itself, which reads to the Stanley
    controller as a 180 degree turn: it commands full steering lock and, because
    the car can never make forward progress along a segment that points
    backwards, the waypoint index stops advancing and the lock sticks. Every
    route leaving node 14 does this at the traffic-circle entry.

    Dropping the offending points costs a few millimetres of path fidelity and
    leaves the trajectory strictly forward-going.
    """
    if path is None or path.shape[1] < 3:
        return path

    keep = [0]
    for i in range(1, path.shape[1]):
        step = path[:, i] - path[:, keep[-1]]
        stepLength = np.linalg.norm(step)
        if stepLength < minStep:
            # duplicate point, contributes no direction information
            continue
        if len(keep) >= 2:
            previous = path[:, keep[-1]] - path[:, keep[-2]]
            previousLength = np.linalg.norm(previous)
            if previousLength > 0 and np.dot(step/stepLength,
                                             previous/previousLength) < 0:
                # this point would reverse the direction of travel
                continue
        keep.append(i)
    return path[:, keep]


class Route:
    """A planned path: the waypoints to track and the point we are driving to."""

    def __init__(self, waypoints, nodeSequence):
        self.waypoints = waypoints
        self.nodeSequence = nodeSequence
        self.goal = (float(waypoints[0, -1]), float(waypoints[1, -1]))


class RouteManager:
    """Plans routes between roadmap nodes.

    generate_path returns None for a sequence it cannot connect, most commonly
    [n, n] when a ride is dispatched to the node the car is already parked on.
    Rather than propagating that None into the controller, such a sequence is
    collapsed to a stationary route holding the node's own position.
    """

    def __init__(self, roadmap):
        self.roadmap = roadmap

    def _node_position(self, node):
        pose = np.asarray(self.roadmap.get_node_pose(node)).ravel()
        return np.array([[pose[0]], [pose[1]]], dtype=float)

    def _dedupe(self, nodeSequence):
        clean = []
        for node in nodeSequence:
            node = int(node)
            if not clean or clean[-1] != node:
                clean.append(node)
        return clean

    def plan(self, nodeSequence):
        sequence = self._dedupe(nodeSequence)
        if not sequence:
            raise ValueError('cannot plan a route from an empty node sequence')

        if len(sequence) == 1:
            return Route(self._node_position(sequence[0]), sequence)

        waypoints = self.roadmap.generate_path(sequence)
        if waypoints is None or waypoints.shape[1] == 0:
            raise RouteError(
                'no roadmap path for node sequence {}'.format(sequence))
        return Route(sanitize_path(waypoints), sequence)


class RouteError(RuntimeError):
    """Raised when the roadmap cannot connect a requested node sequence."""


class SpeedProfile:
    """Turns the distance still to drive and the current steering angle into a
    target speed.

    This is only the mission's own opinion of how fast to go. The perception
    gain from the YOLO server is applied separately, and the lower of the two
    wins, so a red light or a close car always overrides it.
    """

    def __init__(self, cruiseSpeed=CRUISE_SPEED):
        self.cruiseSpeed = cruiseSpeed
        self._routeStartTime = 0.0
        self._lastTurnTime = 0.0

    def reset(self, now):
        """Begin a new route: hold briefly, then ramp back up to cruise."""
        self._routeStartTime = now
        self._lastTurnTime = now - POST_TURN_RECOVERY_TIME

    def update(self, distToGoal, steering, now):
        if distToGoal < STOP_RADIUS:
            return 0.0

        speed = self.cruiseSpeed
        if distToGoal < DECEL_RADIUS:
            speed = max(MIN_SPEED,
                        self.cruiseSpeed * distToGoal / DECEL_RADIUS)

        absSteering = abs(steering)
        if not CURVE_SLOWDOWN:
            pass
        elif absSteering > SHARP_CURVE_STEER_THRESHOLD:
            speed = min(speed, SHARP_CURVE_SPEED_CAP)
            self._lastTurnTime = now
        elif absSteering > CURVE_STEER_THRESHOLD:
            speed = min(speed, CURVE_SPEED_CAP)
            self._lastTurnTime = now
        elif now - self._lastTurnTime < POST_TURN_RECOVERY_TIME:
            speed = min(speed, POST_TURN_SPEED_CAP)

        return min(speed, self._startupCap(now))

    def _startupCap(self, now):
        """Stand still, then ease up to cruise, whenever a route is loaded.

        Without this the very first steering command of a route is applied at
        full speed while the pose estimate is still settling.
        """
        elapsed = now - self._routeStartTime
        if elapsed >= ROUTE_START_HOLD + ROUTE_START_RAMP:
            return self.cruiseSpeed
        if elapsed < ROUTE_START_HOLD:
            return 0.0
        ramp = (elapsed - ROUTE_START_HOLD) / ROUTE_START_RAMP
        return self.cruiseSpeed * ramp


class TaxiState:
    IDLE_HUB = 'IDLE_HUB'
    NAV_TO_PICKUP = 'NAV_TO_PICKUP'
    PASSENGER_LOADING = 'PASSENGER_LOADING'
    NAV_TO_DROPOFF = 'NAV_TO_DROPOFF'
    PASSENGER_UNLOADING = 'PASSENGER_UNLOADING'
    RETURN_TO_HUB = 'RETURN_TO_HUB'


class MissionCommand:
    """What the mission wants the control loop to do this cycle."""

    def __init__(self, state, speed, route, routeChanged, distToGoal):
        self.state = state
        self.speed = speed
        self.route = route
        self.routeChanged = routeChanged
        self.distToGoal = distToGoal


class TaxiMission:
    """The taxi state machine.

    Rides run hub -> pickup -> drop-off -> hub, forever. The mission only
    decides where to drive and how fast; steering and the perception speed gain
    are applied by the caller.
    """

    def __init__(self, routeManager, pickupNode=DEFAULT_PICKUP_NODE,
                 dropoffNode=DEFAULT_DROPOFF_NODE, hubNode=HUB_NODE,
                 dispatch=None, cruiseSpeed=CRUISE_SPEED):
        self.routeManager = routeManager
        self.hubNode = hubNode
        self.dispatch = dispatch or (lambda: (pickupNode, dropoffNode))
        self.speedProfile = SpeedProfile(cruiseSpeed=cruiseSpeed)

        self.state = TaxiState.IDLE_HUB
        self.route = None
        self.ridesCompleted = 0
        self.pickupNode = pickupNode
        self.dropoffNode = dropoffNode

        self._stateEntryTime = None
        self._routeChanged = False

    # ---------------- helpers ----------------
    def _enter(self, state, now):
        self.state = state
        self._stateEntryTime = now

    def _dwellElapsed(self, now):
        if self._stateEntryTime is None:
            return 0.0
        return now - self._stateEntryTime

    def _loadRoute(self, nodeSequence, now):
        self.route = self.routeManager.plan(nodeSequence)
        self.speedProfile.reset(now)
        self._routeChanged = True

    def _distToGoal(self, position):
        if self.route is None:
            return float('inf')
        return float(np.hypot(self.route.goal[0] - position[0],
                              self.route.goal[1] - position[1]))

    # ---------------- state machine ----------------
    def update(self, position, steering, now):
        if self._stateEntryTime is None:
            self._stateEntryTime = now
        self._routeChanged = False

        distToGoal = self._distToGoal(position)
        handler = self._HANDLERS[self.state]
        handler(self, position, distToGoal, now)

        if self.state in self._STOPPED_STATES:
            speed = 0.0
        else:
            speed = self.speedProfile.update(
                self._distToGoal(position), steering, now)

        return MissionCommand(self.state, speed, self.route,
                              self._routeChanged, distToGoal)

    def _idleAtHub(self, position, distToGoal, now):
        if self._dwellElapsed(now) < IDLE_DWELL:
            return
        self.pickupNode, self.dropoffNode = self.dispatch()
        self._loadRoute([self.hubNode, self.pickupNode], now)
        self._enter(TaxiState.NAV_TO_PICKUP, now)

    def _navToPickup(self, position, distToGoal, now):
        if distToGoal < PICKUP_RADIUS:
            self._enter(TaxiState.PASSENGER_LOADING, now)

    def _loading(self, position, distToGoal, now):
        if self._dwellElapsed(now) < BOARDING_DWELL:
            return
        self._loadRoute([self.pickupNode, self.dropoffNode], now)
        self._enter(TaxiState.NAV_TO_DROPOFF, now)

    def _navToDropoff(self, position, distToGoal, now):
        if distToGoal < ARRIVAL_RADIUS:
            self._enter(TaxiState.PASSENGER_UNLOADING, now)

    def _unloading(self, position, distToGoal, now):
        if self._dwellElapsed(now) < BOARDING_DWELL:
            return
        self._loadRoute([self.dropoffNode, self.hubNode], now)
        self._enter(TaxiState.RETURN_TO_HUB, now)

    def _returnToHub(self, position, distToGoal, now):
        if distToGoal < ARRIVAL_RADIUS:
            self.ridesCompleted += 1
            self._enter(TaxiState.IDLE_HUB, now)

    _HANDLERS = {
        TaxiState.IDLE_HUB: _idleAtHub,
        TaxiState.NAV_TO_PICKUP: _navToPickup,
        TaxiState.PASSENGER_LOADING: _loading,
        TaxiState.NAV_TO_DROPOFF: _navToDropoff,
        TaxiState.PASSENGER_UNLOADING: _unloading,
        TaxiState.RETURN_TO_HUB: _returnToHub,
    }

    _STOPPED_STATES = (TaxiState.IDLE_HUB,
                       TaxiState.PASSENGER_LOADING,
                       TaxiState.PASSENGER_UNLOADING)


# ===== Signal lamps
INDICATOR_STEER_THRESHOLD = 0.15  # rad  steering angle that trips a turn signal
HAZARD_BLINK_PERIOD = 0.66        # s    full on/off cycle for the hazard lights

# Lamp indices in the 8-element array QCar.write() expects. Taken from Quanser's
# own QCar2 hardware examples.
LAMP_LEFT = (0, 2)
LAMP_RIGHT = (1, 3)
LAMP_BRAKE = 4
LAMP_REVERSE = 5
LAMP_HEADLAMPS = (6, 7)


class SignalLamps:
    """Maps mission state onto the QCar2's eight digital lamps.

    The virtual taxi signalled state with a uniform RGB strip, which this car
    does not have. Real car lighting carries the same information: hazards while
    waiting for a fare, brake lights during boarding, indicators through turns.
    """

    def update(self, state, steering, now):
        leds = [0] * 8

        stopped = state in (TaxiState.IDLE_HUB,
                            TaxiState.PASSENGER_LOADING,
                            TaxiState.PASSENGER_UNLOADING)

        if not stopped:
            for i in LAMP_HEADLAMPS:
                leds[i] = 1

        if stopped:
            if state != TaxiState.IDLE_HUB:
                leds[LAMP_BRAKE] = 1
            if self._blinkOn(now):
                for i in LAMP_LEFT + LAMP_RIGHT:
                    leds[i] = 1
        elif steering > INDICATOR_STEER_THRESHOLD:
            for i in LAMP_LEFT:
                leds[i] = 1
        elif steering < -INDICATOR_STEER_THRESHOLD:
            for i in LAMP_RIGHT:
                leds[i] = 1

        return leds

    def _blinkOn(self, now):
        return int(now / (HAZARD_BLINK_PERIOD / 2.0)) % 2 == 0


# ------------------------------------------------------------------
# Controllers
# ------------------------------------------------------------------
class SpeedController:
    """PI controller turning a speed error into motor throttle.

    Same controller as vehicle_control.py; the taxi supplies a target speed that
    varies with the mission instead of a single fixed v_ref.
    """

    def __init__(self, kp=K_P, ki=K_I, maxThrottle=0.3):
        self.maxThrottle = maxThrottle
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt*e

        return np.clip(
            self.kp*e + self.ki*self.ei,
            -self.maxThrottle,
            self.maxThrottle
        )

    def reset(self):
        """Clear the integrator so a stop at a node does not wind it up."""
        self.ei = 0


class SteeringController:
    """Stanley controller tracking a waypoint sequence.

    Same controller as vehicle_control.py, with two additions the taxi needs:
    update_path(), because a ride swaps in a new path at every stage, and a
    guard for single-point paths.
    """

    # How far ahead of the tracked index to look when re-anchoring. Waypoints
    # are ~1 cm apart, so this is a 1 m search: long enough to recover from a
    # large excursion, short enough never to skip a hairpin.
    SEARCH_WINDOW = 100

    def __init__(self, waypoints, k=K_STANLEY, cyclic=True):
        self.maxSteeringAngle = np.pi/6

        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0

        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0

    def update_path(self, waypoints, cyclic=None):
        """Track a new path from its beginning."""
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        if cyclic is not None:
            self.cyclic = cyclic

    def _reanchor(self, p):
        """Snap the tracked index to the nearest waypoint ahead of it.

        Without this the index only advances by projecting the car onto the
        segment it is already tracking, one step per call. Drift sideways and
        that projection stops growing, so the index sticks and the controller
        steers against a segment the car has long since passed -- which showed
        up as the wheel pinned to its limit and an 0.8 m excursion leaving
        node 14. The search only looks forward, so a route that doubles back
        near itself cannot drag the car onto an earlier leg.
        """
        window = min(self.N, self.wpi + self.SEARCH_WINDOW)
        segment = self.wp[:, self.wpi:window]
        if segment.shape[1] == 0:
            return
        distances = np.hypot(segment[0, :] - p[0], segment[1, :] - p[1])
        self.wpi += int(np.argmin(distances))

    def update(self, p, th, speed):
        # A route to the node we are already parked on is a single point. There
        # is no tangent to follow, and computing one divides by a zero-length
        # segment, which numpy answers with nan rather than an exception. Hold
        # the wheel straight instead of steering by nan.
        if self.N < 2:
            return 0.0

        self._reanchor(p)

        wp_1 = self.wp[:, np.mod(self.wpi, self.N-1)]
        wp_2 = self.wp[:, np.mod(self.wpi+1, self.N-1)]

        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        if v_mag < 1e-9:
            # numpy returns nan rather than raising on a zero-length segment,
            # so guard explicitly: step over the degenerate pair instead of
            # steering on a meaningless tangent.
            if self.cyclic or self.wpi < self.N-2:
                self.wpi += 1
            return 0.0
        v_uv = v / v_mag

        tangent = np.arctan2(v_uv[1], v_uv[0])

        s = np.dot(p-wp_1, v_uv)

        if s >= v_mag:
            if self.cyclic or self.wpi < self.N-2:
                self.wpi += 1

        ep = wp_1 + v_uv*s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent-th)

        return float(np.clip(
            wrap_to_pi(psi + np.arctan2(self.k*ect,
                                        max(abs(speed), STANLEY_MIN_SPEED))),
            -self.maxSteeringAngle,
            self.maxSteeringAngle))


class RouteFollower:
    """Keeps a steering controller pointed at the mission's current route.

    Most route changes happen while parked on, or within a few centimetres of,
    the node the new route starts from; Stanley pulls the car onto the path over
    the first metre and no help is needed. Only when the car is genuinely
    elsewhere on the map, which in practice means at start-up, is a road
    following approach path planned via roadmap.initial_check.

    That distance test is deliberately made here rather than delegated to
    initial_check. initial_check measures "have we arrived" with a Dubins path
    length, which for a pose compared against itself returns a full turning
    circle rather than zero, so it answers "not arrived" almost always. Trusting
    it on every route change sent the car on an 11 m lap of the map before
    starting a 3.7 m route.
    """

    APPROACH_REACHED_DISTANCE = 0.2  # m  approach path considered complete
    APPROACH_REQUIRED_DISTANCE = 0.5  # m  beyond this, plan a way back to the route

    def __init__(self, roadmap, k=K_STANLEY):
        self.roadmap = roadmap
        self.k = k
        self.controller = None
        self.approach = None
        self.routeStart = None

    def set_route(self, route, pose):
        self.approach = None
        self.routeStart = None

        if self.controller is None:
            self.controller = SteeringController(route.waypoints, k=self.k)
        else:
            self.controller.update_path(route.waypoints)

        if route.waypoints.shape[1] < 2:
            return

        routeStart = route.waypoints[:, 0]
        pose = np.asarray(pose).ravel()
        if np.linalg.norm(routeStart - pose[:2]) <= self.APPROACH_REQUIRED_DISTANCE:
            return

        startNodeReached, approachWaypoints = self.roadmap.initial_check(
            pose, route.nodeSequence, route.waypoints)

        if not startNodeReached and approachWaypoints is not None \
                and approachWaypoints.shape[1] >= 2:
            approachWaypoints = sanitize_path(approachWaypoints)
            self.approach = SteeringController(approachWaypoints,
                                               k=self.k, cyclic=False)
            self.routeStart = routeStart

    def update(self, p, th, v):
        if self.controller is None:
            return 0.0
        if self.approach is not None:
            if np.linalg.norm(self.routeStart - p) < self.APPROACH_REACHED_DISTANCE:
                self.approach = None
            else:
                return self.approach.update(p, th, v)
        return self.controller.update(p, th, v)


class PoseOffset:
    """Difference between a reported pose and a known node, in useful frames.

    dx, dy are the raw map-frame error, which is the localisation bias itself.
    lateral and longitudinal resolve it relative to the direction of travel at
    that node: lateral is across the lane, longitudinal along it. Which of the
    two a given bias produces depends on the road's heading, so the same bias
    pushes the car out of its lane on one road and merely shifts where it stops
    on another.
    """

    def __init__(self, dx, dy, lateral, longitudinal):
        self.dx = dx
        self.dy = dy
        self.lateral = lateral
        self.longitudinal = longitudinal
        self.distance = float(np.hypot(dx, dy))


def pose_offset(reported, nodePose):
    """Offset of a reported position from a node, positive lateral = left."""
    dx = float(reported[0]) - float(nodePose[0])
    dy = float(reported[1]) - float(nodePose[1])
    heading = float(nodePose[2])

    longitudinal = dx * np.cos(heading) + dy * np.sin(heading)
    lateral = -dx * np.sin(heading) + dy * np.cos(heading)
    return PoseOffset(dx, dy, float(lateral), float(longitudinal))


def nearest_node(roadmap, x, y, heading=None):
    """Index of the roadmap node the car is at, by distance and direction.

    roadmap.get_closest_node measures with a Dubins path, which for a pose
    compared against itself returns a full turning circle rather than zero.
    Parked on a node it will name a different one metres away.

    Plain distance is not enough either: the two directions of a road are
    separate nodes barely 15 cm apart, so without a heading the wrong lane wins
    as often as not. When heading is given, only nodes facing broadly the same
    way are considered.
    """
    best, bestDistance = None, float('inf')
    for index in range(len(roadmap.nodes)):
        pose = np.asarray(roadmap.get_node_pose(index)).ravel()
        if heading is not None:
            difference = np.arctan2(np.sin(heading - pose[2]),
                                    np.cos(heading - pose[2]))
            if abs(difference) > np.pi/2:
                continue
        distance = float(np.hypot(pose[0] - x, pose[1] - y))
        if distance < bestDistance:
            best, bestDistance = index, distance
    return best


def cross_track_error(waypoints, x, y):
    """Signed distance from (x, y) to a path. Positive is left of travel.

    Used for logging: it is the number that says whether the car held its lane,
    and where it did not.
    """
    if waypoints is None or waypoints.shape[1] < 2:
        return 0.0
    distances = np.hypot(waypoints[0, :] - x, waypoints[1, :] - y)
    nearest = int(np.argmin(distances))
    following = min(nearest + 1, waypoints.shape[1] - 1)
    if following == nearest:
        nearest = max(0, nearest - 1)
        following = nearest + 1
    tangent = waypoints[:, following] - waypoints[:, nearest]
    length = np.linalg.norm(tangent)
    if length < 1e-9:
        return 0.0
    tangent = tangent / length
    offset = np.array([x - waypoints[0, nearest], y - waypoints[1, nearest]])
    return float(np.cross(tangent, offset))


def apply_steering_trim(steering, trim, limit=np.pi/6):
    """Add a constant trim to a steering command, respecting the servo limit.

    Stanley has no integral term, so a constant disturbance -- wheels that are
    not centred at zero command, or a heading estimate with a fixed bias --
    settles the car at a permanent cross-track error rather than being removed.
    A measured trim on the output cancels it. The same value corrects either
    cause, because both appear to the controller as the same steady offset.
    """
    return float(np.clip(steering + trim, -limit, limit))


def apply_heading_trim(heading, trim):
    """Correct a biased heading estimate before it reaches the controller.

    Stanley's first term is the heading error between the car and the path, so a
    biased heading estimate steers the car off a wrong angle and it settles at a
    permanent cross-track offset -- pointing at the path but never reaching it.
    A steering trim can mask that, but it biases every turn and consumes the
    steering range asymmetrically. Correcting the angle addresses the cause.
    """
    return float(wrap_to_pi(heading + trim))
