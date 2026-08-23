import numpy as np
import pytest

from hal.products.mats import SDCSRoadMap

import taxi_mission

HUB = taxi_mission.HUB_NODE


@pytest.fixture
def roadmap():
    return SDCSRoadMap(leftHandTraffic=False)


class TestRouteManager:

    def test_plan_returns_waypoints_ending_at_the_goal_node(self, roadmap):
        manager = taxi_mission.RouteManager(roadmap)

        route = manager.plan([taxi_mission.HUB_NODE, 4])

        goalPose = roadmap.get_node_pose(4).squeeze()
        assert route.waypoints.shape[0] == 2
        assert route.waypoints.shape[1] > 1
        assert route.goal == pytest.approx((goalPose[0], goalPose[1]), abs=0.05)

    def test_plan_to_the_node_we_are_already_on_holds_position(self, roadmap):
        # generate_path returns None for [n, n]. The FSM reaches this whenever a
        # ride is dispatched to the node the car is parked on.
        manager = taxi_mission.RouteManager(roadmap)

        route = manager.plan([HUB, HUB])

        hubPose = roadmap.get_node_pose(HUB).squeeze()
        assert route.goal == pytest.approx((hubPose[0], hubPose[1]), abs=0.05)
        assert route.waypoints.shape[1] >= 1

    def test_plan_ignores_repeated_nodes_in_a_sequence(self, roadmap):
        manager = taxi_mission.RouteManager(roadmap)

        route = manager.plan([HUB, HUB, 4, 4])

        goalPose = roadmap.get_node_pose(4).squeeze()
        assert route.nodeSequence == [HUB, 4]
        assert route.goal == pytest.approx((goalPose[0], goalPose[1]), abs=0.05)


class TestSpeedProfile:
    """Speed shaping ported from the virtual taxi. Units are metres and m/s, the
    same on hardware, so the tuned constants carry over unchanged."""

    def settled(self):
        """A profile whose start-up ramp has already finished."""
        profile = taxi_mission.SpeedProfile()
        profile.reset(now=0.0)
        return profile

    def test_cruises_when_far_from_the_goal_and_driving_straight(self):
        profile = self.settled()

        speed = profile.update(distToGoal=3.0, steering=0.0, now=10.0)

        assert speed == pytest.approx(taxi_mission.CRUISE_SPEED)

    def test_slows_down_approaching_the_goal(self):
        profile = self.settled()

        speed = profile.update(distToGoal=0.3, steering=0.0, now=10.0)

        assert speed < taxi_mission.CRUISE_SPEED
        assert speed >= taxi_mission.MIN_SPEED

    def test_stops_inside_the_stop_radius(self):
        profile = self.settled()

        speed = profile.update(distToGoal=0.1, steering=0.0, now=10.0)

        assert speed == 0.0

    def test_holds_cruise_speed_through_a_curve(self):
        # Curve slowdown is off: slowing raises Stanley's cross-track gain and
        # saturates the steering on the very corner it slowed for. The demo
        # controller that tracks this map cleanly holds a constant speed.
        profile = self.settled()

        speed = profile.update(distToGoal=3.0, steering=0.5, now=10.0)

        assert speed == pytest.approx(taxi_mission.CRUISE_SPEED)

    def test_curve_slowdown_still_works_when_switched_on(self, monkeypatch):
        monkeypatch.setattr(taxi_mission, 'CURVE_SLOWDOWN', True)
        profile = self.settled()

        speed = profile.update(distToGoal=3.0, steering=0.5, now=10.0)

        assert speed == pytest.approx(taxi_mission.SHARP_CURVE_SPEED_CAP)

    def test_holds_still_at_the_start_of_a_new_route(self):
        profile = taxi_mission.SpeedProfile()
        profile.reset(now=100.0)

        speed = profile.update(distToGoal=3.0, steering=0.0, now=100.2)

        assert speed == 0.0

    def test_route_change_replays_the_start_hold(self):
        profile = self.settled()
        assert profile.update(distToGoal=3.0, steering=0.0, now=10.0) > 0.0

        profile.reset(now=20.0)

        assert profile.update(distToGoal=3.0, steering=0.0, now=20.2) == 0.0

    def test_ramps_up_gradually_after_the_start_hold(self):
        profile = taxi_mission.SpeedProfile()
        profile.reset(now=100.0)
        holdEnd = 100.0 + taxi_mission.ROUTE_START_HOLD

        early = profile.update(distToGoal=3.0, steering=0.0,
                               now=holdEnd + 0.3)
        late = profile.update(distToGoal=3.0, steering=0.0,
                              now=holdEnd + 1.5)

        assert 0.0 < early < late < taxi_mission.CRUISE_SPEED


PICKUP = 4
DROPOFF = 22


class MissionDriver:
    """Drives a TaxiMission the way the control loop does, with a fake clock so
    dwell times pass instantly."""

    def __init__(self, roadmap, **kwargs):
        self.roadmap = roadmap
        self.mission = taxi_mission.TaxiMission(
            taxi_mission.RouteManager(roadmap), **kwargs)
        self.now = 0.0
        self.command = None

    def nodeXY(self, node):
        pose = self.roadmap.get_node_pose(node).squeeze()
        return (float(pose[0]), float(pose[1]))

    def tick(self, position=(0.0, 0.0), steering=0.0, seconds=0.02):
        self.now += seconds
        self.command = self.mission.update(position, steering, self.now)
        return self.command

    def arriveAt(self, node, seconds=0.02):
        return self.tick(self.nodeXY(node), seconds=seconds)

    def waitAt(self, node, seconds):
        """Sit at a node long enough for any dwell timer to expire."""
        self.arriveAt(node)
        return self.arriveAt(node, seconds=seconds)


class TestTaxiMission:

    def test_starts_idle_at_the_hub(self, roadmap):
        driver = MissionDriver(roadmap)

        assert driver.mission.state == taxi_mission.TaxiState.IDLE_HUB

    def test_stays_stopped_while_idling_at_the_hub(self, roadmap):
        driver = MissionDriver(roadmap)

        command = driver.arriveAt(HUB)

        assert command.speed == 0.0
        assert driver.mission.state == taxi_mission.TaxiState.IDLE_HUB

    def test_dispatches_a_ride_to_the_pickup_node_after_the_idle_dwell(self, roadmap):
        driver = MissionDriver(roadmap, pickupNode=PICKUP, dropoffNode=DROPOFF)

        driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)

        assert driver.mission.state == taxi_mission.TaxiState.NAV_TO_PICKUP
        assert driver.mission.route.goal == pytest.approx(
            driver.nodeXY(PICKUP), abs=0.05)

    def test_switches_to_loading_on_reaching_the_pickup(self, roadmap):
        driver = MissionDriver(roadmap, pickupNode=PICKUP, dropoffNode=DROPOFF)
        driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)

        driver.arriveAt(PICKUP)

        assert driver.mission.state == taxi_mission.TaxiState.PASSENGER_LOADING

    def test_holds_still_while_the_passenger_boards(self, roadmap):
        driver = MissionDriver(roadmap, pickupNode=PICKUP, dropoffNode=DROPOFF)
        driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)

        command = driver.arriveAt(PICKUP)

        assert command.speed == 0.0

    def test_drives_to_the_dropoff_once_boarding_is_done(self, roadmap):
        driver = MissionDriver(roadmap, pickupNode=PICKUP, dropoffNode=DROPOFF)
        driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)

        driver.waitAt(PICKUP, seconds=taxi_mission.BOARDING_DWELL + 0.1)

        assert driver.mission.state == taxi_mission.TaxiState.NAV_TO_DROPOFF
        assert driver.mission.route.goal == pytest.approx(
            driver.nodeXY(DROPOFF), abs=0.05)

    def test_unloads_then_returns_to_the_hub(self, roadmap):
        driver = MissionDriver(roadmap, pickupNode=PICKUP, dropoffNode=DROPOFF)
        driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)
        driver.waitAt(PICKUP, seconds=taxi_mission.BOARDING_DWELL + 0.1)

        driver.arriveAt(DROPOFF)
        assert driver.mission.state == taxi_mission.TaxiState.PASSENGER_UNLOADING

        driver.waitAt(DROPOFF, seconds=taxi_mission.BOARDING_DWELL + 0.1)
        assert driver.mission.state == taxi_mission.TaxiState.RETURN_TO_HUB
        assert driver.mission.route.goal == pytest.approx(
            driver.nodeXY(HUB), abs=0.05)

    def test_completes_a_full_ride_and_becomes_available_again(self, roadmap):
        driver = MissionDriver(roadmap, pickupNode=PICKUP, dropoffNode=DROPOFF)
        driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)
        driver.waitAt(PICKUP, seconds=taxi_mission.BOARDING_DWELL + 0.1)
        driver.waitAt(DROPOFF, seconds=taxi_mission.BOARDING_DWELL + 0.1)

        driver.arriveAt(HUB)

        assert driver.mission.state == taxi_mission.TaxiState.IDLE_HUB
        assert driver.mission.ridesCompleted == 1

    def test_reports_a_route_change_only_on_the_step_that_changes_it(self, roadmap):
        driver = MissionDriver(roadmap, pickupNode=PICKUP, dropoffNode=DROPOFF)

        dispatched = driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)
        following = driver.tick(driver.nodeXY(HUB))

        assert dispatched.routeChanged
        assert not following.routeChanged

    def test_signals_lamps_are_sized_for_the_qcar2_led_array(self, roadmap):
        driver = MissionDriver(roadmap)
        lamps = taxi_mission.SignalLamps()

        assert len(lamps.update(driver.tick().state, steering=0.0, now=0.0)) == 8

    def test_a_ride_dispatched_to_the_hub_itself_still_completes(self, roadmap):
        # Degenerate dispatch: pickup == hub. The roadmap cannot route [10, 10],
        # so this is the case that used to crash route generation.
        driver = MissionDriver(roadmap, pickupNode=HUB, dropoffNode=DROPOFF)

        driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)

        assert driver.mission.state == taxi_mission.TaxiState.NAV_TO_PICKUP
        driver.arriveAt(HUB)
        assert driver.mission.state == taxi_mission.TaxiState.PASSENGER_LOADING


class TestSignalLamps:
    """The QCar2 has eight digital lamps, not the RGB strip the virtual taxi
    used, so mission state is signalled with real car lighting.
    Index convention comes from Quanser's own hardware examples:
    [0],[2] left indicator, [1],[3] right, [4] brake, [5] reverse, [6],[7] headlamps.
    """

    LEFT = (0, 2)
    RIGHT = (1, 3)
    BRAKE = 4
    HEADLAMPS = (6, 7)

    def test_headlamps_are_on_while_driving(self):
        lamps = taxi_mission.SignalLamps()

        leds = lamps.update(taxi_mission.TaxiState.NAV_TO_PICKUP,
                            steering=0.0, now=0.0)

        assert all(leds[i] == 1 for i in self.HEADLAMPS)
        assert leds[self.BRAKE] == 0

    def test_no_indicators_when_driving_straight(self):
        lamps = taxi_mission.SignalLamps()

        leds = lamps.update(taxi_mission.TaxiState.NAV_TO_PICKUP,
                            steering=0.0, now=0.0)

        assert all(leds[i] == 0 for i in self.LEFT + self.RIGHT)

    def test_indicates_left_when_steering_left(self):
        lamps = taxi_mission.SignalLamps()

        leds = lamps.update(taxi_mission.TaxiState.NAV_TO_DROPOFF,
                            steering=0.3, now=0.0)

        assert all(leds[i] == 1 for i in self.LEFT)
        assert all(leds[i] == 0 for i in self.RIGHT)

    def test_indicates_right_when_steering_right(self):
        lamps = taxi_mission.SignalLamps()

        leds = lamps.update(taxi_mission.TaxiState.NAV_TO_DROPOFF,
                            steering=-0.3, now=0.0)

        assert all(leds[i] == 1 for i in self.RIGHT)
        assert all(leds[i] == 0 for i in self.LEFT)

    def test_brake_lights_come_on_while_a_passenger_boards(self):
        lamps = taxi_mission.SignalLamps()

        leds = lamps.update(taxi_mission.TaxiState.PASSENGER_LOADING,
                            steering=0.0, now=0.0)

        assert leds[self.BRAKE] == 1

    def test_hazards_blink_while_waiting_at_the_hub(self):
        lamps = taxi_mission.SignalLamps()
        half = taxi_mission.HAZARD_BLINK_PERIOD / 2.0

        first = lamps.update(taxi_mission.TaxiState.IDLE_HUB,
                             steering=0.0, now=0.0)
        second = lamps.update(taxi_mission.TaxiState.IDLE_HUB,
                              steering=0.0, now=half + 0.01)

        allIndicators = self.LEFT + self.RIGHT
        assert all(first[i] == 1 for i in allIndicators)
        assert all(second[i] == 0 for i in allIndicators)


class TestSteeringController:

    def straightPath(self):
        return np.array([[0.0, 1.0, 2.0, 3.0],
                         [0.0, 0.0, 0.0, 0.0]])

    def test_steers_toward_a_path_offset_to_the_left(self):
        controller = taxi_mission.SteeringController(self.straightPath(), k=0.7)

        steering = controller.update(np.array([0.5, -0.3]), 0.0, 0.5)

        assert steering > 0.0

    def test_steers_toward_a_path_offset_to_the_right(self):
        controller = taxi_mission.SteeringController(self.straightPath(), k=0.7)

        steering = controller.update(np.array([0.5, 0.3]), 0.0, 0.5)

        assert steering < 0.0

    def test_holds_straight_on_a_stationary_single_waypoint_route(self):
        # A route to the node the car is already parked on is a single point.
        # Stanley cannot form a tangent from one waypoint; it must not emit NaN
        # into qcar.write().
        controller = taxi_mission.SteeringController(
            np.array([[1.0], [2.0]]), k=0.7)

        steering = controller.update(np.array([1.0, 2.0]), 0.0, 0.0)

        assert np.isfinite(steering)
        assert steering == 0.0

    def test_update_path_restarts_tracking_on_the_new_route(self):
        controller = taxi_mission.SteeringController(self.straightPath(), k=0.7)
        controller.update(np.array([2.5, 0.0]), 0.0, 0.5)
        assert controller.wpi > 0

        controller.update_path(self.straightPath())

        assert controller.wpi == 0


class TestRouteFollower:

    def follower(self, roadmap):
        return taxi_mission.RouteFollower(roadmap), taxi_mission.RouteManager(roadmap)

    def nodePose(self, roadmap, node):
        return np.asarray(roadmap.get_node_pose(node)).ravel()

    def test_steers_a_normal_route_with_a_finite_angle(self, roadmap):
        follower, manager = self.follower(roadmap)
        route = manager.plan([HUB, PICKUP])
        pose = self.nodePose(roadmap, HUB)

        follower.set_route(route, pose)
        steering = follower.update(pose[:2], pose[2], 0.5)

        assert np.isfinite(steering)
        assert abs(steering) <= np.pi/6

    def test_a_stationary_route_produces_no_steering(self, roadmap):
        follower, manager = self.follower(roadmap)
        route = manager.plan([HUB, HUB])
        pose = self.nodePose(roadmap, HUB)

        follower.set_route(route, pose)

        assert follower.update(pose[:2], pose[2], 0.0) == 0.0

    def test_no_approach_path_when_parked_at_the_route_start(self, roadmap):
        follower, manager = self.follower(roadmap)
        route = manager.plan([PICKUP, DROPOFF])
        pose = self.nodePose(roadmap, PICKUP)

        follower.set_route(route, pose)

        assert follower.approach is None

    def test_no_approach_path_when_parked_just_short_of_the_route_start(self, roadmap):
        # The car parks up to PICKUP_RADIUS from the node, so this is the normal
        # mid-ride case. Stanley converges over the first metre on its own.
        follower, manager = self.follower(roadmap)
        route = manager.plan([PICKUP, DROPOFF])
        pose = self.nodePose(roadmap, PICKUP).copy()
        pose[0] += taxi_mission.PICKUP_RADIUS

        follower.set_route(route, pose)

        assert follower.approach is None

    def test_does_not_detour_when_a_route_starts_at_the_node_we_are_parked_on(self, roadmap):
        # roadmap.initial_check judges "have we arrived" with a Dubins distance
        # that reports a full turning circle for an identical pose. Trusting it
        # here sent the car on an 11 m lap before a 3.7 m route.
        follower, manager = self.follower(roadmap)
        route = manager.plan([DROPOFF, HUB])
        pose = self.nodePose(roadmap, DROPOFF)

        follower.set_route(route, pose)

        assert follower.approach is None

    def test_builds_an_approach_path_from_a_distant_pose(self, roadmap):
        # Startup: the car is powered on somewhere else on the map entirely.
        follower, manager = self.follower(roadmap)
        route = manager.plan([PICKUP, DROPOFF])
        pose = self.nodePose(roadmap, HUB)

        follower.set_route(route, pose)

        assert follower.approach is not None
        assert np.isfinite(follower.update(pose[:2], pose[2], 0.3))


class TestFullRide:
    """End-to-end check of mission, routing and steering against a kinematic
    bicycle model. Covers everything taxi_control.py does except the device I/O:
    if a change makes the taxi stop completing rides, or wander out of its lane,
    this fails."""

    WHEELBASE = 0.256
    DT = 0.02
    SPEED_LAG = 0.30  # s, first-order approximation of the drivetrain

    def drive(self, roadmap, seconds=200.0, rides=1):
        mission = taxi_mission.TaxiMission(
            taxi_mission.RouteManager(roadmap),
            pickupNode=PICKUP, dropoffNode=DROPOFF)
        follower = taxi_mission.RouteFollower(roadmap)

        x, y, th = np.asarray(
            roadmap.get_node_pose(HUB)).ravel().astype(float)
        v, now, delta, routeAge = 0.0, 0.0, 0.0, 0.0
        deviations = []

        for _ in range(int(seconds / self.DT)):
            now += self.DT
            command = mission.update((x, y), delta, now)
            if command.routeChanged:
                follower.set_route(command.route, np.array([x, y, th]))
                routeAge = 0.0
            routeAge += self.DT

            frontAxle = (np.array([x, y])
                         + np.array([np.cos(th), np.sin(th)]) * 0.2)
            delta = follower.update(frontAxle, th, max(v, 0.05)) \
                if follower.controller else 0.0

            v += (command.speed - v) * self.DT / self.SPEED_LAG
            x += v * np.cos(th) * self.DT
            y += v * np.sin(th) * self.DT
            th += v * np.tan(delta) / self.WHEELBASE * self.DT
            th = np.arctan2(np.sin(th), np.cos(th))

            onRoute = (command.route is not None
                       and command.route.waypoints.shape[1] > 1
                       and follower.approach is None
                       and routeAge > 2.0)
            if onRoute:
                deviations.append(float(np.min(np.hypot(
                    command.route.waypoints[0, :] - x,
                    command.route.waypoints[1, :] - y))))

            if mission.ridesCompleted >= rides:
                break

        return mission, np.array(deviations)

    def test_completes_a_ride_and_returns_to_the_hub(self, roadmap):
        mission, _ = self.drive(roadmap)

        assert mission.ridesCompleted >= 1
        assert mission.state == taxi_mission.TaxiState.IDLE_HUB

    def test_keeps_the_car_on_the_planned_path(self, roadmap):
        _, deviations = self.drive(roadmap)

        assert len(deviations) > 100
        assert deviations.mean() < 0.05
        assert deviations.max() < 0.20

    def test_keeps_running_across_consecutive_rides(self, roadmap):
        mission, _ = self.drive(roadmap, seconds=400.0, rides=2)

        assert mission.ridesCompleted >= 2


class TestContinuousOperation:
    """The taxi is meant to keep working unattended, drawing a new pickup and
    drop-off for every ride, until it is stopped."""

    def test_each_ride_draws_a_fresh_pickup_and_dropoff(self, roadmap):
        rides = [(0, 14), (4, 20), (2, 0)]
        issued = []

        def dispatch():
            return rides[min(len(issued), len(rides) - 1)]

        driver = MissionDriver(roadmap, dispatch=dispatch)
        seen = []

        for _ in range(len(rides)):
            driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)
            issued.append(1)
            seen.append((driver.mission.pickupNode, driver.mission.dropoffNode))
            driver.waitAt(driver.mission.pickupNode,
                          seconds=taxi_mission.BOARDING_DWELL + 0.1)
            driver.waitAt(driver.mission.dropoffNode,
                          seconds=taxi_mission.BOARDING_DWELL + 0.1)
            driver.arriveAt(HUB)

        assert seen == rides
        assert driver.mission.ridesCompleted == len(rides)

    def test_every_random_pair_the_dispatcher_can_pick_is_drivable(self, roadmap):
        # A pair that cannot be routed would end the shift with a RouteError.
        manager = taxi_mission.RouteManager(roadmap)
        for pool in taxi_mission.PICKUP_POOLS.values():
            for pickup in pool:
                for dropoff in pool:
                    if pickup == dropoff:
                        continue
                    for leg in ([HUB, pickup], [pickup, dropoff], [dropoff, HUB]):
                        assert manager.plan(leg).waypoints.shape[1] > 1

    def test_keeps_dispatching_rides_indefinitely(self, roadmap):
        import random
        rng = random.Random(7)
        pool = taxi_mission.PICKUP_POOLS[0]

        def dispatch():
            pickup = rng.choice(pool)
            return pickup, rng.choice([n for n in pool if n != pickup])

        driver = MissionDriver(roadmap, dispatch=dispatch)

        for _ in range(12):
            driver.waitAt(HUB, seconds=taxi_mission.IDLE_DWELL + 0.1)
            driver.waitAt(driver.mission.pickupNode,
                          seconds=taxi_mission.BOARDING_DWELL + 0.1)
            driver.waitAt(driver.mission.dropoffNode,
                          seconds=taxi_mission.BOARDING_DWELL + 0.1)
            driver.arriveAt(HUB)

        assert driver.mission.ridesCompleted == 12
        assert driver.mission.state == taxi_mission.TaxiState.IDLE_HUB


class TestSteeringAtLowSpeed:
    """Stanley's cross-track term is atan(k*e/v), so the gain grows without
    bound as the car slows. The demo controller never meets this because it does
    not stop; the taxi stops at every node, so the speed is floored."""

    def straightPath(self):
        return np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0]])

    def test_matches_the_demo_controller_at_cruise(self):
        # At and above the floor the behaviour must be identical to the
        # controller that tracks this map cleanly: atan(k*e/v), no softening.
        controller = taxi_mission.SteeringController(self.straightPath())
        offset, speed = 0.10, taxi_mission.CRUISE_SPEED

        steering = controller.update(np.array([0.5, -offset]), 0.0, speed)

        expected = np.arctan2(taxi_mission.K_STANLEY * offset, speed)
        assert steering == pytest.approx(expected, abs=1e-6)

    def test_still_corrects_in_the_right_direction_at_low_speed(self):
        controller = taxi_mission.SteeringController(self.straightPath())

        left = controller.update(np.array([0.5, -0.3]), 0.0,
                                 taxi_mission.SHARP_CURVE_SPEED_CAP)
        controller.update_path(self.straightPath())
        right = controller.update(np.array([0.5, 0.3]), 0.0,
                                  taxi_mission.SHARP_CURVE_SPEED_CAP)

        assert left > 0.05
        assert right < -0.05

    def test_gain_stops_growing_below_the_speed_floor(self):
        controller = taxi_mission.SteeringController(self.straightPath())

        atFloor = controller.update(np.array([0.5, -0.05]), 0.0,
                                    taxi_mission.STANLEY_MIN_SPEED)
        controller.update_path(self.straightPath())
        crawling = controller.update(np.array([0.5, -0.05]), 0.0, 0.01)
        controller.update_path(self.straightPath())
        stopped = controller.update(np.array([0.5, -0.05]), 0.0, 0.0)

        assert crawling == pytest.approx(atFloor, abs=1e-6)
        assert stopped == pytest.approx(atFloor, abs=1e-6)


class TestSteeringWaypointTracking:
    """vehicle_control.py's Stanley advances its waypoint index only by
    projecting the car onto the segment it is currently tracking, one index per
    call. If the car ends up beside or past that segment, the projection stops
    growing and the index sticks: the controller then steers against a segment
    the car left long ago."""

    def arcPath(self, n=400, radius=0.7):
        t = np.linspace(0.0, np.pi, n)
        return np.vstack([radius * np.cos(t), radius * np.sin(t)])

    def straightPath(self, n=300, spacing=0.01):
        xs = np.arange(n) * spacing
        return np.vstack([xs, np.zeros(n)])

    def test_catches_up_when_the_car_is_ahead_of_the_tracked_index(self):
        path = self.straightPath()
        controller = taxi_mission.SteeringController(path)

        # Car is at waypoint 60 but the controller still thinks it is at 0.
        controller.update(path[:, 60].copy(), 0.0, 0.4)

        assert controller.wpi >= 55

    def test_catch_up_is_capped_so_a_hairpin_cannot_be_skipped(self):
        # Re-anchoring searches a bounded window ahead, so a car that is a long
        # way down the path converges over several cycles rather than teleporting
        # past a tight bend that doubles back within that distance.
        path = self.straightPath()
        controller = taxi_mission.SteeringController(path)
        target = 250

        controller.update(path[:, target].copy(), 0.0, 0.4)
        afterOne = controller.wpi
        for _ in range(4):
            controller.update(path[:, target].copy(), 0.0, 0.4)

        assert afterOne <= controller.SEARCH_WINDOW
        assert controller.wpi >= target - 5

    def test_index_keeps_advancing_when_the_car_drifts_off_to_the_side(self):
        # The lock-up case: motion perpendicular to the tracked segment never
        # increases the along-segment projection.
        path = self.straightPath()
        controller = taxi_mission.SteeringController(path)
        controller.update(path[:, 50].copy(), 0.0, 0.4)
        stuckIndex = controller.wpi

        for step in range(1, 40):
            drifted = path[:, 50 + step].copy()
            drifted[1] += 0.4  # pushed sideways off the path
            controller.update(drifted, 0.0, 0.25)

        assert controller.wpi > stuckIndex

    def test_steers_back_toward_the_path_after_drifting_sideways(self):
        path = self.straightPath()
        controller = taxi_mission.SteeringController(path)
        offPath = path[:, 100].copy()
        offPath[1] += 0.4  # 0.4 m to the left of the path

        steering = controller.update(offPath, 0.0, 0.25)

        assert steering < 0.0  # must turn right, back toward the path

    def test_does_not_jump_backwards_onto_an_earlier_part_of_the_route(self):
        # A route can pass close to itself; re-anchoring must only look forward.
        path = self.arcPath()
        controller = taxi_mission.SteeringController(path)
        controller.update(path[:, 200].copy(), 0.0, 0.4)
        advanced = controller.wpi

        controller.update(path[:, 205].copy(), 0.0, 0.4)

        assert controller.wpi >= advanced


class TestPoseOffset:
    """Diagnostic geometry for measuring localisation bias: park on a known
    node, compare what the GPS reports against what the node actually is."""

    def test_no_offset_when_the_report_matches_the_node(self):
        offset = taxi_mission.pose_offset(reported=(1.0, 2.0),
                                          nodePose=(1.0, 2.0, 0.0))

        assert offset.dx == pytest.approx(0.0)
        assert offset.dy == pytest.approx(0.0)
        assert offset.distance == pytest.approx(0.0)

    def test_reports_the_raw_frame_offset(self):
        offset = taxi_mission.pose_offset(reported=(1.0, 1.84),
                                          nodePose=(1.0, 2.0, 0.0))

        assert offset.dy == pytest.approx(-0.16)
        assert offset.distance == pytest.approx(0.16)

    def test_a_y_offset_is_lateral_on_an_east_west_road(self):
        # Heading 0 = driving +x, so a y error puts the car across its lane.
        offset = taxi_mission.pose_offset(reported=(1.0, 1.84),
                                          nodePose=(1.0, 2.0, 0.0))

        assert abs(offset.lateral) == pytest.approx(0.16)
        assert offset.longitudinal == pytest.approx(0.0, abs=1e-9)

    def test_the_same_y_offset_is_along_track_on_a_north_south_road(self):
        # Heading -pi/2 = driving -y, so the same error only shifts where the
        # car thinks it is along the road, not across it.
        offset = taxi_mission.pose_offset(reported=(1.0, 1.84),
                                          nodePose=(1.0, 2.0, -np.pi/2))

        assert abs(offset.longitudinal) == pytest.approx(0.16)
        assert offset.lateral == pytest.approx(0.0, abs=1e-9)

    def test_lateral_sign_is_positive_when_reported_left_of_the_node(self):
        # Heading 0 (+x); left is +y.
        offset = taxi_mission.pose_offset(reported=(1.0, 2.2),
                                          nodePose=(1.0, 2.0, 0.0))

        assert offset.lateral > 0


class TestNearestNode:
    """roadmap.get_closest_node ranks by Dubins path length, which returns a
    full turning circle for a pose compared against itself. Sitting on a node it
    therefore prefers a different one metres away. Diagnostics need plain
    distance."""

    def test_picks_the_node_the_car_is_sitting_on(self, roadmap):
        pose = np.asarray(roadmap.get_node_pose(0)).ravel()

        assert taxi_mission.nearest_node(roadmap, pose[0], pose[1]) == 0

    def test_uses_heading_to_tell_the_two_lanes_of_a_road_apart(self, roadmap):
        # The exact reading that made get_closest_node answer node 2. Node 1 is
        # 2 cm nearer than node 0, but it is the opposite carriageway.
        facingNode0 = np.radians(-87.8)

        assert taxi_mission.nearest_node(roadmap, 0.147, 0.120,
                                         heading=facingNode0) == 0
        assert taxi_mission.nearest_node(roadmap, 0.147, 0.120) == 1

    def test_picks_the_opposite_lane_when_facing_the_other_way(self, roadmap):
        facingNode1 = np.radians(90.0)

        assert taxi_mission.nearest_node(roadmap, 0.147, 0.120,
                                         heading=facingNode1) == 1

    def test_agrees_with_euclidean_distance_across_the_map(self, roadmap):
        for node in (0, 2, 4, 10, 11, 20, 22):
            pose = np.asarray(roadmap.get_node_pose(node)).ravel()
            assert taxi_mission.nearest_node(roadmap, pose[0], pose[1],
                                             heading=pose[2]) == node


class TestSanitizePath:
    """roadmap.generate_path stitches each edge's waypoints together with the
    pose of the node that follows it. Where an edge overshoots that node by a
    few millimetres the join doubles back, and Stanley reads the reversed
    tangent as a 180 degree turn: full steering lock, right at a junction.
    Routes leaving node 14 do this at the traffic-circle entry."""

    def straight(self):
        return np.vstack([np.arange(10) * 0.01, np.zeros(10)])

    def test_leaves_a_clean_path_alone(self):
        path = self.straight()

        assert taxi_mission.sanitize_path(path).shape == path.shape

    def test_drops_duplicated_waypoints(self):
        path = np.array([[0.0, 0.01, 0.01, 0.02],
                         [0.0, 0.00, 0.00, 0.00]])

        cleaned = taxi_mission.sanitize_path(path)

        assert cleaned.shape[1] == 3

    def test_drops_a_waypoint_that_reverses_direction(self):
        # third point steps backwards, creating a 180 degree tangent flip
        path = np.array([[0.0, 0.01, 0.005, 0.02, 0.03],
                         [0.0, 0.00, 0.000, 0.00, 0.00]])

        cleaned = taxi_mission.sanitize_path(path)

        steps = np.diff(cleaned, axis=1)
        directions = steps / np.linalg.norm(steps, axis=0)
        dots = np.sum(directions[:, 1:] * directions[:, :-1], axis=0)
        assert np.all(dots > 0)

    def test_keeps_the_path_endpoints_meaningful(self):
        path = self.straight()

        cleaned = taxi_mission.sanitize_path(path)

        assert cleaned[:, 0] == pytest.approx(path[:, 0])
        assert cleaned[:, -1] == pytest.approx(path[:, -1])

    def test_planned_routes_never_contain_a_reversal(self, roadmap):
        # The real defect: every route out of node 14 had one.
        manager = taxi_mission.RouteManager(roadmap)
        for leg in ([14, 10], [14, 0], [14, 2], [14, 4], [HUB, 4], [4, 22]):
            path = manager.plan(leg).waypoints
            steps = np.diff(path, axis=1)
            lengths = np.linalg.norm(steps, axis=0)
            assert np.all(lengths > 1e-9), 'duplicate waypoint in %s' % leg
            directions = steps / lengths
            dots = np.sum(directions[:, 1:] * directions[:, :-1], axis=0)
            assert np.all(dots > -0.5), 'reversal in route %s' % leg


class TestCrossTrackError:
    """Signed distance from the car to its planned path, so a run can be judged
    after the fact instead of by eye."""

    def path(self):
        return np.vstack([np.arange(50) * 0.01, np.zeros(50)])

    def test_zero_on_the_path(self):
        assert taxi_mission.cross_track_error(self.path(), 0.25, 0.0) == \
            pytest.approx(0.0, abs=1e-9)

    def test_positive_to_the_left_of_travel(self):
        # path runs +x, so +y is the driver's left
        assert taxi_mission.cross_track_error(self.path(), 0.25, 0.05) > 0

    def test_negative_to_the_right_of_travel(self):
        assert taxi_mission.cross_track_error(self.path(), 0.25, -0.05) < 0

    def test_reports_the_perpendicular_distance(self):
        assert taxi_mission.cross_track_error(self.path(), 0.25, 0.08) == \
            pytest.approx(0.08, abs=1e-3)

    def test_safe_on_a_single_point_path(self):
        assert taxi_mission.cross_track_error(np.array([[0.0], [0.0]]),
                                              1.0, 1.0) == 0.0


class TestSteeringTrim:
    """Stanley is purely proportional, so any constant disturbance -- wheels not
    centred at zero command, or a biased heading estimate -- leaves a permanent
    cross-track error it cannot remove. A fixed trim on the output cancels it,
    and the same trim works for either cause."""

    def test_zero_trim_passes_the_command_through(self):
        assert taxi_mission.apply_steering_trim(0.12, 0.0) == pytest.approx(0.12)

    def test_trim_shifts_the_command(self):
        assert taxi_mission.apply_steering_trim(0.12, -0.05) == \
            pytest.approx(0.07)

    def test_result_stays_within_the_mechanical_limit(self):
        limit = np.pi/6
        assert taxi_mission.apply_steering_trim(0.50, 0.20) == \
            pytest.approx(limit)
        assert taxi_mission.apply_steering_trim(-0.50, -0.20) == \
            pytest.approx(-limit)

    def test_a_left_riding_car_is_corrected_by_a_negative_trim(self):
        # Car sits left of the lane holding -3.4 deg; trimming that out lets it
        # settle with zero commanded steering.
        held = np.radians(-3.4)

        assert taxi_mission.apply_steering_trim(held, -held) == \
            pytest.approx(0.0, abs=1e-9)


class TestHeadingTrim:
    """The car settles pointing several degrees off the path yet never
    converges, which a non-holonomic car cannot do. The heading estimate is
    biased, so Stanley computes its heading error from a wrong angle. Correcting
    the angle fixes the cause; a steering trim only masks it and skews turns."""

    def test_zero_trim_passes_the_heading_through(self):
        assert taxi_mission.apply_heading_trim(1.0, 0.0) == pytest.approx(1.0)

    def test_adds_the_trim(self):
        assert taxi_mission.apply_heading_trim(1.0, 0.1) == pytest.approx(1.1)

    def test_result_stays_wrapped(self):
        trimmed = taxi_mission.apply_heading_trim(np.pi - 0.05, 0.2)

        assert -np.pi <= trimmed <= np.pi
        assert trimmed == pytest.approx(-np.pi + 0.15, abs=1e-9)

    def test_removes_the_apparent_heading_error(self):
        # Car truly parallel to the path, but the estimate reads 6.1 deg low.
        tangent = 0.7
        bias = np.radians(-6.1)
        reported = tangent + bias

        corrected = taxi_mission.apply_heading_trim(reported, -bias)

        assert corrected == pytest.approx(tangent, abs=1e-9)
