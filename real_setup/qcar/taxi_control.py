# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
taxi_control.py

Autonomous taxi application for the physical QCar 2.

Runs the ride state machine from taxi_mission.py on real hardware: pose comes
from the LiDAR scan-matching GPS fused in an EKF, steering from a Stanley
controller tracking SDCS roadmap waypoints, and the speed gain for signs,
lights, cars and pedestrians from yolo_server.py over the local YOLO stream.

Launch it the same way as vehicle_control.py, by setting APP=taxi_control.py in
config.txt and running run_all_cars.bat. yolo_server.py must be running too.
"""
import signal
import argparse
import os
import random
import time
from threading import Thread, Lock

import numpy as np

from pal.products.qcar import QCar, QCarGPS, QCarCameras
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from hal.utilities.image_processing import ImageProcessing

import cv2

import taxi_mission
import lane_vision
from utils import YOLOReceiver, YOLODriveLogic

parser = argparse.ArgumentParser(prog='Autonomous taxi')
parser.add_argument('-c', '--calibrate', default=False,
                    help='capture a reference scan instead of driving')
parser.add_argument('-n', '--node_configuration', type=int, default=0,
                    help='pickup pool for this car, so a second taxi can use '
                         'non-conflicting nodes')
parser.add_argument('-f', '--fixed', action='store_true',
                    help='drive the same pickup and drop-off every ride '
                         'instead of drawing new ones at random')
parser.add_argument('-p', '--pickup', type=int,
                    default=taxi_mission.DEFAULT_PICKUP_NODE,
                    help='pickup node when --fixed is used (default: %d)'
                         % taxi_mission.DEFAULT_PICKUP_NODE)
parser.add_argument('-d', '--dropoff', type=int,
                    default=taxi_mission.DEFAULT_DROPOFF_NODE,
                    help='drop-off node when --fixed is used (default: %d)'
                         % taxi_mission.DEFAULT_DROPOFF_NODE)
parser.add_argument('-v', '--cruise', type=float,
                    default=taxi_mission.CRUISE_SPEED,
                    help='cruise speed in m/s (default: %.2f)'
                         % taxi_mission.CRUISE_SPEED)
parser.add_argument('--lane-trim', dest='lane_trim', action='store_true',
                    help='enable the experimental camera lane-keeping trim. '
                         'Off by default: the roadmap path already defines the '
                         'right-hand lane, and an absolute vision trim fights '
                         'it (see lane_vision.py)')
parser.add_argument('--heading-trim', type=float, default=0.0,
                    help='constant correction added to the heading estimate, '
                         'in radians. Use this when the car settles pointing at '
                         'its lane but never reaches it. Measure with '
                         'analyse_run.py.')
parser.add_argument('--steer-trim', type=float, default=0.0,
                    help='constant steering trim in radians, added to every '
                         'command. Use it to cancel a car that rides to one '
                         'side of its lane: negative moves it right. Measure '
                         'with analyse_run.py.')
parser.add_argument('--log', type=str, default=None,
                    help='write a CSV of the run (pose, commands, cross-track '
                         'error) so a bad manoeuvre can be examined afterwards')
parser.add_argument('--preview', type=float, default=0.20,
                    help='Stanley reference point ahead of the rear axle, in '
                         'metres (default: 0.20). Raise it if the car cuts '
                         'corners, lower it if it swings wide.')
parser.add_argument('--lane-survey', action='store_true',
                    help='drive normally with the trim disabled while measuring '
                         'the lane signal, and report the nominal to put in '
                         'lane_vision.NOMINAL_STEERING')
parser.add_argument('--lane-nominal', type=float, default=None,
                    help='surveyed lane nominal for this car (default: the '
                         'value stored in lane_vision.NOMINAL_STEERING)')
parser.add_argument('--lane-rate', type=float, default=30.0,
                    help='lane camera update rate in Hz (default: 30)')
parser.add_argument('--duration', type=float, default=None,
                    help='stop after this many seconds (default: run until '
                         'stopped with Ctrl+C or stop_all_cars.sh)')

args = parser.parse_args()
calibrate = args.calibrate

#================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: how long to keep driving. The taxi is a service, not an experiment: by
#   default it works rides until it is stopped, so this is unbounded unless
#   --duration is given.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = args.duration if args.duration is not None else float('inf')
startDelay = 1
controllerUpdateRate = 200

# Define the calibration pose
#
# This is handed to the qcarLidarToGPS scan matcher as pose_0: it declares the
# map-frame coordinates of the spot where the reference scan was captured. It
# must agree with the existing angles_new.mat / distance_new.mat, not with any
# particular node, because every pose the GPS reports is measured from it.
#
# Measured on this car with pose_check.py: parked on node 11 the GPS reported
# y = +1.969 while node 11 is at y = +2.163, so the existing reference scan was
# captured on node 11 but was being declared as [0, 2, -pi/2]. That mislabel
# biased every reported position by 19 cm -- 72% of a lane width -- which is
# why the car would not hold its lane.
#
# The scan itself is fine: pose_0 only labels where it was taken, so correcting
# the label fixes the frame without recalibrating. The car MUST be parked on
# node 11 when the GPS starts, since this doubles as the matcher's initial
# guess.
calibrationPose = [0, 2.000, -np.pi/2]  # on the x=0 road, 16 cm short of node 11

# Used to enable safe keyboard triggered shutdown
global KILL_THREAD
KILL_THREAD = False


def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)
#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


LANE_MAX_AGE = 0.3  # s, ignore lane data older than this


class RunLog:
    """CSV record of a run, for diagnosing a manoeuvre after the fact.

    crossTrack is the number that matters: signed metres from the planned path,
    positive to the left of travel. A lane is 0.27 m wide, so anything beyond
    about 0.14 m has the car outside its lane.
    """

    COLUMNS = ('t', 'state', 'nodes', 'x', 'y', 'th', 'v', 'throttle',
               'delta', 'pathSteering', 'laneTrim', 'yoloGain',
               'crossTrack', 'distToGoal', 'wpi')

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.handle = None
        self.rows = 0

        directory = os.path.dirname(self.path)
        try:
            if directory:
                os.makedirs(directory, exist_ok=True)
            self.handle = open(self.path, 'w')
            self.handle.write(','.join(self.COLUMNS) + '\n')
            print('Logging the run to %s' % self.path)
        except OSError as e:
            # Diagnostics must never stop the car from driving.
            print('Could not open log file %s (%s); continuing without a log.'
                  % (self.path, e))

    def add(self, t, command, x, y, th, v, throttle, delta, pathSteering,
            trim, yoloGain):
        if self.handle is None:
            return
        route = command.route
        waypoints = route.waypoints if route is not None else None
        crossTrack = taxi_mission.cross_track_error(waypoints, x, y)
        nodes = ('|'.join(str(n) for n in route.nodeSequence)
                 if route is not None else '')
        self.handle.write(
            '%.3f,%s,%s,%.4f,%.4f,%.4f,%.3f,%.4f,%.4f,%.4f,%.4f,%.3f,'
            '%.4f,%.3f,%d\n'
            % (t, command.state, nodes, x, y, th, v, throttle, delta,
               pathSteering, trim, yoloGain, crossTrack, command.distToGoal,
               getattr(command, 'wpi', 0)))
        self.rows += 1

    def close(self):
        if self.handle is None:
            return
        try:
            self.handle.close()
            print('\nRun log: %d samples written to %s' % (self.rows, self.path))
        except Exception:
            pass


class LaneSurvey:
    """Measures where the yellow line sits while the roadmap drives the car.

    The lane trim must be applied relative to this value, not to the lab
    example's nominal, which was calibrated for a different camera mount and a
    different desired lane offset. Samples are taken only when the trim itself
    would act, so the number is directly usable as its zero point.
    """

    def __init__(self):
        self.samples = []

    def add(self, slope, intercept, pixelCount, pathSteering):
        if pixelCount < lane_vision.MIN_LANE_PIXELS:
            return
        if abs(pathSteering) > lane_vision.TURN_SUPPRESS_STEERING:
            return
        value = lane_vision.lane_steering_from_fit(slope, intercept)
        if np.isfinite(value):
            self.samples.append(value)

    def report(self):
        if len(self.samples) < 200:
            print('\nLane survey: only %d usable samples; drive a few full '
                  'laps with the lane visible before trusting a nominal.'
                  % len(self.samples))
            return
        values = np.array(self.samples)
        median = float(np.median(values))
        print('\n=== Lane survey ===')
        print('  samples      : %d' % values.size)
        print('  median       : %+.4f   <-- put this in '
              'lane_vision.NOMINAL_STEERING' % median)
        print('  mean / std   : %+.4f / %.4f' % (values.mean(), values.std()))
        print('  10th / 90th  : %+.4f / %+.4f'
              % (np.percentile(values, 10), np.percentile(values, 90)))
        print('  Then run with --lane-trim --lane-nominal %+.4f' % median)


class LaneCamera:
    """Background CSI front camera reader producing a yellow-line fit.

    Runs off the control loop so the 200 Hz controller never waits on a frame,
    and uses the CSI camera rather than the RealSense, which yolo_server.py
    already owns. Any failure here leaves the taxi steering from the roadmap
    alone rather than stopping it: lane trim is an improvement, not a
    dependency.
    """

    def __init__(self, rateHz):
        self.rateHz = max(1.0, rateHz)
        self._lock = Lock()
        self._fit = (lane_vision.NOMINAL_SLOPE, lane_vision.NOMINAL_INTERCEPT,
                     0, 0.0)
        self._cameras = None
        self._thread = None
        self.failure = None

    def start(self):
        try:
            self._cameras = QCarCameras(
                frameWidth=lane_vision.IMAGE_WIDTH,
                frameHeight=lane_vision.IMAGE_HEIGHT,
                frameRate=self.rateHz,
                enableFront=True)
        except Exception as e:
            self.failure = e
            print('Lane camera unavailable (%s); steering from the roadmap '
                  'only.' % e)
            return False
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()
        return True

    def _worker(self):
        rowLo, rowHi = lane_vision.CROP_ROWS
        colLo, colHi = lane_vision.CROP_COLS
        period = 1.0 / self.rateHz
        while not KILL_THREAD:
            loopStart = time.time()
            try:
                self._cameras.readAll()
                frame = self._cameras.csiFront.imageData
                if frame is not None and frame.size:
                    crop = frame[rowLo:rowHi, colLo:colHi]
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    binary = ImageProcessing.binary_thresholding(
                        frame=hsv,
                        lowerBounds=lane_vision.HSV_LOWER,
                        upperBounds=lane_vision.HSV_UPPER)
                    pixels = int(np.count_nonzero(binary))
                    slope, intercept = \
                        ImageProcessing.find_slope_intercept_from_binary(
                            binary=binary / 255)
                    with self._lock:
                        self._fit = (slope, intercept, pixels, time.time())
            except Exception as e:
                # A dropped frame must never take the car down.
                self.failure = e
                with self._lock:
                    self._fit = (lane_vision.NOMINAL_SLOPE,
                                 lane_vision.NOMINAL_INTERCEPT, 0, time.time())
            elapsed = time.time() - loopStart
            time.sleep(max(0.0, period - elapsed))

    def latest(self):
        """(slope, intercept, pixelCount) if the fit is fresh, else no detection."""
        with self._lock:
            slope, intercept, pixels, stamp = self._fit
        if stamp <= 0.0 or (time.time() - stamp) > LANE_MAX_AGE:
            return lane_vision.NOMINAL_SLOPE, lane_vision.NOMINAL_INTERCEPT, 0
        return slope, intercept, pixels

    def terminate(self):
        if self._cameras is not None:
            try:
                self._cameras.terminate()
            except Exception:
                pass


def build_dispatcher():
    """Chooses the pickup and drop-off nodes for each ride.

    By default every ride draws a new pair at random from this car's node pool,
    the way the virtual taxi did. Pools are disjoint per node_configuration, so
    a second car can be added without both taxis chasing the same fare.
    """
    if args.fixed:
        pickup, dropoff = args.pickup, args.dropoff
        return pickup, dropoff, (lambda: (pickup, dropoff))

    pool = taxi_mission.PICKUP_POOLS.get(
        args.node_configuration, taxi_mission.PICKUP_POOLS[0])

    def dispatch():
        pickup = random.choice(pool)
        dropoff = random.choice([n for n in pool if n != pickup])
        return pickup, dropoff

    # Seed the first ride so the startup banner shows a real destination.
    return pool[0], pool[-1], dispatch


def controlLoop():
    global KILL_THREAD

    roadmap = SDCSRoadMap(leftHandTraffic=False)
    pickup, dropoff, dispatch = build_dispatcher()

    mission = taxi_mission.TaxiMission(
        taxi_mission.RouteManager(roadmap),
        pickupNode=pickup,
        dropoffNode=dropoff,
        dispatch=dispatch,
        cruiseSpeed=args.cruise,
    )
    follower = taxi_mission.RouteFollower(roadmap)
    lamps = taxi_mission.SignalLamps()
    speedController = taxi_mission.SpeedController()

    runLog = RunLog(args.log) if args.log else None
    laneTrim = lane_vision.LaneTrim(nominalSteering=args.lane_nominal)
    survey = LaneSurvey() if args.lane_survey else None
    wantCamera = args.lane_trim or args.lane_survey
    laneCamera = LaneCamera(args.lane_rate) if wantCamera else None
    if laneCamera is not None and not laneCamera.start():
        laneCamera = None

    initialPose = roadmap.get_node_pose(taxi_mission.HUB_NODE).squeeze()

    #region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    yolo = YOLOReceiver()
    yoloDrive = YOLODriveLogic(pulseLength=controllerUpdateRate*3)
    ekf = QCarEKF(x_0=initialPose)
    gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
    #endregion

    with qcar, gps, yolo:
        if calibrate:
            return

        # Seed the estimator from a real GPS fix before commanding anything, so
        # the first steering command is not computed from the assumed pose.
        new = False
        while not new and not KILL_THREAD:
            new = gps.readGPS()
        if KILL_THREAD:
            return
        pose = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
        print('Taxi starting at pose [%.2f, %.2f, %.2f]'
              % (pose[0], pose[1], pose[2]))
        if args.fixed:
            print('Fixed rides: hub %d -> pickup %d -> drop-off %d'
                  % (taxi_mission.HUB_NODE, pickup, dropoff))
        else:
            pool = taxi_mission.PICKUP_POOLS.get(
                args.node_configuration, taxi_mission.PICKUP_POOLS[0])
            print('Random rides from hub %d, drawing nodes from %s'
                  % (taxi_mission.HUB_NODE, pool))
        if args.lane_survey:
            print('Lane survey: driving with the trim DISABLED and measuring '
                  'the lane signal. Result is printed on exit.')
        elif laneCamera is not None:
            print('Lane trim on: nominal %+.4f, +/-%.2f rad, %.0f Hz.'
                  % (laneTrim.nominalSteering, lane_vision.TRIM_LIMIT,
                     args.lane_rate))
        else:
            print('Lane trim off: steering from the roadmap and GPS only.')
        if args.heading_trim:
            print('Heading trim : %+.4f rad (%+.2f deg)'
                  % (args.heading_trim, np.degrees(args.heading_trim)))
        if args.steer_trim:
            print('Steering trim: %+.4f rad (%+.2f deg)'
                  % (args.steer_trim, np.degrees(args.steer_trim)))
        print('Running until stopped. Ctrl+C, or stop_all_cars.sh from the host.'
              if args.duration is None else
              'Running for %.0f s.' % args.duration)

        t0 = time.time()
        t = 0
        delta = 0
        pathSteering = 0.0
        trim = 0.0
        lastState = None

        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Read from sensors and update state estimates
            qcar.read()

            yolo.read()
            vGain = yoloDrive.check_yolo(yolo.stopSign,
                                         yolo.trafficlight,
                                         yolo.cars,
                                         yolo.yieldSign,
                                         yolo.person)

            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2]
                ])
                ekf.update([qcar.motorTach, delta], dt, y_gps,
                           qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, delta], dt, None,
                           qcar.gyroscope[2])

            x = ekf.x_hat[0, 0]
            y = ekf.x_hat[1, 0]
            # Corrected before anything uses it, so the front-axle projection
            # and the controller share one consistent heading.
            th = taxi_mission.apply_heading_trim(ekf.x_hat[2, 0],
                                                 args.heading_trim)
            p = (np.array([x, y])
                 + np.array([np.cos(th), np.sin(th)]) * args.preview)
            v = qcar.motorTach
            #endregion

            #region : Mission update
            command = mission.update((x, y), pathSteering, time.time())

            if command.routeChanged:
                follower.set_route(command.route, np.array([x, y, th]))
                speedController.reset()

            if command.state != lastState:
                if command.state == taxi_mission.TaxiState.NAV_TO_PICKUP:
                    print('\n[%s] Ride %d: pickup node %d, drop-off node %d'
                          % (time.strftime('%H:%M:%S'),
                             mission.ridesCompleted + 1,
                             mission.pickupNode, mission.dropoffNode))
                elif command.state == taxi_mission.TaxiState.IDLE_HUB:
                    print('\n[%s] Ride %d complete, back at the hub.'
                          % (time.strftime('%H:%M:%S'), mission.ridesCompleted))
                else:
                    print('\n[%s] %s  (goal %.2f m away)'
                          % (time.strftime('%H:%M:%S'), command.state,
                             command.distToGoal))
                lastState = command.state
            #endregion

            #region : Update controllers and write to car
            if t < startDelay or follower.controller is None:
                u = 0
                delta = 0
                pathSteering = 0.0
                trim = 0.0
            else:
                # The mission decides how fast this leg of the ride should go;
                # the YOLO gain independently slows or stops the car for signs,
                # lights, other cars and pedestrians. The lower one wins.
                v_ref = min(command.speed, args.cruise * vGain)

                # Roadmap steering first, then a bounded camera trim toward the
                # centre of the actual lane. The roadmap decides where to go;
                # vision only corrects for localisation bias within its lane.
                pathSteering = follower.update(p, th, v)

                trim = 0.0
                if laneCamera is not None:
                    slope, intercept, lanePixels = laneCamera.latest()
                    if survey is not None:
                        survey.add(slope, intercept, lanePixels, pathSteering)
                    if args.lane_trim:
                        trim = laneTrim.update(slope, intercept, lanePixels,
                                               pathSteering, dt)

                delta = taxi_mission.apply_steering_trim(
                    pathSteering + trim, args.steer_trim,
                    follower.controller.maxSteeringAngle)

                u = speedController.update(v, v_ref, dt)
                if v_ref == 0.0:
                    u = min(u, 0.0)

            if runLog is not None:
                runLog.add(t, command, x, y, th, v, u, delta, pathSteering,
                           trim, vGain)

            leds = lamps.update(command.state, delta, time.time())
            qcar.write(u, delta, leds)
            #endregion

            print('Pose [x %+.2f y %+.2f th %+.2f]  v %.2f ref %.2f gain %.2f '
                  'goal %.2f m  lane %s%+.3f   '
                  % (x, y, th, v,
                     min(command.speed, args.cruise * vGain),
                     vGain, command.distToGoal,
                     '*' if laneTrim.active else ' ', trim),
                  end='\r')

        qcar.read_write_std(throttle=0, steering=0)
        if runLog is not None:
            runLog.close()
        if laneCamera is not None:
            laneCamera.terminate()
        if survey is not None:
            survey.report()


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Setup and run experiment
if __name__ == '__main__':

    # sleep to wait for the YOLO server to start
    time.sleep(10)

    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
