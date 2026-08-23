"""
pose_check.py

Measures localisation bias. Read-only: it never writes throttle or steering, so
the car cannot move while this runs.

Park the car on a node whose position you know, run this, and compare what the
GPS reports against what that node actually is. A consistent offset is the bias
carried by every position the taxi navigates to, and it moves the whole driven
trajectory sideways regardless of how well the controller tracks its path.

    python3 pose_check.py --node 11

The car MUST start on the calibration spot: calibrationPose is the matcher's initial
guess as well as its frame origin, so starting anywhere else does not measure a
bias, it just fails to converge. A heading error near 180 degrees, or position
noise above a few centimetres, means exactly that -- discard the reading.

To check a second node, do not restart this at that node. Start on the
calibration spot with
--monitor and push the car there with the GPS still running, so the matcher
tracks continuously:

    python3 pose_check.py --monitor
"""

import argparse
import time

import numpy as np

from pal.products.qcar import QCarGPS
from hal.products.mats import SDCSRoadMap

import taxi_mission

parser = argparse.ArgumentParser(prog='Localisation bias check')
parser.add_argument('-n', '--node', type=int, default=None,
                    help='node the car is parked on (default: nearest)')
parser.add_argument('-s', '--samples', type=int, default=200,
                    help='GPS fixes to average (default: 200)')
parser.add_argument('--timeout', type=float, default=30.0,
                    help='seconds to wait for those fixes (default: 30)')
parser.add_argument('-m', '--monitor', action='store_true',
                    help='track continuously: start on node 11, then move the '
                         'car by hand and read the offset at any other node')
args = parser.parse_args()

# Must match taxi_control.py, since this is the declaration under test.
calibrationPose = [0, 2.000, -np.pi/2]


def collect(gps, wanted, timeout):
    positions, headings = [], []
    deadline = time.time() + timeout
    while len(positions) < wanted and time.time() < deadline:
        if gps.readGPS():
            positions.append([gps.position[0], gps.position[1]])
            headings.append(gps.orientation[2])
        time.sleep(0.005)
    return np.array(positions), np.array(headings)


def monitor(gps, roadmap):
    """Print live pose and the offset to the nearest node until Ctrl+C."""
    print('Tracking. Push the car slowly to another node and read the offset '
          'there. Ctrl+C to stop.\n')
    lastPrint = 0.0
    while True:
        if not gps.readGPS():
            time.sleep(0.005)
            continue
        now = time.time()
        if now - lastPrint < 0.5:
            continue
        lastPrint = now
        pose = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
        node = (args.node if args.node is not None
                else taxi_mission.nearest_node(roadmap, pose[0], pose[1],
                                              heading=pose[2]))
        nodePose = np.asarray(roadmap.get_node_pose(node)).ravel()
        offset = taxi_mission.pose_offset(pose[:2], nodePose)
        print('  reports x %+.3f y %+.3f th %+6.1f | node %2d | '
              'dx %+.3f dy %+.3f | across %+.3f  along %+.3f   '
              % (pose[0], pose[1], np.degrees(pose[2]), node,
                 offset.dx, offset.dy, offset.lateral, offset.longitudinal),
              end='\r')


def main():
    roadmap = SDCSRoadMap(leftHandTraffic=False)

    print('Starting GPS with calibrationPose %s' % calibrationPose)
    print('The car will NOT move. Keep it still on its node.\n')
    gps = QCarGPS(initialPose=calibrationPose, calibrate=False)

    with gps:
        if args.monitor:
            try:
                monitor(gps, roadmap)
            except KeyboardInterrupt:
                print('\nStopped.')
            return

        positions, headings = collect(gps, args.samples, args.timeout)

        if len(positions) < 10:
            print('Only %d GPS fixes in %.0f s. Is qcarLidarToGPS running, and '
                  'are angles_new.mat / distance_new.mat present?'
                  % (len(positions), args.timeout))
            return

        reported = positions.mean(axis=0)
        spread = positions.std(axis=0)
        heading = float(np.arctan2(np.sin(headings).mean(),
                                   np.cos(headings).mean()))

        node = args.node
        if node is None:
            node = taxi_mission.nearest_node(roadmap, reported[0], reported[1],
                                         heading=heading)
            print('No --node given; nearest is node %d.\n' % node)

        nodePose = np.asarray(roadmap.get_node_pose(node)).ravel()
        offset = taxi_mission.pose_offset(reported, nodePose)
        headingError = float(np.arctan2(np.sin(heading - nodePose[2]),
                                        np.cos(heading - nodePose[2])))

        print('=== Localisation check at node %d ===' % node)
        print('  fixes averaged : %d  (noise %.3f, %.3f m)'
              % (len(positions), spread[0], spread[1]))
        print('  node truth     : x %+.3f  y %+.3f  heading %+.1f deg'
              % (nodePose[0], nodePose[1], np.degrees(nodePose[2])))
        print('  GPS reports    : x %+.3f  y %+.3f  heading %+.1f deg'
              % (reported[0], reported[1], np.degrees(heading)))
        print('  offset         : dx %+.3f  dy %+.3f   (%.3f m total)'
              % (offset.dx, offset.dy, offset.distance))
        print('  across lane    : %+.3f m' % offset.lateral)
        print('  along lane     : %+.3f m' % offset.longitudinal)
        print('  heading error  : %+.1f deg' % np.degrees(headingError))

        print('')
        if offset.distance < 0.05:
            print('  Localisation looks sound. If the car still leaves its '
                  'lane, the cause is tracking, not the pose estimate.')
        else:
            print('  %.0f cm of bias. The lane is about 27 cm wide, so this '
                  'moves the car %.0f%% of a lane width.'
                  % (offset.distance * 100,
                     100 * offset.distance / 0.27))
            print('  Every position the taxi drives to carries this offset.')
            if spread.max() > 0.05 or abs(np.degrees(headingError)) > 20:
                print('  BUT the noise and/or heading error are too large for '
                      'this to be a real measurement: the matcher has not '
                      'converged. That happens when the car is not on the '
                      'calibration node at startup. Restart on node 11.')
            elif abs(offset.dy) > 0.10 and abs(offset.dx) < 0.06:
                print('  A y-only offset of this size means the reference scan '
                      'was taken on node 11 but labelled [0, 2, -pi/2]. Set '
                      'calibrationPose to [0, 2.163, -pi/2]; the scan itself '
                      'is still valid, so no recalibration is needed.')


if __name__ == '__main__':
    main()
