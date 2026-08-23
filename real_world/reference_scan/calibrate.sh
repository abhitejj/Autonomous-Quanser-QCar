#!/bin/bash
# Captures the LiDAR reference scan used to localize the car.
#
# Place the calibrating QCar at (0.000, +2.000) facing -90 degrees -- on the
# x=0 road, 16 cm short of node 11 (+2.163) and 15 cm past node 12 (+1.850),
# squared up as accurately as you can manage. This is the pose the driving
# application declares as calibrationPose, and the physical spot, its heading
# and that declaration must all agree -- any difference becomes a constant
# offset, or a rotation, on every position the GPS reports for the whole map.
#
# Square the car up carefully: a heading error here rotates the entire frame,
# displacing distant nodes by roughly distance * sin(error). 1.3 degrees put
# node 2 out by 8 cm, about a third of a lane.
#
# Re-run this whenever the physical environment around the track changes.

cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ../config.sh
load_config ../config.txt || exit 1

echo "Calibrating from QCar $CALIBRATE_IP using $APP"
echo "Confirm the car is at (0.000, +2.000), heading -90 deg,"
echo "  squared up as accurately as you can -- heading error here rotates"
echo "  the whole map frame."
read -r -p "Press Enter to start calibration, or Ctrl+C to abort... "

python3 ../python/calibrate.py \
    -ip "$QCAR_IPS" \
    -cip "$CALIBRATE_IP" \
    -rp "$REMOTE_PATH" \
    -a "$APP"
