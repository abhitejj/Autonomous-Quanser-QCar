#!/bin/bash
# Uploads qcar/*.py to every QCar in config.txt and starts the driving
# application (APP in config.txt) plus yolo_server.py on each one.
#
# The car must be sitting at its calibration spot (node 11) before you run
# this, and calibrate.sh must have been run at least once.
#
# Pass extra arguments straight through to the application, e.g.
#   ./run_all_cars.sh --random --cruise 0.4

cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./config.sh
load_config || exit 1

echo "Application : $APP $APP_ARGS"
echo "QCar(s)     : $QCAR_IPS"
echo "Observer    : $LOCAL_IP"
echo ""

python3 python/start.py \
    -ip "$QCAR_IPS" \
    -ipp "$PROBING_IP" \
    -rp "$REMOTE_PATH" \
    -ipl "$LOCAL_IP" \
    -w "$WIDTH" \
    -ht "$HEIGHT" \
    -a "$APP" \
    --app_args "$APP_ARGS $*"
