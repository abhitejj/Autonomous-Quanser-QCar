#!/bin/bash
# Copy a run log off the QCar.
#
#   ./fetch_log.sh                 # fetches run1.csv to the repo root
#   ./fetch_log.sh run2.csv        # a different file
#   ./fetch_log.sh run2.csv /tmp   # to a different destination

cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./config.sh
load_config || exit 1

CAR_DIR="/home/nvidia/Documents/Quanser_Academic_Resources-dev-windows"
FILE="${1:-run1.csv}"

# Default destination: the repository root this checkout lives in
DEST="${2:-}"
if [ -z "$DEST" ]; then
    DEST="$PWD"
    while [ "$DEST" != "/" ] && [ ! -d "$DEST/0_libraries/python" ]; do
        DEST="$(dirname "$DEST")"
    done
    [ -d "$DEST/0_libraries/python" ] || DEST="$PWD"
fi

IP="$(echo "$CALIBRATE_IP" | xargs)"

echo "Fetching $FILE from $IP:$CAR_DIR"
echo "             to $DEST"
scp "nvidia@$IP:$CAR_DIR/$FILE" "$DEST/" || exit 1

echo ""
ls -lh "$DEST/$FILE"
echo "rows: $(( $(wc -l < "$DEST/$FILE") - 1 ))"
