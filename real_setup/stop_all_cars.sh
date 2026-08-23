#!/bin/bash
# Shell script to stop the Python Script(s) and the QUARC models on the QCar

# Abort on unset variables (e.g. a key missing from config.txt)
set -u

# Run from the directory holding this script so the relative paths below
# resolve the same way they did when the .bat was launched on Windows
cd "$(dirname "$(readlink -f "$0")")"

# Path to the config file
configFile="/home/user/Documents/Quanser_Academic_Resources-dev-windows/5_research/sdcs/qcar2/hardware/applications/multi_vehicle_self_driving/config.txt"

# Read each line from the config file and
# set variable values according to config.txt
while IFS='=' read -r key value || [ -n "$key" ]; do
    key="$(echo "$key" | tr -d '[:space:]')"
    # Skip blank lines and comments
    case "$key" in ''|'#'*) continue ;; esac
    # Drop a trailing carriage return in case config.txt has CRLF endings
    value="${value%$'\r'}"
    printf -v "$key" '%s' "$value"
done < "$configFile"

# Remove [ and ]
QCAR_IPS="$(echo "$QCAR_IPS" | tr -d '[]')"

# Split the comma separated list of IPs into an array
IFS=',' read -ra QCAR_IP_LIST <<< "$QCAR_IPS"

# Pick the Python interpreter to use, override with e.g. PYTHON=python3.10
if [ -z "${PYTHON:-}" ]; then
    if command -v python3 > /dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

# Stopping the script(s) on every QCar
set -x
"$PYTHON" python/stop.py -ip "$QCAR_IPS"
set +x

# Stopping the QUARC models via the remote QUARC run-time manager
# The model name is quoted so the shell passes the wildcard through to quarc_run
for ip in "${QCAR_IP_LIST[@]}"; do
    ip="$(echo "$ip" | tr -d '[:space:]')"
    [ -z "$ip" ] && continue
    set -x
    quarc_run -q -Q -t "tcpip://$ip:17000" '*.rt-linux_qcar2'
    set +x
    sleep 1
done
