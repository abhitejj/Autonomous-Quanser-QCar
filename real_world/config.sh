#!/bin/bash
# Shared config loader for the Linux launch scripts.
# Sourced by run_all_cars.sh, stop_all_cars.sh, ssh_setup.sh and
# reference_scan/calibrate.sh so they all read config.txt the same way.

load_config() {
    local configFile="${1:-config.txt}"

    if [ ! -f "$configFile" ]; then
        echo "Config file $configFile not found!" >&2
        return 1
    fi

    REMOTE_PATH=$(grep "^REMOTE_PATH=" "$configFile" | cut -d '=' -f 2)
    QCAR_IPS=$(grep "^QCAR_IPS=" "$configFile" | cut -d '=' -f 2 | tr -d '[]')
    CALIBRATE_IP=$(grep "^CALIBRATE_IP=" "$configFile" | cut -d '=' -f 2)
    PROBING_IP=$(grep "^PROBING_IP=" "$configFile" | cut -d '=' -f 2)
    LOCAL_IP=$(grep "^LOCAL_IP=" "$configFile" | cut -d '=' -f 2)
    WIDTH=$(grep "^WIDTH=" "$configFile" | cut -d '=' -f 2)
    HEIGHT=$(grep "^HEIGHT=" "$configFile" | cut -d '=' -f 2)
    APP=$(grep "^APP=" "$configFile" | cut -d '=' -f 2)
    APP_ARGS=$(grep "^APP_ARGS=" "$configFile" | cut -d '=' -f 2-)

    # APP and APP_ARGS are optional, so an older config.txt still works
    APP="${APP:-vehicle_control.py}"
    APP_ARGS="${APP_ARGS:-}"

    return 0
}
