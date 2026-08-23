#!/bin/bash
# Adds the SSH host keys of every QCar in config.txt to known_hosts.
# Only needs running once for a new car.

cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./config.sh
load_config || exit 1

IFS=',' read -ra IP_ARRAY <<< "$QCAR_IPS"

mkdir -p ~/.ssh

for ip in "${IP_ARRAY[@]}"; do
    ip=$(echo "$ip" | xargs)
    echo "Scanning $ip..."
    ssh-keyscan "$ip" >> ~/.ssh/known_hosts
done

echo "Done."
