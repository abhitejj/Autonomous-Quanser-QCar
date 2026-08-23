import sys
import paramiko
import argparse
from scp import SCPClient
import subprocess
import os
import glob
import time

parser = argparse.ArgumentParser(description="Run a demo script on a remote server.")
parser.add_argument('-ip', '--qcar_ips', type=str, required=True, help="IP addresses of QCars")
parser.add_argument('-rp','--remote_path',type=str,required=True, help="Remote path on QCar")
parser.add_argument('-ipp','--probing_ip', type=str, required=True, help="IP addresses of QCar to probe")
parser.add_argument('-ipl','--local_ip', type=str, required=True,help="IP addresses of local machine")
parser.add_argument('-w','--width', default=320, help="width of to image to be displayed in the observer")
parser.add_argument('-ht','--height', default=200, help="height of to image to be displayed in the observer")
parser.add_argument('-a','--app', default="vehicle_control.py", help="application to run on each QCar")
parser.add_argument('--app_args', default="", help="extra arguments passed through to the application")
args = parser.parse_args()

QCAR_IPS = args.qcar_ips.split(",")  # Comma-separated list of IPs
LOCAL_IP = args.local_ip
PROBING_IP = args.probing_ip
WIDTH = args.width
HEIGHT = args.height
REMOTE_PATH = args.remote_path
APP = args.app
APP_ARGS = args.app_args
USERNAME = "nvidia"
PASSWORD = "nvidia"
LOCAL_SCRIPTS_PATH = "../qcar"
LOCAL_OBSERVER_PATH = "python/observer.py"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_PATH = os.path.normpath(os.path.join(CURRENT_DIR,LOCAL_SCRIPTS_PATH))

# --- Helper to give the observer access to the pal/hal libraries ---
def observer_environment():
    """Environment for observer.py with the repo's python libraries on PYTHONPATH.

    pal and hal are installed system-wide on the QCar but not on a host machine,
    where they only exist inside the resources tree. Without this the observer
    dies with ModuleNotFoundError while the cars carry on driving.
    """
    env = os.environ.copy()
    directory = CURRENT_DIR
    while True:
        candidate = os.path.join(directory, '0_libraries', 'python')
        if os.path.isdir(candidate):
            existing = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = (candidate + os.pathsep + existing) if existing \
                else candidate
            return env
        parent = os.path.dirname(directory)
        if parent == directory:
            print("Warning: could not locate 0_libraries/python; "
                  "the observer window may not start.")
            return env
        directory = parent


# --- Helper to create SSH + SCP connections ---
def create_ssh_and_scp(ip):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=USERNAME, password=PASSWORD)
    scp = SCPClient(ssh.get_transport())
    return ssh, scp

for ip in QCAR_IPS:
    print(f"\n Starting QCar: {ip}")

    # Upload Python scripts
    ssh, scp = create_ssh_and_scp(ip)

    # .sh as well as .py, so run_car.sh lands on the car too
    local_files = (glob.glob(os.path.normpath(os.path.join(SCRIPTS_PATH, "*.py")))
                   + glob.glob(os.path.normpath(os.path.join(SCRIPTS_PATH, "*.sh"))))
    for file in local_files:
        scp.put(file, REMOTE_PATH)
    print(f"[{ip}] Uploaded QCar scripts.")

    # Set PROBING flag
    is_probing = (ip == PROBING_IP)
    probing_flag = "True" if is_probing else "False"

    # Start observer.py locally if this is the probing QCar
    if is_probing:
        print(f"[{ip}] Starting observer.py locally (probing mode).")
        # sys.executable, not "python": on Linux hosts that name often does not
        # exist, and the observer would silently fail to start.
        # The observer needs pal/hal, which live in the repo's 0_libraries tree
        # rather than site-packages, so put them on the child's import path.
        subprocess.Popen([sys.executable, LOCAL_OBSERVER_PATH],
                         env=observer_environment())

    # YOLO server first, then the driving application. This is the order
    # run_car.sh uses on the car itself: the controller connects to the YOLO
    # stream on startup, so bringing the server up first avoids a race.
    cmd_yolo = (
        f"cd {REMOTE_PATH} && "
        f"python3 yolo_server.py -p {probing_flag} -i {LOCAL_IP} -w {WIDTH} -ht {HEIGHT} "
        f"> yolo.log 2>&1 &"
    )
    ssh.exec_command(cmd_yolo)
    print(f"[{ip}] Started yolo_server.py.")

    time.sleep(1)

    # Redirect to a log: launching over SSH with & discards stdout, so a crash
    # here is otherwise completely silent from the host.
    cmd_vehicle = (f"cd {REMOTE_PATH} && "
                   f"python3 {APP} {APP_ARGS} > app.log 2>&1 &")
    ssh.exec_command(cmd_vehicle)
    print(f"[{ip}] Started {APP}.  (log: {REMOTE_PATH}/app.log)")

    ssh.close()
    scp.close()

    time.sleep(3) 

input("\n All QCars started, press Enter to exit...")  # Wait for user input before exiting
