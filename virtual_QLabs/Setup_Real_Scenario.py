# -*- coding: utf-8 -*-
"""
Setup_Real_Scenario  --  FIXED VERSION
======================================
Fixes (vs. the original student code):
  [Gap 3] setup() now genuinely reuses the qlabs connection handle passed in
          by the caller instead of silently opening a second connection inside
          the function. It only connects on its own when qlabs=None.
  - Removed os.system('cls') (Windows-only, and it wiped the caller's logs).
  - destroy_all_spawned_actors / terminate_all_real_time_models are now
    controlled by an optional parameter so the caller decides whether to clear
    the world (defaults to clearing, preserving the original standalone behavior).
  - The real-time model launch is skipped entirely when RTMODELS_DIR is not
    set: pure qvl digital-twin mode no longer crashes or triggers the QLabs
    'file not found' dialog.
All actor locations / rotations / scales are identical to the original.
"""
import os

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.qcar_flooring import QLabsQCarFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.yield_sign import QLabsYieldSign
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight


def setup(qlabs=None,
          initialPosition=[-1.205, -0.83, 0.005],
          initialOrientation=[0, 0, -44.7],
          clearWorld=True):
    """Build the official ACC competition map and spawn the QCar2.

    :param qlabs: An already-open QuanserInteractiveLabs connection. Pass None to connect to localhost automatically.
    :param clearWorld: Whether to destroy all existing actors and stop real-time models first.
    :return: QLabsQCar2 instance (actorNumber=0), or None on failure.
    """
    # [Gap 3] Reuse the provided connection; only create one if none was given
    if qlabs is None:
        qlabs = QuanserInteractiveLabs()
        print("Connecting to QLabs...")
        try:
            qlabs.open("localhost")
            print("Connected to QLabs")
        except Exception:
            print("Unable to connect to QLabs")
            return None

    if clearWorld:
        qlabs.destroy_all_spawned_actors()
        try:
            QLabsRealTime().terminate_all_real_time_models()
        except Exception:
            pass

    # Workspace title
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('ACC Self Driving Car Competition', waitForConfirmation=True)

    ### Flooring
    x_offset = 0.13
    y_offset = 1.67
    hFloor = QLabsQCarFlooring(qlabs)
    hFloor.spawn_degrees([x_offset, y_offset, 0.001], rotation=[0, 0, -90])

    ### Walls
    hWall = QLabsWalls(qlabs)
    hWall.set_enable_dynamics(False)

    for y in range(5):
        hWall.spawn_degrees(location=[-2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001],
                            rotation=[0, 0, 0])
    for x in range(5):
        hWall.spawn_degrees(location=[-1.9 + x + x_offset, 3.05 + y_offset, 0.001],
                            rotation=[0, 0, 90])
    for y in range(6):
        hWall.spawn_degrees(location=[2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001],
                            rotation=[0, 0, 0])
    for x in range(4):
        hWall.spawn_degrees(location=[-0.9 + x + x_offset, -3.05 + y_offset, 0.001],
                            rotation=[0, 0, 90])
    hWall.spawn_degrees(location=[-2.03 + x_offset, -2.275 + y_offset, 0.001],
                        rotation=[0, 0, 48])
    hWall.spawn_degrees(location=[-1.575 + x_offset, -2.7 + y_offset, 0.001],
                        rotation=[0, 0, 48])

    # Spawn the QCar 2 at the Taxi Hub
    car2 = QLabsQCar2(qlabs)
    car2.spawn_id(actorNumber=0,
                  location=initialPosition,
                  rotation=initialOrientation,
                  scale=[.1, .1, .1],
                  configuration=0,
                  waitForConfirmation=True)

    # Cameras: birds-eye + edge, possess edge camera
    camera1 = QLabsFreeCamera(qlabs)
    camera1.spawn_degrees(location=[0.15, 1.7, 5], rotation=[0, 90, 0])

    camera2 = QLabsFreeCamera(qlabs)
    camera2.spawn_degrees(location=[-0.36 + x_offset, -3.691 + y_offset, 2.652],
                          rotation=[0, 47, 90])
    camera2.possess()

    # Stop signs
    myStopSign = QLabsStopSign(qlabs)
    myStopSign.spawn_degrees(location=[-1.5, 3.6, 0.006], rotation=[0, 0, -35],
                             scale=[0.1, 0.1, 0.1], waitForConfirmation=False)
    myStopSign.spawn_degrees(location=[-1.5, 2.2, 0.006], rotation=[0, 0, 35],
                             scale=[0.1, 0.1, 0.1], waitForConfirmation=False)
    myStopSign.spawn_degrees(location=[2.410, 0.206, 0.006], rotation=[0, 0, -90],
                             scale=[0.1, 0.1, 0.1], waitForConfirmation=False)
    myStopSign.spawn_degrees(location=[1.766, 1.697, 0.006], rotation=[0, 0, 90],
                             scale=[0.1, 0.1, 0.1], waitForConfirmation=False)

    # Roundabout signs
    myRoundaboutSign = QLabsRoundaboutSign(qlabs)
    myRoundaboutSign.spawn_degrees(location=[2.392, 2.522, 0.006], rotation=[0, 0, -90],
                                   scale=[0.1, 0.1, 0.1], waitForConfirmation=False)
    myRoundaboutSign.spawn_degrees(location=[0.698, 2.483, 0.006], rotation=[0, 0, -145],
                                   scale=[0.1, 0.1, 0.1], waitForConfirmation=False)
    myRoundaboutSign.spawn_degrees(location=[0.007, 3.973, 0.006], rotation=[0, 0, 135],
                                   scale=[0.1, 0.1, 0.1], waitForConfirmation=False)

    # Yield signs
    myYieldSign = QLabsYieldSign(qlabs)
    myYieldSign.spawn_degrees(location=[0.0, -1.3, 0.006], rotation=[0, 0, -180],
                              scale=[0.1, 0.1, 0.1], waitForConfirmation=False)
    myYieldSign.spawn_degrees(location=[2.4, 3.2, 0.006], rotation=[0, 0, -90],
                              scale=[0.1, 0.1, 0.1], waitForConfirmation=False)
    myYieldSign.spawn_degrees(location=[1.1, 2.8, 0.006], rotation=[0, 0, -145],
                              scale=[0.1, 0.1, 0.1], waitForConfirmation=False)
    myYieldSign.spawn_degrees(location=[0.49, 3.8, 0.006], rotation=[0, 0, 135],
                              scale=[0.1, 0.1, 0.1], waitForConfirmation=False)

    # Crosswalks
    myCrossWalk = QLabsCrosswalk(qlabs)
    myCrossWalk.spawn_degrees(location=[-2 + x_offset, -1.475 + y_offset, 0.01],
                              rotation=[0, 0, 0], scale=[0.1, 0.1, 0.075], configuration=0)
    myCrossWalk.spawn_degrees(location=[-0.5, 0.95, 0.006],
                              rotation=[0, 0, 90], scale=[0.1, 0.1, 0.075], configuration=0)
    myCrossWalk.spawn_degrees(location=[0.15, 0.32, 0.006],
                              rotation=[0, 0, 0], scale=[0.1, 0.1, 0.075], configuration=0)
    myCrossWalk.spawn_degrees(location=[0.75, 0.95, 0.006],
                              rotation=[0, 0, 90], scale=[0.1, 0.1, 0.075], configuration=0)
    myCrossWalk.spawn_degrees(location=[0.13, 1.57, 0.006],
                              rotation=[0, 0, 0], scale=[0.1, 0.1, 0.075], configuration=0)
    myCrossWalk.spawn_degrees(location=[1.45, 0.95, 0.006],
                              rotation=[0, 0, 90], scale=[0.1, 0.1, 0.075], configuration=0)

    # White guide lines
    mySpline = QLabsBasicShape(qlabs)
    mySpline.spawn_degrees(location=[2.21, 0.2, 0.006], rotation=[0, 0, 0],
                           scale=[0.27, 0.02, 0.001], waitForConfirmation=False)
    mySpline.spawn_degrees(location=[1.951, 1.68, 0.006], rotation=[0, 0, 0],
                           scale=[0.27, 0.02, 0.001], waitForConfirmation=False)
    mySpline.spawn_degrees(location=[-0.05, -1.02, 0.006], rotation=[0, 0, 90],
                           scale=[0.38, 0.02, 0.001], waitForConfirmation=False)

    # QUARC real-time model: only attempt to load when the RTMODELS_DIR env var exists.
    # In pure qvl digital-twin mode this is skipped entirely, avoiding the QLabs
    # 'file not found' dialog
    if 'RTMODELS_DIR' in os.environ:
        try:
            rtModel = os.path.normpath(
                os.path.join(os.environ['RTMODELS_DIR'],
                             'QCar2', 'QCar2_Workspace_studio'))
            QLabsRealTime().start_real_time_model(rtModel)
        except Exception:
            pass

    return car2


def terminate():
    """Stop the QCar2 workspace real-time model (if running)."""
    if 'RTMODELS_DIR' not in os.environ:
        return
    try:
        rtModel = os.path.normpath(
            os.path.join(os.environ['RTMODELS_DIR'],
                         'QCar2', 'QCar2_Workspace_studio'))
        QLabsRealTime().terminate_real_time_model(rtModel)
    except Exception:
        pass


def main():
    """Standalone run: build the map + traffic-light loop (matches the original behavior)."""
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    try:
        qlabs.open("localhost")
        print("Connected to QLabs")
    except Exception:
        print("Unable to connect to QLabs")
        return

    setup(qlabs=qlabs)

    trafficLight1 = QLabsTrafficLight(qlabs)
    trafficLight2 = QLabsTrafficLight(qlabs)
    trafficLight3 = QLabsTrafficLight(qlabs)
    trafficLight4 = QLabsTrafficLight(qlabs)

    trafficLight1.spawn_id_degrees(actorNumber=1, location=[0.6, 1.55, 0.006],
        rotation=[0, 0, 0], scale=[0.1, 0.1, 0.1], configuration=0, waitForConfirmation=False)
    trafficLight2.spawn_id_degrees(actorNumber=2, location=[-0.6, 1.28, 0.006],
        rotation=[0, 0, 90], scale=[0.1, 0.1, 0.1], configuration=0, waitForConfirmation=False)
    trafficLight3.spawn_id_degrees(actorNumber=3, location=[-0.37, 0.3, 0.006],
        rotation=[0, 0, 180], scale=[0.1, 0.1, 0.1], configuration=0, waitForConfirmation=False)
    trafficLight4.spawn_id_degrees(actorNumber=4, location=[0.75, 0.48, 0.006],
        rotation=[0, 0, -90], scale=[0.1, 0.1, 0.1], configuration=0, waitForConfirmation=False)

    import time
    flag = 0
    print('Starting Traffic Light Sequence')
    while True:
        TL = QLabsTrafficLight
        if flag == 0:
            trafficLight1.set_color(color=TL.COLOR_RED)
            trafficLight3.set_color(color=TL.COLOR_RED)
            trafficLight2.set_color(color=TL.COLOR_GREEN)
            trafficLight4.set_color(color=TL.COLOR_GREEN)
        elif flag == 1:
            trafficLight2.set_color(color=TL.COLOR_YELLOW)
            trafficLight4.set_color(color=TL.COLOR_YELLOW)
        elif flag == 2:
            trafficLight1.set_color(color=TL.COLOR_GREEN)
            trafficLight3.set_color(color=TL.COLOR_GREEN)
            trafficLight2.set_color(color=TL.COLOR_RED)
            trafficLight4.set_color(color=TL.COLOR_RED)
        elif flag == 3:
            trafficLight1.set_color(color=TL.COLOR_YELLOW)
            trafficLight3.set_color(color=TL.COLOR_YELLOW)
        flag = (flag + 1) % 4
        time.sleep(5)


if __name__ == '__main__':
    main()