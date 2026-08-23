"""Test path setup.

These tests exercise the taxi mission logic on a development machine, where the
Quanser libraries are not installed system-wide the way they are on the QCar.
Both the shared 0_libraries/python tree and the qcar/ application folder are put
on sys.path so `hal.products.mats` and `taxi_mission` import the same way they
do on the car.
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(TESTS_DIR)
QCAR_DIR = os.path.join(APP_DIR, 'qcar')


def _find_libraries_dir(start):
    d = start
    while True:
        candidate = os.path.join(d, '0_libraries', 'python')
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError('could not locate 0_libraries/python above ' + start)
        d = parent


for path in (QCAR_DIR, _find_libraries_dir(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
