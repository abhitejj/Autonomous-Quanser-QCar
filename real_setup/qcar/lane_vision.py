"""
lane_vision.py

Vision lane keeping for the autonomous taxi.

The taxi steers by tracking SDCS roadmap waypoints against the scan-matching
GPS. That keeps it on the planned route, but only as accurately as the pose
estimate: any bias in localisation moves the whole trajectory sideways, and the
car sits off-centre in its lane while believing it is perfectly on path.

The CSI front camera measures the yellow centre line directly, which is an
independent, absolute observation of where the lane actually is. This module
turns that observation into a small bounded trim added to the roadmap steering,
so localisation bias is corrected without handing lane keeping over to vision.

The slope/intercept to steering relation and the HSV bounds are the ones
calibrated on the physical car in
  5_research/sdcs/qcar2/hardware/applications/lane_following/
      QCar2_task_lane_following_probe.py

Deliberately free of hardware imports: camera I/O lives in taxi_control.py.
"""

import numpy as np

# ===== Camera geometry, from the lab lane-following example
# The steering relation below is calibrated for exactly this capture size and
# crop. Changing either invalidates the constants.
IMAGE_WIDTH = 1640
IMAGE_HEIGHT = 820
CROP_ROWS = (524, 674)
CROP_COLS = (0, 820)

# HSV bounds isolating the yellow centre line.
HSV_LOWER = np.array([10, 50, 100])
HSV_UPPER = np.array([45, 255, 255])

# ===== Lane fit
# find_slope_intercept_from_binary returns this slope with intercept 0 when it
# finds nothing, so the nominal fit is not by itself evidence of a detection.
NOMINAL_SLOPE = 0.3419
NOMINAL_INTERCEPT = -5.0
SLOPE_GAIN = 1.5
INTERCEPT_GAIN = 1.0/150.0

# ===== Trim behaviour
MIN_LANE_PIXELS = 1500   # below this the fit is noise, not a lane
TRIM_LIMIT = 0.06        # rad, hard ceiling on the vision contribution
TRIM_GAIN = 0.35         # fraction of the lane follower's own steering to apply
TRIM_FILTER_HZ = 2.0     # low pass on the trim, in Hz
TURN_SUPPRESS_STEERING = 0.22  # rad of roadmap steering above which vision is
                               # ignored: through junctions and the roundabout
                               # the centre line is not a lane reference


# Measured nominal for THIS car. lane_steering_from_fit returns the lab's idea
# of "centred", which was calibrated for the lab's camera mount and its own
# desired lane offset -- not for the SDCS lane centre the roadmap drives. The
# difference is a constant, and applying it as a trim shoves the car sideways
# forever: 3 cm of mismatch already tracks worse than no trim at all, and pushes
# toward oncoming traffic. So the trim is taken relative to a value surveyed on
# the actual car (see --lane-survey in taxi_control.py) rather than assumed.
NOMINAL_STEERING = 0.0


def lane_steering_from_fit(slope, intercept):
    """Steering the lab lane follower would command for this line fit.

    Zero when the car is centred on its lane; positive steers left, matching
    the QCar convention.
    """
    return (SLOPE_GAIN * (slope - NOMINAL_SLOPE)
            + INTERCEPT_GAIN * (intercept - NOMINAL_INTERCEPT))


class LaneTrim:
    """Converts a lane fit into a bounded steering trim.

    Refuses to contribute whenever it should not be trusted: too few lane
    pixels, a degenerate fit, or the car turning hard enough that the centre
    line no longer describes the lane it is following. In all those cases the
    trim decays smoothly to zero rather than being dropped, so the steering
    never steps.
    """

    def __init__(self, gain=TRIM_GAIN, limit=TRIM_LIMIT,
                 minPixels=MIN_LANE_PIXELS,
                 turnSuppress=TURN_SUPPRESS_STEERING,
                 filterHz=TRIM_FILTER_HZ,
                 nominalSteering=None):
        self.nominalSteering = (NOMINAL_STEERING if nominalSteering is None
                                else nominalSteering)
        self.gain = gain
        self.limit = limit
        self.minPixels = minPixels
        self.turnSuppress = turnSuppress
        self.filterHz = filterHz

        self.trim = 0.0
        self.active = False
        self.laneSteering = 0.0

    def update(self, slope, intercept, pixelCount, pathSteering, dt):
        target = 0.0
        self.active = False

        usable = (pixelCount >= self.minPixels
                  and np.isfinite(slope) and np.isfinite(intercept)
                  and abs(pathSteering) <= self.turnSuppress)

        if usable:
            self.laneSteering = lane_steering_from_fit(slope, intercept)
            if np.isfinite(self.laneSteering):
                # Correct only the departure from where the line sits when this
                # car is driving its lane properly, never the absolute reading.
                error = self.laneSteering - self.nominalSteering
                target = float(np.clip(self.gain * error,
                                       -self.limit, self.limit))
                self.active = True
        else:
            self.laneSteering = 0.0

        # First order low pass, so gaining or losing the lane never steps the
        # wheel. alpha is derived from dt so the behaviour does not change with
        # camera frame rate.
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.filterHz * max(dt, 1e-6))
        self.trim += alpha * (target - self.trim)

        if not np.isfinite(self.trim):
            self.trim = 0.0
        return float(np.clip(self.trim, -self.limit, self.limit))
