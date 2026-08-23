import numpy as np
import pytest

from hal.products.mats import SDCSRoadMap

import lane_vision


@pytest.fixture
def roadmap():
    return SDCSRoadMap(leftHandTraffic=False)


# The fit the lab's lane follower expects when the car sits centred in its lane.
CENTRED_SLOPE = lane_vision.NOMINAL_SLOPE
CENTRED_INTERCEPT = lane_vision.NOMINAL_INTERCEPT
PLENTY = lane_vision.MIN_LANE_PIXELS * 4


class TestLaneSteeringFromFit:
    """The slope/intercept -> steering relation is the one calibrated in
    QCar2_task_lane_following_probe.py against the physical car and mat."""

    def test_centred_in_lane_asks_for_no_correction(self):
        steering = lane_vision.lane_steering_from_fit(CENTRED_SLOPE,
                                                      CENTRED_INTERCEPT)

        assert steering == pytest.approx(0.0, abs=1e-6)

    def test_drifting_right_of_the_line_steers_left(self):
        # Car drifts right -> the yellow centre line sits higher at the left
        # edge of the crop -> intercept rises.
        steering = lane_vision.lane_steering_from_fit(CENTRED_SLOPE,
                                                      CENTRED_INTERCEPT + 30)

        assert steering > 0.0  # positive steering is left on the QCar

    def test_drifting_left_of_the_line_steers_right(self):
        steering = lane_vision.lane_steering_from_fit(CENTRED_SLOPE,
                                                      CENTRED_INTERCEPT - 30)

        assert steering < 0.0


class TestLaneTrim:

    def trim(self, **kwargs):
        return lane_vision.LaneTrim(**kwargs)

    def settle(self, trim, slope, intercept, pixels, pathSteering=0.0,
               steps=60, dt=1/30.0):
        value = 0.0
        for _ in range(steps):
            value = trim.update(slope, intercept, pixels, pathSteering, dt)
        return value

    def test_no_trim_when_the_car_is_centred(self):
        trim = self.trim()

        value = self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT, PLENTY)

        assert abs(value) < 0.01

    def test_pushes_back_toward_the_lane_when_drifting(self):
        trim = self.trim()

        value = self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT + 30, PLENTY)

        assert value > 0.01

    def test_correction_is_bounded(self):
        trim = self.trim()

        value = self.settle(trim, CENTRED_SLOPE + 2.0, CENTRED_INTERCEPT + 500,
                            PLENTY)

        assert abs(value) <= lane_vision.TRIM_LIMIT + 1e-9

    def test_ignores_the_camera_when_too_few_lane_pixels_are_found(self):
        # find_slope_intercept_from_binary returns its nominal fit when it sees
        # nothing, which is a real steering value. Acting on that would apply a
        # constant phantom correction on any surface with no yellow line.
        trim = self.trim()

        value = self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT + 30,
                            pixels=lane_vision.MIN_LANE_PIXELS - 1)

        assert value == pytest.approx(0.0, abs=1e-6)

    def test_releases_the_correction_when_the_lane_is_lost(self):
        trim = self.trim()
        self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT + 30, PLENTY)

        value = self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT + 30,
                            pixels=0)

        assert abs(value) < 0.01

    def test_stands_down_while_the_car_is_turning(self):
        # Through intersections and the roundabout the centre line is not a
        # lane reference, and the roadmap controller must not be argued with.
        trim = self.trim()

        value = self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT + 30, PLENTY,
                            pathSteering=lane_vision.TURN_SUPPRESS_STEERING + 0.1)

        assert value == pytest.approx(0.0, abs=1e-6)

    def test_still_trims_during_gentle_curves(self):
        trim = self.trim()

        value = self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT + 30, PLENTY,
                            pathSteering=lane_vision.TURN_SUPPRESS_STEERING * 0.4)

        assert value > 0.01

    def test_ramps_in_rather_than_stepping(self):
        trim = self.trim()

        first = trim.update(CENTRED_SLOPE, CENTRED_INTERCEPT + 60, PLENTY,
                            0.0, 1/30.0)
        settled = self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT + 60,
                              PLENTY)

        assert abs(first) < abs(settled)

    def test_survives_a_nan_fit(self):
        # polyfit on a degenerate binary image can produce nan.
        trim = self.trim()

        value = trim.update(float('nan'), float('nan'), PLENTY, 0.0, 1/30.0)

        assert value == 0.0
        assert np.isfinite(value)

    def test_reports_whether_it_is_currently_acting(self):
        trim = self.trim()

        self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT + 30, PLENTY)
        acting = trim.active

        self.settle(trim, CENTRED_SLOPE, CENTRED_INTERCEPT + 30, pixels=0)

        assert acting is True
        assert trim.active is False


class TestLaneTrimCorrectsLocalisationBias:
    """The reason this feature exists: when the GPS frame is biased, the car
    tracks its path perfectly while sitting off-centre in the real lane. The
    camera sees the true lane, so the trim should pull it back -- and must not
    disturb anything when localisation is already correct."""

    WHEELBASE = 0.256
    DT = 0.02
    SPEED = 0.5
    PIXELS_PER_METRE = 300.0  # intercept units per metre of lateral offset

    def drive(self, roadmap, biasY, useTrim, horizon=120.0):
        import taxi_mission
        route = taxi_mission.RouteManager(roadmap).plan([taxi_mission.HUB_NODE, 4])
        controller = taxi_mission.SteeringController(route.waypoints)
        trimmer = lane_vision.LaneTrim()
        wp = route.waypoints

        x, y = wp[0, 0], wp[1, 0]
        th = np.arctan2(wp[1, 1] - wp[1, 0], wp[0, 1] - wp[0, 0])
        errors = []

        for step in range(int(horizon / self.DT)):
            believed = np.array([x, y + biasY])
            frontAxle = believed + np.array([np.cos(th), np.sin(th)]) * 0.2
            pathSteering = controller.update(frontAxle, th, self.SPEED)

            nearest = int(np.argmin(np.hypot(wp[0, :] - x, wp[1, :] - y)))
            following = min(nearest + 1, wp.shape[1] - 1)
            tangent = np.array([wp[0, following] - wp[0, nearest],
                                wp[1, following] - wp[1, nearest]])
            norm = np.linalg.norm(tangent)
            if norm < 1e-9:
                break
            tangent /= norm
            trueError = float(np.cross(tangent,
                                       np.array([x - wp[0, nearest],
                                                 y - wp[1, nearest]])))

            trim = 0.0
            if useTrim:
                intercept = (lane_vision.NOMINAL_INTERCEPT
                             - trueError * self.PIXELS_PER_METRE)
                trim = trimmer.update(lane_vision.NOMINAL_SLOPE, intercept,
                                      lane_vision.MIN_LANE_PIXELS * 3,
                                      pathSteering, self.DT)

            delta = float(np.clip(pathSteering + trim, -np.pi/6, np.pi/6))
            x += self.SPEED * np.cos(th) * self.DT
            y += self.SPEED * np.sin(th) * self.DT
            th += self.SPEED * np.tan(delta) / self.WHEELBASE * self.DT
            th = np.arctan2(np.sin(th), np.cos(th))

            if step * self.DT > 8.0:
                errors.append(abs(trueError))
            if nearest >= wp.shape[1] - 30:
                break

        return float(np.mean(errors))

    def test_reduces_lane_offset_caused_by_a_biased_pose_estimate(self, roadmap):
        bias = 0.15  # m of localisation error

        without = self.drive(roadmap, bias, useTrim=False)
        with_trim = self.drive(roadmap, bias, useTrim=True)

        # The margin is modest because the roadmap controller already tracks
        # well; the trim only removes what a biased pose estimate adds.
        assert with_trim < 0.85 * without

    def test_does_not_degrade_tracking_when_localisation_is_correct(self, roadmap):
        without = self.drive(roadmap, 0.0, useTrim=False)
        with_trim = self.drive(roadmap, 0.0, useTrim=True)

        assert with_trim <= without + 0.005


class TestSurveyedNominal:
    """The lab's nominal marks a different lateral position than the roadmap's
    lane centre. Trimming toward it shoves the car sideways forever, so the
    trim's zero point has to be surveyed on the actual car."""

    def test_no_correction_where_this_car_normally_sits(self):
        # Whatever the camera reads while the roadmap drives the lane properly
        # is by definition "centred" for this car.
        offset = 40.0
        settled = lane_vision.lane_steering_from_fit(CENTRED_SLOPE,
                                                     CENTRED_INTERCEPT + offset)
        trim = lane_vision.LaneTrim(nominalSteering=settled)

        value = 0.0
        for _ in range(90):
            value = trim.update(CENTRED_SLOPE, CENTRED_INTERCEPT + offset,
                                PLENTY, 0.0, 1/30.0)

        assert abs(value) < 0.005

    def test_still_corrects_departures_from_the_surveyed_nominal(self):
        offset = 40.0
        settled = lane_vision.lane_steering_from_fit(CENTRED_SLOPE,
                                                     CENTRED_INTERCEPT + offset)
        trim = lane_vision.LaneTrim(nominalSteering=settled)

        value = 0.0
        for _ in range(90):
            value = trim.update(CENTRED_SLOPE, CENTRED_INTERCEPT + offset + 40,
                                PLENTY, 0.0, 1/30.0)

        assert value > 0.01

    def test_an_unsurveyed_nominal_is_what_pushed_the_car_off_lane(self):
        # Regression guard for the bug this replaced: with the wrong zero point
        # the trim applies a permanent one-sided correction.
        mismatch = 40.0
        wrong = lane_vision.LaneTrim(nominalSteering=0.0)

        value = 0.0
        for _ in range(90):
            value = wrong.update(CENTRED_SLOPE, CENTRED_INTERCEPT + mismatch,
                                 PLENTY, 0.0, 1/30.0)

        assert value > 0.01  # a constant sideways shove, which is the failure
