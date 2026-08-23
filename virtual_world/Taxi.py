import sys
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import math
import random
import threading
from collections import deque

import numpy as np
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

SHOW_WINDOWS = os.environ.get("ACC_SHOW_WINDOWS", "1") == "1"
if os.name != "nt" and not os.environ.get("DISPLAY"):
    SHOW_WINDOWS = False

SHOW_MAP_DEBUG = os.environ.get("ACC_SHOW_MAP_DEBUG", "1") == "1"
MAP_DEBUG_MAX_TRACE = int(os.environ.get("ACC_MAP_DEBUG_MAX_TRACE", "2500"))
MAP_DEBUG_UPDATE_PERIOD = float(os.environ.get("ACC_MAP_DEBUG_UPDATE_PERIOD", "0.05"))

# Control/visualization safety switches. Vision trim is useful in some demos,
# but on this small ACC map it can fight the roadmap controller and make the car
# wag left-right. Keep it off by default; set ACC_USE_VISION_TRIM=1 only when
# you explicitly want lane-boundary steering trim.
USE_VISION_TRIM = os.environ.get("ACC_USE_VISION_TRIM", "0") == "1"
ROUTE_START_HOLD = float(os.environ.get("ACC_ROUTE_START_HOLD", "0.55"))
ROUTE_START_RAMP = float(os.environ.get("ACC_ROUTE_START_RAMP", "2.20"))
START_STEER_LIMIT = float(os.environ.get("ACC_START_STEER_LIMIT", "0.30"))
START_STEER_LIMIT_TIME = float(os.environ.get("ACC_START_STEER_LIMIT_TIME", "2.40"))


CURVE_SLOWDOWN = os.environ.get("ACC_CURVE_SLOWDOWN", "1") == "1"
CURVE_SPEED_CAP = float(os.environ.get("ACC_CURVE_SPEED_CAP", "0.32"))
SHARP_CURVE_SPEED_CAP = float(os.environ.get("ACC_SHARP_CURVE_SPEED_CAP", "0.24"))
POST_TURN_SPEED_CAP = float(os.environ.get("ACC_POST_TURN_SPEED_CAP", "0.28"))
POST_TURN_RECOVERY_TIME = float(os.environ.get("ACC_POST_TURN_RECOVERY_TIME", "1.25"))
CURVE_STEER_THRESHOLD = float(os.environ.get("ACC_CURVE_STEER_THRESHOLD", "0.24"))
SHARP_CURVE_STEER_THRESHOLD = float(os.environ.get("ACC_SHARP_CURVE_STEER_THRESHOLD", "0.42"))


# Random drop-off node per ride (False = fixed Node 22, matching the official single-ride scenario)
RANDOM_DROPOFF = os.environ.get("ACC_RANDOM_DROPOFF", "0") == "1"

CRUISE_SPEED   = 0.5    # m/s  cruise speed
DECEL_RADIUS   = 0.6    # m    distance at which terminal deceleration begins
STOP_RADIUS    = 0.15   # m    distance treated as 'arrived', brake to stop
PICKUP_SWITCH  = 0.35   # m    NAV_TO_PICKUP -> LOADING switch threshold; 0.20 was too strict in QLabs
MIN_SPEED      = 0.12   # m/s  minimum speed during deceleration so the car can still creep
TL_DETECT_DIST = 2.0    # m    traffic-light look-ahead distance (original 4.0 leaked across intersections)
OBSTACLE_HOLD  = True   # stop and wait when LiDAR detects an in-lane obstacle
LOOKAHEAD      = 0.45   # m  Pure Pursuit lookahead (0.6 cut corners too hard on this 1:10 map)
# Forward obstacle-detection corridor (vehicle frame): narrowed to hug the car
# width so the diagonal walls next to the taxi hub don't false-trigger
OBS_X_MIN      = 0.25   # m  ignore very-near bumper/ground/curb returns that cause stop-creep jitter
OBS_X_MAX      = 0.45   # m  corridor far end (straight driving); reduced to avoid roadside signs/curbs
OBS_X_MAX_TURN = 0.26   # m  shorter corridor while turning/roundabout to avoid island/wall false positives
OBS_Y_HALF     = 0.12   # m  corridor half-width; narrowed to reduce curb/island/sign false triggers
HOLD_CREEP_AFTER = 4.0  # s  if held this long and clearance allows, creep along the path
CREEP_CLEARANCE  = 0.30 # m  minimum frontal clearance required to creep
YIELD_SPEED      = 0.30 # m/s  yield means slow/prepare, not full stop
ROUNDABOUT_SPEED = 0.28 # m/s  roundabout entry/inside speed cap, not full stop
OBSTACLE_EMERGENCY_DIST = 0.16 # m  brake in sign/intersection zones only if something is truly at the bumper
OBS_MIN_CLUSTER_POINTS   = 3    # LiDAR points required in corridor before a normal obstacle can be considered real
OBS_CONFIRM_FRAMES       = 3    # consecutive dense LiDAR frames required before holding; emergency still brakes instantly
CONFIRM_FRAMES = 5      # consecutive frames required to confirm a passenger
STOP_SIGN_FULL_STOP_TIME = 2.0  # s  hold a real full stop; keep holding even if YOLO misses a frame
STOP_SIGN_COOLDOWN       = 4.0  # s  prevents sign-detection flicker from repeatedly stopping the car
STOP_SIGN_AREA_TRIGGER   = 2500 # px^2 stop only when the stop sign is close enough; prevents stopping too early
STOP_SIGN_ARM_AREA       = 1400 # px^2 pre-arm latch; if the sign disappears after this, treat it as reaching the line
STOP_SIGN_LOST_TIME      = 0.35 # s  armed sign-loss window used to catch close pass-by/out-of-frame events
LOOKAHEAD_MIN            = 0.38 # m  adaptive Pure Pursuit lookahead at very low speed
LOOKAHEAD_SPEED_GAIN     = 0.50 # s  Ld = LOOKAHEAD_MIN + LOOKAHEAD_SPEED_GAIN * speed
LOOKAHEAD_MAX            = 0.65 # m  cap adaptive lookahead so the car still respects tight map geometry
TL_MEMORY_TIME           = 0.8  # s  ignore stale traffic-light colors from previous frames
DIRECT_TO_KNOWN_PASSENGER = True  # route directly to the spawned passenger instead of YOLO patrol rerouting
USE_PERSON_YOLO_REROUTE  = False # disable passenger bbox reroute; it caused false [node,node] routes/crashes

# Passenger boarding/alighting animation (QLabsPerson.move_to AI pathfinding)
PASSENGER_ANIMATION = False  # passenger AI walking disabled: instant board/alight only
BOARD_TIMEOUT       = 10.0   # s  max time to wait for the passenger to reach the door
WALKAWAY_TIME       = 0.2    # s  no walk-away animation when PASSENGER_ANIMATION=False
DOOR_OFFSET         = 0.28   # m  door position: lateral offset to the car's right
SIDEWALK_OFFSET     = 0.9    # m  lateral distance to the sidewalk the passenger walks to

# Pure-qvl mode: skip the QUARC real-time model (digital-twin teaching mode)
os.environ.setdefault("RTMODELS_DIR", "/tmp")
import qvl.real_time
qvl.real_time.QLabsRealTime.start_real_time_model = lambda self, path: None

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.traffic_light import QLabsTrafficLight
from qvl.person import QLabsPerson

from core.roadmap import ACCRoadMap2
import Setup_Real_Scenario

# ---- [Gap 1] Optional TrajectoryPlanner import ----
TrajectoryPlanner = None
try:
    from utils.path_plan import TrajectoryPlanner as _TP
    TrajectoryPlanner = _TP
    print(f"[INIT] TrajectoryPlanner loaded (vision trim enabled={USE_VISION_TRIM})")
except Exception:
    print("[INIT] utils/path_plan.TrajectoryPlanner not found -> using HSV lane-detection fallback")

# ---- [Gap 2] YOLO weight path resolution ----
def find_yolo_weights():
    """Search for the traffic-sign YOLO weights by priority: env var > common relative paths. Returns None if not found."""
    env = os.environ.get("ACC_YOLO_WEIGHTS")
    if env and os.path.isfile(env):
        return env
    candidates = [
        os.path.join(BASE_DIR, "data.pt"),
        os.path.join(BASE_DIR, "weights", "data.pt"),
        os.path.join(BASE_DIR, "models", "data.pt"),
        os.path.join(BASE_DIR, "Detection", "data.pt"),
        os.path.join(BASE_DIR, "..", "Competition code", "Detection", "data.pt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False
    print("[INIT] ultralytics not installed -> all YOLO perception disabled (pip install ultralytics)")


def imshow_safe(title, img):
    if SHOW_WINDOWS:
        cv2.imshow(title, img)


# ------------------------------------------------------------------
# 1. ENUMS & CONSTANTS
# ------------------------------------------------------------------
class FSMState:
    STATE_IDLE_HUB          = 0
    STATE_NAV_TO_PICKUP     = 1
    STATE_PASSENGER_LOADING = 2
    STATE_NAV_TO_DROPOFF    = 3
    STATE_RETURN_TO_HUB     = 4
    STATE_PATROLLING        = 5

HUB_NODE     = 10
DROPOFF_NODE = 22
VALID_NODES  = [0, 2, 4, 8, 10, 14, 20, 23]

# LED state colors (0.0-1.0 floats per the qvl set_led_strip_uniform spec)
LED_DRIVING = [1.0, 1.0, 1.0]   # white: normal driving
LED_PICKUP  = [0.0, 1.0, 0.0]   # green: picking up
LED_DROPOFF = [0.0, 0.3, 1.0]   # blue: dropping off
LED_IDLE    = [1.0, 0.8, 0.0]   # yellow: idle at hub


# ------------------------------------------------------------------
# 2. PERCEPTION
# ------------------------------------------------------------------
class PerceptionSystem:

    def __init__(self, qcar):
        self.qcar = qcar
        self.last_steering = 0.0
        self.last_error = 0.0
        self.smoothed_center_x = 320.0

        # [Gap 1] planner is optional
        self.planner = TrajectoryPlanner() if TrajectoryPlanner else None
        self.lane_boundaries_xz = []

        # [Gap 2] traffic-sign model: degrade gracefully if missing
        self.yolo_model = None
        self.person_yolo_model = None
        if HAS_YOLO:
            weights = find_yolo_weights()
            if weights:
                print(f"[INIT] Traffic-sign YOLO weights: {weights}")
                self.yolo_model = YOLO(weights)
            else:
                print("[INIT] data.pt not found -> sign/traffic-light vision disabled "
                      "(put the weights in this folder or set ACC_YOLO_WEIGHTS)")
            # yolov8n is auto-downloaded and cached by ultralytics
            try:
                self.person_yolo_model = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"[INIT] Failed to load yolov8n.pt ({e}) -> passenger vision disabled")

        # Shared state for background inference
        self.latest_image = None
        self.latest_detected_objects = []
        self.latest_tl_state = None
        self.latest_boxes = []
        self.latest_sign_info = {}
        self.latest_person_bbox = None
        self.person_streak = 0          # consecutive-confirmation frame counter
        self.vision_lock = threading.Lock()

        if self.yolo_model or self.person_yolo_model:
            self.vision_thread = threading.Thread(target=self._yolo_worker, daemon=True)
            self.vision_thread.start()

    # ---------------- Background YOLO thread ----------------
    def _yolo_worker(self):
        while True:
            img = None
            with self.vision_lock:
                if self.latest_image is not None:
                    img = self.latest_image.copy()

            if img is not None:
                h, w = img.shape[:2]
                crop_img = img[:int(h * 0.95), :]

                detected_objects, boxes_to_draw = [], []
                tl_state, person_bbox = None, None
                tl_candidates = []  # choose the most relevant traffic light instead of RED-first priority
                sign_candidates = {}  # best valid sign box by label, used for close-stop gating

                # ---- Passenger detection (COCO person) + consecutive-frame confirmation ----
                raw_person = None
                if self.person_yolo_model:
                    for result in self.person_yolo_model(crop_img, conf=0.25, verbose=False):
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            if int(box.cls[0]) == 0:
                                px1, py1, px2, py2 = map(int, box.xyxy[0])
                                raw_person = (px1, py1, px2, py2)
                                boxes_to_draw.append((px1, py1, px2, py2, "Person",
                                                      float(box.conf[0]), (255, 100, 100)))
                if raw_person is not None:
                    self.person_streak += 1
                    if self.person_streak >= CONFIRM_FRAMES:
                        detected_objects.append("PERSON")
                        person_bbox = raw_person
                else:
                    self.person_streak = 0

                # ---- Traffic signs / traffic lights ----
                if self.yolo_model:
                    for result in self.yolo_model(crop_img, conf=0.2, verbose=False):
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            label = self.yolo_model.names[cls_id]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            area = (x2 - x1) * (y2 - y1)

                            is_valid = True
                            if label in ("Stop", "Yield", "Roundabout"):
                                if confidence < 0.98:
                                    is_valid = False
                            elif label in ("Redlight", "Greenlight", "Yellowlight"):
                                if confidence < 0.85:
                                    is_valid = False
                            elif confidence < 0.60:
                                is_valid = False

                            if label in ("Redlight", "Greenlight", "Yellowlight"):
                                if area < 150:
                                    is_valid = False
                            elif area < 1000:
                                is_valid = False

                            if is_valid:
                                color = (0, 255, 255)
                                if label == "Stop":
                                    detected_objects.append("STOP");       color = (0, 0, 255)
                                    # Keep the largest valid Stop-sign box. We stop only when it is close enough;
                                    # otherwise the vehicle may stop too early, finish the cooldown, and then drive
                                    # through the actual stop line.
                                    best = sign_candidates.get("STOP")
                                    if best is None or area > best["area"]:
                                        sign_candidates["STOP"] = {
                                            "area": area, "conf": confidence,
                                            "center": (0.5 * (x1 + x2), 0.5 * (y1 + y2))
                                        }
                                elif label == "Yield":
                                    detected_objects.append("YIELD");      color = (180, 0, 180)
                                elif label == "Crosswalk":
                                    detected_objects.append("CROSSWALK");  color = (255, 200, 0)
                                elif label == "Redlight":
                                    detected_objects.append("TL_RED");     color = (0, 0, 255)
                                    cx = 0.5 * (x1 + x2)
                                    center_weight = max(0.2, 1.0 - abs(cx - w / 2.0) / (w / 2.0))
                                    tl_candidates.append((area * confidence * center_weight, "RED"))
                                elif label == "Yellowlight":
                                    detected_objects.append("TL_YELLOW");  color = (0, 255, 255)
                                    cx = 0.5 * (x1 + x2)
                                    center_weight = max(0.2, 1.0 - abs(cx - w / 2.0) / (w / 2.0))
                                    tl_candidates.append((area * confidence * center_weight, "YELLOW"))
                                elif label == "Greenlight":
                                    detected_objects.append("TL_GREEN");   color = (0, 255, 0)
                                    cx = 0.5 * (x1 + x2)
                                    center_weight = max(0.2, 1.0 - abs(cx - w / 2.0) / (w / 2.0))
                                    tl_candidates.append((area * confidence * center_weight, "GREEN"))
                                elif label == "Roundabout":
                                    detected_objects.append("ROUNDABOUT"); color = (0, 165, 255)
                                boxes_to_draw.append((x1, y1, x2, y2, label, confidence, color))
                            else:
                                boxes_to_draw.append((x1, y1, x2, y2, label, confidence,
                                                      (160, 160, 160)))

                # If multiple lights are visible, do not let any distant red light override
                # the lane-relevant green. Use the largest/most central traffic-light box.
                if tl_candidates:
                    tl_state = max(tl_candidates, key=lambda t: t[0])[1]

                with self.vision_lock:
                    self.latest_detected_objects = detected_objects
                    self.latest_tl_state = tl_state
                    self.latest_boxes = boxes_to_draw
                    self.latest_sign_info = sign_candidates
                    self.latest_person_bbox = person_bbox

            time.sleep(0.01)

    # ---------------- LiDAR ----------------
    # QLabs LiDAR angles increase CLOCKWISE (matching Quanser's physical LiDAR
    # convention). Vehicle-frame math convention is counter-clockwise, so the
    # bearing must be negated wherever the two meet. Getting this wrong mirrors
    # the point cloud left/right and samples passenger range on the wrong side.
    @staticmethod
    def lidar_angle_to_vehicle(a):
        """Convert a raw (clockwise) LiDAR angle to a vehicle-frame CCW bearing in [-pi, pi]."""
        aa = a - 2 * math.pi if a > math.pi else a
        return -aa

    def lidar_median_distance(self, angles, distances, target_angle, half_window=math.radians(3)):
        """[Enhancement] Median of LiDAR distances within target_angle +/- half_window, robust to single-point noise.
        target_angle is a vehicle-frame CCW bearing (e.g. derived from the camera)."""
        sel = []
        for a, d in zip(angles, distances):
            av = self.lidar_angle_to_vehicle(a)
            if abs(av - target_angle) <= half_window and 0.05 < d < 30.0:
                sel.append(d)
        if not sel:
            return None
        return float(np.median(sel))

    def process_lidar_for_obstacles(self, steering=0.0):
        """Forward-cone obstacle detection + professional 2D point-cloud view.
        Returns (obstacle_detected, obstacle lateral offset y, min frontal distance).

        The corridor shortens while turning (|steering| > 0.2) so that outer
        walls on bend exits are not treated as in-lane obstacles.

        Note: the QCar2 LiDAR in QLabs is a 2D single-layer scanner --
        get_lidar() returns one 360-degree polar sweep (angles, distances)
        per frame, with a single horizontal scan plane and no vertical
        dimension. A 3D point cloud requires back-projecting the RealSense
        depth camera (CAMERA_DEPTH) instead.
        """
        success, angles, distances = self.qcar.get_lidar(samplePoints=400)
        if not success:
            return False, 0.0, float("inf")

        x_max = OBS_X_MAX_TURN if abs(steering) > 0.2 else OBS_X_MAX

        # ---- Obstacle detection (decoupled from rendering) ----
        # Normal obstacles must appear as a small point cluster and persist for a
        # few frames. Single curb/wall/sign-pole grazing returns are usually 1-2
        # sparse points, so they no longer latch the hold state. A truly close
        # bumper hit still bypasses confirmation and brakes immediately.
        obstacle_pts = []
        emergency_pts = []
        min_front = float("inf")

        for a, dist in zip(angles, distances):
            av = self.lidar_angle_to_vehicle(a)   # CCW vehicle-frame bearing
            x_local = dist * math.cos(av)
            y_local = dist * math.sin(av)         # positive = car's LEFT

            if abs(av) < 0.5 and 0.05 < dist < 1.0 and -OBS_Y_HALF <= y_local <= OBS_Y_HALF:
                # Emergency zone deliberately starts in front of the bumper, not
                # at OBS_X_MIN, otherwise OBSTACLE_EMERGENCY_DIST=0.16 can never fire.
                if 0.05 < x_local < x_max:
                    min_front = min(min_front, dist)
                    if dist < OBSTACLE_EMERGENCY_DIST:
                        emergency_pts.append((x_local, y_local))

                if OBS_X_MIN < x_local < x_max:
                    obstacle_pts.append((x_local, y_local))

        dense_obstacle = len(obstacle_pts) >= OBS_MIN_CLUSTER_POINTS
        emergency_obstacle = len(emergency_pts) > 0

        if emergency_obstacle:
            self._obstacle_confirm_count = OBS_CONFIRM_FRAMES
            confirmed_obstacle = True
            active_pts = emergency_pts or obstacle_pts
        elif dense_obstacle:
            self._obstacle_confirm_count = getattr(self, "_obstacle_confirm_count", 0) + 1
            confirmed_obstacle = self._obstacle_confirm_count >= OBS_CONFIRM_FRAMES
            active_pts = obstacle_pts
        else:
            self._obstacle_confirm_count = 0
            confirmed_obstacle = False
            active_pts = obstacle_pts

        if active_pts:
            closest = min(active_pts, key=lambda p: math.hypot(p[0], p[1]))
            obstacle_y = float(closest[1])
        else:
            obstacle_y = 0.0

        if SHOW_WINDOWS:
            self.render_lidar_view(angles, distances,
                                   obstacle_pts + emergency_pts, min_front, x_max=x_max)
        return confirmed_obstacle, obstacle_y, min_front

    def render_lidar_view(self, angles, distances, obstacle_pts, min_front,
                          x_max=OBS_X_MAX, view_radius_m=4.0, size=560):
        """[Enhancement] Professional LiDAR 2D point-cloud view:
        range rings / 45-deg rays / distance-colored points / obstacle zone
        + hit markers / passenger bearing line."""
        c = size // 2
        scale = (c - 30) / view_radius_m  # px per meter
        img = np.full((size, size, 3), 18, dtype=np.uint8)

        def to_px(x_fwd, y_left):
            # Vehicle frame: x forward (up on screen), y left (left on screen)
            return (int(c - y_left * scale), int(c - x_fwd * scale))

        # Range rings + labels
        for r in (1.0, 2.0, 3.0, 4.0):
            cv2.circle(img, (c, c), int(r * scale), (55, 55, 55), 1, cv2.LINE_AA)
            cv2.putText(img, f"{r:.0f}m", (c + int(r * scale) - 24, c - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
        # 45-degree rays
        for k in range(8):
            th = k * math.pi / 4
            ex, ey = to_px(view_radius_m * math.cos(th), view_radius_m * math.sin(th))
            cv2.line(img, (c, c), (ex, ey), (40, 40, 40), 1, cv2.LINE_AA)

        # Obstacle-detection zone (forward rectangular corridor, synced with the detection constants)
        zone = np.array([to_px(OBS_X_MIN, OBS_Y_HALF), to_px(x_max, OBS_Y_HALF),
                         to_px(x_max, -OBS_Y_HALF), to_px(OBS_X_MIN, -OBS_Y_HALF)],
                        dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [zone], (0, 45, 90))
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.polylines(img, [zone], True, (0, 120, 255), 1, cv2.LINE_AA)

        # Scan points: near=red -> far=cyan gradient
        n_valid = 0
        for a, dist in zip(angles, distances):
            if not (0.05 < dist < view_radius_m):
                continue
            n_valid += 1
            av = self.lidar_angle_to_vehicle(a)  # mirror fix: raw angles are clockwise
            px, py = to_px(dist * math.cos(av), dist * math.sin(av))
            if 0 <= px < size and 0 <= py < size:
                t = min(dist / view_radius_m, 1.0)
                color = (int(230 * t), int(200 * t), int(255 * (1 - t)))  # BGR
                cv2.circle(img, (px, py), 2, color, -1, cv2.LINE_AA)

        # Obstacle hit markers
        for (ox, oy) in obstacle_pts:
            px, py = to_px(ox, oy)
            cv2.circle(img, (px, py), 6, (0, 0, 255), 2, cv2.LINE_AA)

        # Passenger bearing line (if the background YOLO has a confirmed passenger bbox)
        with self.vision_lock:
            pb = self.latest_person_bbox
        if pb is not None:
            center_u = (pb[0] + pb[2]) / 2
            p_ang = -((center_u - 320) / 320.0) * (41.0 * math.pi / 180.0)
            p_d = self.lidar_median_distance(angles, distances, p_ang)
            end_d = min(p_d, view_radius_m) if p_d else view_radius_m
            ex, ey = to_px(end_d * math.cos(p_ang), end_d * math.sin(p_ang))
            cv2.line(img, (c, c), (ex, ey), (255, 160, 60), 2, cv2.LINE_AA)
            label = f"PASSENGER {p_d:.2f}m" if p_d else "PASSENGER ?"
            cv2.putText(img, label, (ex + 6, ey), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 180, 90), 1, cv2.LINE_AA)

        # Vehicle body + heading arrow (up on screen = front of car)
        cv2.arrowedLine(img, (c, c + 12), (c, c - 22), (0, 255, 0), 2,
                        cv2.LINE_AA, tipLength=0.45)
        cv2.rectangle(img, (c - 7, c - 12), (c + 7, c + 12), (0, 255, 0), 1, cv2.LINE_AA)

        # Status bar
        cv2.putText(img, "2D LiDAR  360deg x 1 layer", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"points: {n_valid}/400 in {view_radius_m:.0f}m", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        if obstacle_pts:
            cv2.putText(img, f"OBSTACLE {min_front:.2f}m - HOLDING", (10, size - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        imshow_safe("LiDAR View", img)
        cv2.waitKey(1)

    # ---------------- Visualization overlay ----------------
    def detect_traffic_signs_and_lights(self, img):
        if img is None:
            return [], None, {}, img
        with self.vision_lock:
            self.latest_image = img.copy()
            detected_objects = list(self.latest_detected_objects)
            tl_state = self.latest_tl_state
            boxes_to_draw = list(self.latest_boxes)
            sign_info = dict(self.latest_sign_info)

        # Clean YOLO overlay: draw only accepted detections, use small labels with
        # a dark label background so the camera view is readable.
        for (x1, y1, x2, y2, label, conf, color) in boxes_to_draw:
            if tuple(color) == (160, 160, 160) or conf < 0.45:
                continue
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            label_text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            y0 = max(0, y1 - th - 7)
            cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y1), (25, 25, 25), -1)
            cv2.putText(img, label_text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Compact status bar instead of large repeated text in the camera view.
        status = []
        if tl_state:
            status.append(f"TL:{tl_state}")
        if detected_objects:
            compact = [o for o in detected_objects if o not in ("TL_RED", "TL_GREEN", "TL_YELLOW")]
            if compact:
                status.append("SIG:" + ",".join(sorted(set(compact))))
        if status:
            msg = "  ".join(status)
            cv2.rectangle(img, (6, 8), (min(img.shape[1] - 1, 18 + 9 * len(msg)), 34), (25, 25, 25), -1)
            cv2.putText(img, msg, (12, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)
        return detected_objects, tl_state, sign_info, img

    # ---------------- HSV lane detection    # ---------------- HSV lane detection (fallback path) ----------------
    def compute_vision_steering(self):
        success, img = self.qcar.get_image(self.qcar.CAMERA_CSI_FRONT)
        if not success or img is None:
            return self.last_steering, 0
        img_processed = cv2.resize(img, (640, 360))
        hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(hsv, np.array([10, 30, 40]), np.array([45, 255, 255]))
        roi_y = mask_y[200:340, :]
        pts_y = np.argwhere(roi_y > 0)

        kp, kd = 0.012, 0.025
        if len(pts_y) > 40:
            raw_center = np.mean(pts_y[:, 1])
            self.smoothed_center_x = 0.6 * self.smoothed_center_x + 0.4 * raw_center
            error = self.smoothed_center_x - 120
            derivative = np.clip(error - self.last_error, -25, 25)
            steering = error * kp + derivative * kd
            self.last_error = error
            self.last_steering = np.clip(steering, -0.58, 0.58)
            return self.last_steering, 2
        self.last_error = 0.0
        return self.last_steering, 0


# ------------------------------------------------------------------
# 3. STATE ESTIMATION
# ------------------------------------------------------------------
class LocalizationEKF:
    """EKF over [x, y, yaw]. predict uses the kinematic model; update fuses an
    external pose measurement (in digital-twin mode, the pose returned by QLabs)."""

    def __init__(self):
        self.x = np.zeros(3)
        self.P = np.eye(3)
        self.Q = np.diag([0.1, 0.1, 0.05])
        self.R = np.diag([0.5, 0.5, 0.1])

    def predict(self, v, omega, dt):
        # [Gap 4] omega is computed by the caller as v*tan(delta)/L, no longer constant 0
        self.x[0] += v * math.cos(self.x[2]) * dt
        self.x[1] += v * math.sin(self.x[2]) * dt
        self.x[2] += omega * dt
        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))

        F = np.eye(3)
        F[0, 2] = -v * math.sin(self.x[2]) * dt
        F[1, 2] =  v * math.cos(self.x[2]) * dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.eye(3)
        y = z - self.x
        y[2] = math.atan2(math.sin(y[2]), math.cos(y[2]))
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))
        self.P = (np.eye(3) - K @ H) @ self.P


# ------------------------------------------------------------------
# 4. CONTROL
# ------------------------------------------------------------------
class PurePursuitController:
    """Pure Pursuit lateral control."""

    def __init__(self, wheelbase=0.256, lookahead_dist=0.6):
        self.L = wheelbase
        self.Ld = lookahead_dist
        self.current_wp_index = 0
        self.target_waypoints = []

    def compute_steering(self, current_pose, path_waypoints):
        if not path_waypoints or self.current_wp_index >= len(path_waypoints) - 1:
            return 0.0, True

        xy = np.array([current_pose[0], current_pose[1]])
        local_path = np.array(path_waypoints[self.current_wp_index:
                                             self.current_wp_index + 100])
        nearest_offset = int(np.argmin(np.linalg.norm(local_path - xy, axis=1)))
        self.current_wp_index += nearest_offset

        target_index = self.current_wp_index
        while target_index < len(path_waypoints) - 1:
            tx, ty = path_waypoints[target_index]
            if math.hypot(tx - current_pose[0], ty - current_pose[1]) >= self.Ld:
                break
            target_index += 1

        tx, ty = path_waypoints[target_index]
        alpha = math.atan2(ty - current_pose[1], tx - current_pose[0]) - current_pose[2]
        alpha = (alpha + math.pi) % (2.0 * math.pi) - math.pi
        distance = max(math.hypot(tx - current_pose[0], ty - current_pose[1]), 1e-3)
        steering = math.atan2(2.0 * self.L * math.sin(alpha), distance)
        return float(np.clip(steering, -0.58, 0.58)), False


class RoadMapDebugWindow:
    """Quanser-style top-down roadmap visualizer.

    This is intentionally visualization-only. It does not filter steering, change
    lookahead, or modify speed. It draws a reusable top-down road background from
    the ACCRoadMap2 path geometry, then overlays the current reference trajectory
    and the estimated vehicle trajectory.

    If an official bitmap background exists in the same folder, set
    ACC_MAP_IMAGE to its path. Without that asset, the background is generated
    from the official roadmap centerlines available through ACCRoadMap2.
    """

    def __init__(self, roadmap, enabled=True, width=620, height=620):
        self.enabled = bool(enabled and SHOW_WINDOWS)
        self.roadmap = roadmap
        self.width = int(width)
        self.height = int(height)
        self.panel_width = 320
        self.trace = deque(maxlen=MAP_DEBUG_MAX_TRACE)
        self.history = deque(maxlen=240)
        self.travel_distance = 0.0
        self._last_trace_point = None
        self.last_update = 0.0
        self._disabled = False
        self.bg = None
        self.bounds = None
        self.base_paths = []
        if self.enabled:
            try:
                self._build_background()
            except Exception as e:
                print(f"[MAP_DEBUG] Background build failed; fallback route-only view will be used: {e}")
                self.bg = np.full((self.height, self.width, 3), 235, dtype=np.uint8)
                self.bounds = (-1.8, 1.8, -0.5, 5.5)

    def _as_xy_path(self, path):
        if path is None:
            return []
        arr = np.asarray(path, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape((-1, 2))
        if arr.shape[1] >= 2:
            return [(float(p[0]), float(p[1])) for p in arr]
        return []

    def _node_ids(self):
        nodes = getattr(self.roadmap, 'nodes', None)
        if nodes is None:
            return list(dict.fromkeys(VALID_NODES + [DROPOFF_NODE]))
        try:
            ids = list(nodes.keys())
        except Exception:
            try:
                ids = list(range(len(nodes)))
            except Exception:
                ids = list(dict.fromkeys(VALID_NODES + [DROPOFF_NODE]))
        # Keep the view focused on the ACC road network. Some roadmap objects may
        # contain helper nodes outside the drivable area.
        known = list(dict.fromkeys(VALID_NODES + [DROPOFF_NODE] + ids))
        return known

    def _try_path(self, seq):
        try:
            result = self.roadmap.generate_path(seq)
            if result is None:
                return []
            path = result[0] if isinstance(result, tuple) else result
            return self._as_xy_path(path)
        except Exception:
            return []

    def _build_background(self):
        official_img = os.environ.get('ACC_MAP_IMAGE')
        if official_img and os.path.isfile(official_img):
            img = cv2.imread(official_img)
            if img is not None:
                self.bg = cv2.resize(img, (self.width, self.height))
                self.bounds = self._estimate_bounds_from_nodes()
                return

        node_ids = self._node_ids()
        raw_paths = []
        all_xy = []

        # Draw the main mission loop first so the background is useful even if
        # the shortest-path solver rejects some arbitrary pairs.
        route_candidates = [
            [10, 0, 4, 14, 23, 20, 10],
            [10, 0, 4],
            [4, 14, 23, 20, 10],
            [14, 22],
            [23, 22],
            [20, 22],
        ]
        for seq in route_candidates:
            p = self._try_path(seq)
            if len(p) >= 2:
                raw_paths.append(p)
                all_xy.extend(p)

        # Add many two-node paths to recover most of the official drivable graph.
        # This is done once at startup, not every control step.
        for a in node_ids:
            for b in node_ids:
                if a == b:
                    continue
                p = self._try_path([a, b])
                if len(p) >= 2:
                    # Reject extremely long all-around routes from the background;
                    # they overpaint the map and make it less readable.
                    length = sum(math.hypot(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1]) for i in range(len(p)-1))
                    if length <= 8.0:
                        raw_paths.append(p)
                        all_xy.extend(p)

        if not all_xy:
            # Fallback to node poses only.
            for n in node_ids:
                try:
                    p = np.asarray(self.roadmap.nodes[n].pose).ravel()
                    all_xy.append((float(p[0]), float(p[1])))
                except Exception:
                    pass

        if all_xy:
            xs = [p[0] for p in all_xy]
            ys = [p[1] for p in all_xy]
            margin = 0.35
            self.bounds = (min(xs)-margin, max(xs)+margin, min(ys)-margin, max(ys)+margin)
        else:
            self.bounds = (-1.8, 1.8, -0.5, 5.5)

        self.base_paths = raw_paths
        self.bg = self._draw_vector_map()

    def _estimate_bounds_from_nodes(self):
        xs, ys = [], []
        for n in self._node_ids():
            try:
                p = np.asarray(self.roadmap.nodes[n].pose).ravel()
                xs.append(float(p[0])); ys.append(float(p[1]))
            except Exception:
                pass
        if xs and ys:
            return (min(xs)-0.4, max(xs)+0.4, min(ys)-0.4, max(ys)+0.4)
        return (-1.8, 1.8, -0.5, 5.5)

    def _world_to_px(self, x, y):
        xmin, xmax, ymin, ymax = self.bounds
        # Preserve aspect ratio by expanding the smaller axis.
        bx = xmax - xmin
        by = ymax - ymin
        canvas_aspect = self.width / max(1, self.height)
        data_aspect = bx / max(1e-6, by)
        if data_aspect > canvas_aspect:
            new_by = bx / canvas_aspect
            c = 0.5 * (ymin + ymax)
            ymin, ymax = c - 0.5 * new_by, c + 0.5 * new_by
        else:
            new_bx = by * canvas_aspect
            c = 0.5 * (xmin + xmax)
            xmin, xmax = c - 0.5 * new_bx, c + 0.5 * new_bx
        u = int((x - xmin) / max(1e-6, xmax - xmin) * (self.width - 1))
        v = self.height - 1 - int((y - ymin) / max(1e-6, ymax - ymin) * (self.height - 1))
        return u, v

    def _path_to_px(self, path):
        pts = [self._world_to_px(float(x), float(y)) for x, y in path]
        return np.asarray(pts, dtype=np.int32) if len(pts) >= 2 else None

    def _draw_vector_map(self):
        canvas = np.full((self.height, self.width, 3), 218, dtype=np.uint8)
        # Light grid similar to the QLabs top-down floor.
        for u in range(0, self.width, 40):
            cv2.line(canvas, (u, 0), (u, self.height), (205, 205, 205), 1)
        for v in range(0, self.height, 40):
            cv2.line(canvas, (0, v), (self.width, v), (205, 205, 205), 1)

        # Draw roads as white shoulders + black asphalt. Paths are centerlines
        # generated by the official roadmap; thick strokes create a readable map.
        for path in self.base_paths:
            pts = self._path_to_px(path)
            if pts is not None:
                cv2.polylines(canvas, [pts], False, (250, 250, 250), 34, lineType=cv2.LINE_AA)
        for path in self.base_paths:
            pts = self._path_to_px(path)
            if pts is not None:
                cv2.polylines(canvas, [pts], False, (18, 18, 18), 24, lineType=cv2.LINE_AA)
        for path in self.base_paths:
            pts = self._path_to_px(path)
            if pts is not None:
                cv2.polylines(canvas, [pts], False, (180, 180, 180), 1, lineType=cv2.LINE_AA)

        # Mark major nodes so the planned route is easier to interpret.
        for n in list(dict.fromkeys(VALID_NODES + [DROPOFF_NODE])):
            try:
                p = np.asarray(self.roadmap.nodes[n].pose).ravel()
                u, v = self._world_to_px(float(p[0]), float(p[1]))
                cv2.circle(canvas, (u, v), 4, (235, 235, 235), -1, lineType=cv2.LINE_AA)
                cv2.putText(canvas, str(n), (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 70), 1)
            except Exception:
                pass
        return canvas

    def reset_trace(self):
        """Clear the green measured trajectory and x/y history at the start of each route."""
        self.trace.clear()
        self.history.clear()
        self.travel_distance = 0.0
        self._last_trace_point = None

    def _draw_sparkline(self, panel, rect, values_a, values_b=None,
                        label='', color_a=(80, 80, 235), color_b=(80, 210, 80)):
        x0, y0, w, h = rect
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (58, 58, 58), 1, cv2.LINE_AA)
        cv2.putText(panel, label, (x0 + 6, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (210, 210, 210), 1, cv2.LINE_AA)
        vals = []
        vals.extend([v for v in values_a if np.isfinite(v)])
        if values_b is not None:
            vals.extend([v for v in values_b if np.isfinite(v)])
        if len(vals) < 2:
            return
        mn, mx = min(vals), max(vals)
        if abs(mx - mn) < 1e-6:
            mn -= 0.5
            mx += 0.5
        pad = 0.12 * (mx - mn)
        mn -= pad
        mx += pad

        def pts_for(values):
            values = list(values)
            if len(values) < 2:
                return None
            pts = []
            n = len(values)
            for i, val in enumerate(values):
                if not np.isfinite(val):
                    continue
                u = int(x0 + 6 + i * (w - 12) / max(1, n - 1))
                v = int(y0 + h - 6 - (val - mn) / max(1e-9, mx - mn) * (h - 22))
                pts.append((u, v))
            return np.asarray(pts, dtype=np.int32) if len(pts) >= 2 else None

        pts = pts_for(values_a)
        if pts is not None:
            cv2.polylines(panel, [pts], False, color_a, 1, cv2.LINE_AA)
        if values_b is not None:
            pts = pts_for(values_b)
            if pts is not None:
                cv2.polylines(panel, [pts], False, color_b, 1, cv2.LINE_AA)

    def update(self, pose, route_path, wp_index=0, target_destination=None,
               target_v=0.0, steering=0.0, state=None, obstacle=False):
        if not self.enabled or self._disabled:
            return
        now = time.time()
        if now - self.last_update < MAP_DEBUG_UPDATE_PERIOD:
            return
        self.last_update = now
        try:
            x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
            self.trace.append((x, y))
            if self._last_trace_point is not None:
                self.travel_distance += math.hypot(x - self._last_trace_point[0], y - self._last_trace_point[1])
            self._last_trace_point = (x, y)

            route_xy = self._as_xy_path(route_path)
            ref_x = ref_y = float('nan')
            if len(route_xy) >= 2:
                idx = min(max(int(wp_index), 0), len(route_xy) - 1)
                ref_x, ref_y = route_xy[idx]
            dx = x - ref_x if np.isfinite(ref_x) else float('nan')
            dy = y - ref_y if np.isfinite(ref_y) else float('nan')
            pos_err = math.hypot(dx, dy) if np.isfinite(dx) and np.isfinite(dy) else float('nan')
            head_ref = float('nan')
            if len(route_xy) >= 2:
                idx2 = min(max(int(wp_index), 0), len(route_xy) - 2)
                head_ref = math.atan2(route_xy[idx2 + 1][1] - route_xy[idx2][1],
                                      route_xy[idx2 + 1][0] - route_xy[idx2][0])
            head_err = math.atan2(math.sin(yaw - head_ref), math.cos(yaw - head_ref)) if np.isfinite(head_ref) else float('nan')
            self.history.append((x, y, yaw, ref_x, ref_y, head_ref, dx, dy, pos_err, steering, target_v))

            map_canvas = self.bg.copy() if self.bg is not None else np.full((self.height, self.width, 3), 235, dtype=np.uint8)

            if len(route_xy) >= 2:
                pts = self._path_to_px(route_xy)
                if pts is not None:
                    cv2.polylines(map_canvas, [pts], False, (35, 55, 235), 3, lineType=cv2.LINE_AA)
                if np.isfinite(ref_x):
                    u, v = self._world_to_px(ref_x, ref_y)
                    cv2.circle(map_canvas, (u, v), 6, (255, 180, 0), -1, lineType=cv2.LINE_AA)

            if len(self.trace) >= 2:
                pts = self._path_to_px(list(self.trace))
                if pts is not None:
                    cv2.polylines(map_canvas, [pts], False, (40, 190, 40), 3, lineType=cv2.LINE_AA)

            if target_destination is not None:
                tx, ty = float(target_destination[0]), float(target_destination[1])
                u, v = self._world_to_px(tx, ty)
                cv2.circle(map_canvas, (u, v), 8, (200, 0, 200), 2, lineType=cv2.LINE_AA)

            u, v = self._world_to_px(x, y)
            cv2.circle(map_canvas, (u, v), 7, (0, 230, 255), -1, lineType=cv2.LINE_AA)
            hu = int(u + 22 * math.cos(yaw))
            hv = int(v - 22 * math.sin(yaw))
            cv2.arrowedLine(map_canvas, (u, v), (hu, hv), (0, 160, 255), 2, cv2.LINE_AA, tipLength=0.35)

            state_name = {
                FSMState.STATE_IDLE_HUB: 'IDLE_HUB',
                FSMState.STATE_NAV_TO_PICKUP: 'NAV_TO_PICKUP',
                FSMState.STATE_PASSENGER_LOADING: 'LOADING',
                FSMState.STATE_NAV_TO_DROPOFF: 'NAV_TO_DROPOFF',
                FSMState.STATE_RETURN_TO_HUB: 'RETURN_TO_HUB',
                FSMState.STATE_PATROLLING: 'PATROLLING',
            }.get(state, str(state))

            total_h = self.height + 72
            panel = np.zeros((total_h, self.panel_width, 3), dtype=np.uint8)
            panel[:] = (24, 24, 24)
            cv2.putText(panel, 'Vehicle Control', (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (238, 238, 238), 1, cv2.LINE_AA)
            cv2.putText(panel, state_name, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 255), 1, cv2.LINE_AA)
            if obstacle:
                cv2.putText(panel, 'OBSTACLE/HOLD', (178, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 70, 255), 1, cv2.LINE_AA)

            rows = [
                ('x_ref', ref_x, 'm'), ('x_meas', x, 'm'), ('x_err', dx, 'm'),
                ('y_ref', ref_y, 'm'), ('y_meas', y, 'm'), ('y_err', dy, 'm'),
                ('pos_err', pos_err, 'm'), ('head_err', head_err, 'rad'),
                ('distance', self.travel_distance, 'm'), ('speed', target_v, 'm/s'),
                ('steer', steering, 'rad'),
            ]
            yy = 82
            for i, (name, val, unit) in enumerate(rows):
                col = 14 if i < 6 else 164
                row_y = yy + (i % 6) * 23
                if not np.isfinite(val):
                    s = '---'
                elif name.endswith('err') or name == 'steer':
                    s = f'{val:+.3f}'
                else:
                    s = f'{val:.3f}'
                cv2.putText(panel, f'{name:<8} {s} {unit}', (col, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

            hist = list(self.history)
            xs = [h[0] for h in hist]; ys = [h[1] for h in hist]; th = [h[2] for h in hist]
            xrs = [h[3] for h in hist]; yrs = [h[4] for h in hist]; thrs = [h[5] for h in hist]
            dxs = [h[6] for h in hist]; dys = [h[7] for h in hist]
            self._draw_sparkline(panel, (14, 230, 292, 84), xrs, xs, 'X position: ref(red) / meas(green)')
            self._draw_sparkline(panel, (14, 326, 292, 84), yrs, ys, 'Y position: ref(red) / meas(green)')
            self._draw_sparkline(panel, (14, 422, 292, 84), thrs, th, 'Heading: ref(red) / meas(green)')
            self._draw_sparkline(panel, (14, 518, 292, 84), dxs, dys, 'XY offset: dx(red) / dy(green)')
            cv2.putText(panel, 'Route trace resets on every new route.', (14, total_h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

            right = np.zeros((total_h, self.width, 3), dtype=np.uint8)
            right[:] = (30, 30, 30)
            cv2.putText(right, 'Roadmap View', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(right, f'v={target_v:.2f} m/s   steer={steering:+.2f} rad   red=route  green=car trace  yellow=car',
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            right[72:72 + self.height, :, :] = map_canvas
            out = np.hstack([panel, right])
            cv2.imshow('Vehicle Steering Control', out)
            cv2.waitKey(1)
        except Exception as e:
            print(f"[MAP_DEBUG] Disabled after error: {e}")
            self._disabled = True


# ------------------------------------------------------------------
# 5. MISSION PLANNER & FSM
# ------------------------------------------------------------------
class AutonomousTaxiMission:
    def __init__(self):
        self.state = FSMState.STATE_IDLE_HUB
        self.qlabs = QuanserInteractiveLabs()
        self.car = None
        self.perception = None
        self.localization = LocalizationEKF()

        self.dt = 0.01
        self.current_velocity = 0.0
        self.last_steering_cmd = 0.0
        self._route_start_time = time.time()
        self._route_warmup_until = self._route_start_time
        self._last_control_time = self._route_start_time
        self._last_turn_time = 0.0

        self.pure_pursuit = PurePursuitController(lookahead_dist=LOOKAHEAD)
        self.roadmap = ACCRoadMap2()
        self.map_window = RoadMapDebugWindow(self.roadmap, enabled=SHOW_MAP_DEBUG)

        self.start_time = time.time()
        self.last_light_phase = -1
        self.traffic_lights = []
        self.first_person_spawned = False
        self.current_led = None
        self.last_seen_tl_color = None
        self.last_seen_tl_time = 0.0
        self.clearing_intersection = False
        self._stop_sign_active = False
        self._stop_sign_done = False
        self._stop_sign_release_until = 0.0
        self._stop_sign_armed = False
        self._stop_sign_last_seen_time = 0.0
        self._stop_sign_max_area = 0.0
        self.patrol_start_time = None


    # ---------------- Node pose (compatible with (3,1) ndarray) ----------------
    def node_pose(self, n):
        """Return node n's (x, y, yaw) as plain Python floats, compatible with the (3,1) numpy shape."""
        p = np.asarray(self.roadmap.nodes[n].pose).ravel()
        return float(p[0]), float(p[1]), float(p[2])

    # ---------------- Passenger boarding/alighting animation ----------------
    def car_side_point(self, lateral, forward=0.0):
        """World coordinates of a vehicle-frame offset point based on the current pose (lateral > 0 is the car's right side)."""
        x, y, yaw = self.localization.x
        wx = x + forward * math.cos(yaw) + lateral * math.cos(yaw - math.pi / 2)
        wy = y + forward * math.sin(yaw) + lateral * math.sin(yaw - math.pi / 2)
        return float(wx), float(wy)

    def passenger_board(self):
        """[Enhancement] Pickup animation: passenger AI-pathfinds to the car door,
        then boards (actor destroyed). If move_to fails or times out, fall back
        to instant boarding so the mission never blocks."""
        if not hasattr(self, "person"):
            return
        door_x, door_y = self.car_side_point(DOOR_OFFSET)
        walked = False
        if PASSENGER_ANIMATION:
            try:
                walked = self.person.move_to(
                    location=[door_x, door_y, 0.0],
                    speed=QLabsPerson.WALK, waitForConfirmation=True)
            except Exception:
                walked = False
        if walked:
            # move_to is non-blocking: estimate walk time as distance/speed, capped at BOARD_TIMEOUT
            px, py, pyaw = self.node_pose(self.spawn_node)
            est = math.hypot(door_x - px, door_y - py) / QLabsPerson.WALK + 1.0
            wait = min(est, BOARD_TIMEOUT)
            print(f"[LOADING] Passenger walking to the door (~{wait:.1f}s)...")
            time.sleep(wait)
        try:
            self.person.destroy()
        except Exception:
            pass
        print("[LOADING] Passenger has boarded.")

    def passenger_alight(self):
        """[Enhancement] Drop-off animation: spawn the passenger at the door -> walk to the sidewalk -> clean up."""
        if not hasattr(self, "person"):
            return
        door_x, door_y = self.car_side_point(DOOR_OFFSET)
        sw_x, sw_y = self.car_side_point(SIDEWALK_OFFSET, forward=0.3)
        spawned = False
        if PASSENGER_ANIMATION:
            try:
                self.person.spawn_id(actorNumber=0,
                                     location=[door_x, door_y, 0.05],
                                     rotation=[0, 0, 0], scale=[0.1] * 3,
                                     configuration=0, waitForConfirmation=True)
                spawned = True
                self.person.move_to(location=[sw_x, sw_y, 0.0],
                                    speed=QLabsPerson.WALK,
                                    waitForConfirmation=False)
                print("[DROPOFF] Passenger walking to the sidewalk...")
            except Exception:
                spawned = False
        if PASSENGER_ANIMATION:
            time.sleep(WALKAWAY_TIME)
        if spawned:
            try:
                self.person.destroy()
            except Exception:
                pass
        print("[DROPOFF] Passenger drop-off complete.")

    # ---------------- LED state signals ----------------
    def set_led(self, color):
        """[Enhancement] LED state signal; deduplicated to avoid spamming commands."""
        if self.car is None or color == self.current_led:
            return
        try:
            self.car.set_led_strip_uniform(color=color, waitForConfirmation=False)
            self.current_led = color
        except Exception:
            pass

    # ---------------- Scene setup ----------------
    def setup_sim(self):
        print("Connecting to QLabs...")
        try:
            self.qlabs.open("localhost")
        except Exception:
            print("Error: unable to connect to QLabs. Please open the QLabs application first.")
            return False

        print("Building the official ACC competition map...")
        # [Gap 3] the fixed setup() genuinely reuses this connection
        self.car = Setup_Real_Scenario.setup(self.qlabs)
        if self.car is None:
            print("Error: scene setup failed.")
            return False

        print("Spawning active traffic lights...")
        for i in range(4):
            self.traffic_lights.append(QLabsTrafficLight(self.qlabs))
        self.traffic_lights[0].spawn_id_degrees(actorNumber=1, location=[0.6, 1.55, 0.006],
            rotation=[0, 0, 0],   scale=[0.1]*3, configuration=0, waitForConfirmation=False)
        self.traffic_lights[1].spawn_id_degrees(actorNumber=2, location=[-0.6, 1.28, 0.006],
            rotation=[0, 0, 90],  scale=[0.1]*3, configuration=0, waitForConfirmation=False)
        self.traffic_lights[2].spawn_id_degrees(actorNumber=3, location=[-0.37, 0.3, 0.006],
            rotation=[0, 0, 180], scale=[0.1]*3, configuration=0, waitForConfirmation=False)
        self.traffic_lights[3].spawn_id_degrees(actorNumber=4, location=[0.75, 0.48, 0.006],
            rotation=[0, 0, -90], scale=[0.1]*3, configuration=0, waitForConfirmation=True)

        self.car.possess(self.car.CAMERA_TRAILING)
        self.perception = PerceptionSystem(self.car)
        # Hold the vehicle still and seed EKF from the real QLabs pose before the
        # first Pure Pursuit command. This prevents the very first command from
        # being computed from [0,0,0], which can launch the car sideways.
        for _ in range(8):
            status, loc, rot, _, _ = self.car.set_velocity_and_request_state(
                forward=0.0, turn=0.0, headlights=255,
                leftTurnSignal=0, rightTurnSignal=0, brakeSignal=255, reverseSignal=0)
            if status:
                self.localization.x = np.array([loc[0], loc[1], rot[2]], dtype=float)
                self.localization.P = np.eye(3) * 0.01
            time.sleep(0.02)
        self.set_led(LED_IDLE)
        return True

    # ---------------- Traffic-light phase animation ----------------
    def animate_traffic_lights(self):
        elapsed = int(time.time() - self.start_time)
        phase = (elapsed // 5) % 4
        if phase != self.last_light_phase and len(self.traffic_lights) == 4:
            TL = QLabsTrafficLight
            if phase == 0:
                self.traffic_lights[0].set_color(color=TL.COLOR_RED)
                self.traffic_lights[2].set_color(color=TL.COLOR_RED)
                self.traffic_lights[1].set_color(color=TL.COLOR_GREEN)
                self.traffic_lights[3].set_color(color=TL.COLOR_GREEN)
            elif phase == 1:
                self.traffic_lights[1].set_color(color=TL.COLOR_YELLOW)
                self.traffic_lights[3].set_color(color=TL.COLOR_YELLOW)
            elif phase == 2:
                self.traffic_lights[0].set_color(color=TL.COLOR_GREEN)
                self.traffic_lights[2].set_color(color=TL.COLOR_GREEN)
                self.traffic_lights[1].set_color(color=TL.COLOR_RED)
                self.traffic_lights[3].set_color(color=TL.COLOR_RED)
            elif phase == 3:
                self.traffic_lights[0].set_color(color=TL.COLOR_YELLOW)
                self.traffic_lights[2].set_color(color=TL.COLOR_YELLOW)
            self.last_light_phase = phase

    # ---------------- Route dispatch ----------------
    def _clean_node_sequence(self, node_sequence):
        clean = []
        for n in node_sequence:
            if n is None:
                continue
            n = int(n)
            if not clean or clean[-1] != n:
                clean.append(n)
        return clean

    def _node_point_path(self, n):
        x, y, _ = self.node_pose(n)
        return np.array([[x, y]], dtype=float)

    def _try_generate_path(self, node_sequence):
        seq = self._clean_node_sequence(node_sequence)
        if len(seq) == 0:
            return None, seq
        if len(seq) == 1:
            return self._node_point_path(seq[0]), seq
        try:
            result = self.roadmap.generate_path(seq)
            if result is None:
                return None, seq
            path, _ = result
            if path is None or len(path) == 0:
                return None, seq
            return np.asarray(path, dtype=float), seq
        except Exception as e:
            print(f"\n[ROUTE] generate_path failed for {seq}: {e}")
            return None, seq

    def load_route(self, node_sequence):
        # Robust route loader. The Quanser roadmap returns None for some degenerate
        # requests, especially [node, node]. The old code crashed on that None.
        path, seq = self._try_generate_path(node_sequence)

        if path is None and len(seq) >= 2:
            # Fallback: follow the known main loop in the legal driving direction until goal.
            # This is safer than asking the shortest-path solver for a pair it cannot connect.
            loop = [10, 0, 4, 14, 23, 20, 10]
            start, goal = seq[0], seq[-1]
            if start in loop and goal in loop:
                i = loop.index(start)
                route = [start]
                guard = 0
                while route[-1] != goal and guard < len(loop) + 2:
                    i = (i + 1) % (len(loop) - 1)
                    route.append(loop[i])
                    guard += 1
                path, seq = self._try_generate_path(route)
                if path is not None:
                    print(f"\n[ROUTE] Fallback main-loop route: {route}")

        if path is None:
            raise RuntimeError(f"No valid roadmap path for node sequence {node_sequence}; cleaned={seq}")

        self.pure_pursuit.target_waypoints = path.tolist()
        self.pure_pursuit.current_wp_index = 0
        self.target_destination = self.pure_pursuit.target_waypoints[-1]
        self._route_start_time = time.time()
        self._route_warmup_until = self._route_start_time + ROUTE_START_HOLD
        self._last_control_time = self._route_start_time
        self._last_turn_time = self._route_start_time
        try:
            self.map_window.reset_trace()
        except Exception:
            pass
        return self.target_destination

    def closest_node(self, x, y, exclude=None):
        pool = [n for n in VALID_NODES if n != exclude] if exclude is not None else VALID_NODES
        def d(n):
            nx, ny, _ = self.node_pose(n)
            return math.hypot(nx - x, ny - y)
        return min(pool, key=d)

    # ---------------- Single step: perceive-plan-control ----------------
    def execute_trajectory_tracking(self, target_destination):
        self.animate_traffic_lights()

        now_control = time.time()
        self.dt = float(np.clip(now_control - getattr(self, "_last_control_time", now_control), 0.005, 0.05))
        self._last_control_time = now_control

        # At the start of every route, hold the vehicle and refresh the EKF from
        # QLabs. This removes the start-up left/right jerk without changing the
        # normal Pure Pursuit controller once the route is under way.
        if now_control < getattr(self, "_route_warmup_until", 0.0):
            status, loc, rot, _, _ = self.car.set_velocity_and_request_state(
                forward=0.0, turn=0.0, headlights=255,
                leftTurnSignal=0, rightTurnSignal=0, brakeSignal=255, reverseSignal=0)
            if status:
                self.localization.x = np.array([loc[0], loc[1], rot[2]], dtype=float)
                self.localization.P = np.eye(3) * 0.01
                try:
                    self.map_window.update(
                        pose=self.localization.x,
                        route_path=self.pure_pursuit.target_waypoints,
                        wp_index=self.pure_pursuit.current_wp_index,
                        target_destination=target_destination,
                        target_v=0.0,
                        steering=0.0,
                        state=self.state,
                        obstacle=False,
                    )
                except Exception:
                    pass
            return False

        current_pose = self.localization.x

        # Speed-adaptive Pure Pursuit lookahead: do not filter steering. Instead,
        # look farther ahead on faster straight exits to stop target-point hopping,
        # while keeping tighter preview automatically in slow/sharp bends.
        adaptive_ld = LOOKAHEAD_MIN + LOOKAHEAD_SPEED_GAIN * abs(float(getattr(self, "current_velocity", 0.0)))
        self.pure_pursuit.Ld = float(np.clip(adaptive_ld, LOOKAHEAD_MIN, LOOKAHEAD_MAX))

        pp_steer, target_reached = self.pure_pursuit.compute_steering(
            current_pose, self.pure_pursuit.target_waypoints)

        # Vision trim is disabled by default because it can fight the roadmap
        # controller on the small ACC map and cause start-up left/right wagging.
        vision_steering, vision_active = 0.0, False
        if USE_VISION_TRIM and self.perception.planner is not None and len(self.perception.lane_boundaries_xz) > 0:
            valid_x = [pt[0] for pt in self.perception.lane_boundaries_xz
                       if 0.3 <= pt[1] <= 1.5 and pt[0] < 0.85]
            if len(valid_x) > 5:
                error_x = 0.32 - float(np.mean(valid_x))
                vision_steering = float(np.clip(error_x * 0.15, -0.15, 0.15))
                vision_active = True

        if abs(pp_steer) > 0.25:
            steering = pp_steer
        elif vision_active:
            steering = float(np.clip(pp_steer + vision_steering, -0.58, 0.58))
        else:
            steering = pp_steer

        route_elapsed = time.time() - getattr(self, "_route_start_time", time.time())
        if route_elapsed < START_STEER_LIMIT_TIME:
            steering = float(np.clip(steering, -START_STEER_LIMIT, START_STEER_LIMIT))

        if target_reached:
            return True

        # ---- [Gap 5] Terminal deceleration curve ----
        dist_to_target = math.hypot(target_destination[0] - current_pose[0],
                                    target_destination[1] - current_pose[1])
        target_v = CRUISE_SPEED * math.cos(steering)
        if dist_to_target < DECEL_RADIUS:
            target_v = max(MIN_SPEED, CRUISE_SPEED * dist_to_target / DECEL_RADIUS)
        if dist_to_target < STOP_RADIUS:
            target_v = 0.0

        # Soft start for every newly loaded route: after the initial brake/sync
        # hold, ramp speed up smoothly instead of jumping straight to cruise.
        route_elapsed = time.time() - getattr(self, "_route_start_time", time.time())
        if route_elapsed < ROUTE_START_HOLD + ROUTE_START_RAMP:
            ramp = max(0.0, (route_elapsed - ROUTE_START_HOLD) / max(1e-6, ROUTE_START_RAMP))
            speed_cap = max(0.08, CRUISE_SPEED * min(1.0, ramp))
            target_v = min(target_v, speed_cap)

        # Speed-only stabilization: keep v11 steering response, but reduce speed
        # through corners and for a short moment after leaving a corner. This
        # lowers lateral overshoot without adding steering lag.
        if CURVE_SLOWDOWN:
            abs_steer_for_speed = abs(steering)
            if abs_steer_for_speed > SHARP_CURVE_STEER_THRESHOLD:
                target_v = min(target_v, SHARP_CURVE_SPEED_CAP)
                self._last_turn_time = time.time()
            elif abs_steer_for_speed > CURVE_STEER_THRESHOLD:
                target_v = min(target_v, CURVE_SPEED_CAP)
                self._last_turn_time = time.time()
            elif time.time() - getattr(self, "_last_turn_time", 0.0) < POST_TURN_RECOVERY_TIME:
                target_v = min(target_v, POST_TURN_SPEED_CAP)

        # ---- Traffic rules (geometric intersection + vision light color) ----
        heading = (current_pose[2] + math.pi) % (2 * math.pi) - math.pi
        x, y = current_pose[0], current_pose[1]
        is_east  = abs(heading) <= math.pi / 4
        is_west  = abs(heading) >= 3 * math.pi / 4
        is_north = math.pi / 4 < heading < 3 * math.pi / 4
        is_south = -3 * math.pi / 4 < heading < -math.pi / 4

        box_min_x, box_max_x = -0.651, 0.875
        box_min_y, box_max_y = 0.153, 1.853
        stop_dist = 0.75
        D = TL_DETECT_DIST  # [Enhancement] 2.0 m, prevents adjacent intersections from leaking light colors

        in_stop_zone = in_detection_zone = False
        if is_north and (box_min_x <= x <= box_max_x + 0.5):
            in_stop_zone      = box_min_y - stop_dist <= y <= box_min_y + 0.5
            in_detection_zone = box_min_y - D        <= y <= box_min_y + 0.5
        elif is_south and (box_min_x <= x <= box_max_x + 0.5):
            in_stop_zone      = box_max_y - 0.5 <= y <= box_max_y + stop_dist
            in_detection_zone = box_max_y - 0.5 <= y <= box_max_y + D
        elif is_east and (box_min_y <= y <= box_max_y + 0.5):
            in_stop_zone      = box_min_x - stop_dist <= x <= box_min_x + 0.5
            in_detection_zone = box_min_x - D        <= x <= box_min_x + 0.5
        elif is_west and (box_min_y <= y <= box_max_y + 0.5):
            in_stop_zone      = box_max_x - 0.5 <= x <= box_max_x + stop_dist
            in_detection_zone = box_max_x - 0.5 <= x <= box_max_x + D

        # ---- Camera frame: lane boundaries + sign detection ----
        detected_signs = []
        sign_info = {}
        success, image = self.car.get_image(self.car.CAMERA_RGB)
        if success and image is not None:
            if self.perception.planner is not None:
                try:
                    line_pts = self.perception.planner.line_detect(image)
                    self.perception.lane_boundaries_xz = [(px, pz) for (px, py, pz) in line_pts]
                except Exception:
                    self.perception.lane_boundaries_xz = []

            detected_signs, tl_color, sign_info, image = \
                self.perception.detect_traffic_signs_and_lights(image)
            if tl_color:
                self.last_seen_tl_color = tl_color
                self.last_seen_tl_time = time.time()
            imshow_safe("Camera View", image)
            if SHOW_WINDOWS:
                cv2.waitKey(1)

        signal_red = signal_yellow = False
        now = time.time()
        fresh_tl = (now - self.last_seen_tl_time) <= TL_MEMORY_TIME
        if in_detection_zone and fresh_tl:
            if self.last_seen_tl_color == "RED":
                signal_red = True
            elif self.last_seen_tl_color == "YELLOW":
                signal_yellow = True
        else:
            self.last_seen_tl_color = None

        in_stop_sign_zone  = "STOP" in detected_signs
        stop_sign_area = float(sign_info.get("STOP", {}).get("area", 0.0))

        # Latching stop-sign detector. Once the sign reaches the pre-arm size,
        # keep that context even if YOLO drops a frame or the sign slides out of
        # the side of the camera view before the box reaches the old hard trigger.
        if in_stop_sign_zone:
            self._stop_sign_last_seen_time = now
            self._stop_sign_max_area = max(getattr(self, "_stop_sign_max_area", 0.0), stop_sign_area)
            if stop_sign_area >= STOP_SIGN_ARM_AREA:
                self._stop_sign_armed = True

        stop_sign_close = in_stop_sign_zone and stop_sign_area >= STOP_SIGN_AREA_TRIGGER
        armed_sign_lost = (getattr(self, "_stop_sign_armed", False) and
                           not in_stop_sign_zone and
                           0.0 < now - getattr(self, "_stop_sign_last_seen_time", 0.0) <= STOP_SIGN_LOST_TIME and
                           now >= getattr(self, "_stop_sign_release_until", 0.0))
        stop_sign_trigger = (stop_sign_close or armed_sign_lost)

        in_yield_zone      = "YIELD" in detected_signs
        in_roundabout_zone = "ROUNDABOUT" in detected_signs

        if in_stop_zone:
            if fresh_tl and self.last_seen_tl_color == "GREEN":
                self.clearing_intersection = True
            elif fresh_tl and self.last_seen_tl_color in ("RED", "YELLOW"):
                self.clearing_intersection = False
        else:
            self.clearing_intersection = False

        if in_stop_zone and not self.clearing_intersection and (signal_red or signal_yellow):
            target_v = 0.0
            print("[TRAFFIC] Red/Yellow light, holding...            ", end="\r")
        elif in_stop_zone and self.clearing_intersection:
            # Positive green release: once green is freshly seen for this approach,
            # keep rolling through the intersection even if detection flickers.
            target_v = max(target_v, MIN_SPEED)
            print("[TRAFFIC] Green light, clearing intersection...   ", end="\r")
        elif self._stop_sign_active or stop_sign_trigger:
            # Stop-sign rule: once triggered by either a close box or armed+lost,
            # keep holding for the full dwell time even if YOLO misses the sign.
            if (not self._stop_sign_active) and stop_sign_trigger and now >= self._stop_sign_release_until:
                self._stop_sign_active = True
                self._stop_sign_done = False
                self._stop_sign_start = now
                reason = "close" if stop_sign_close else "armed-lost"
                print(f"\n[TRAFFIC] Stop sign triggered ({reason}), area={stop_sign_area:.0f}, "
                      f"max={getattr(self, '_stop_sign_max_area', 0.0):.0f}")
            if self._stop_sign_active and not self._stop_sign_done:
                remaining = STOP_SIGN_FULL_STOP_TIME - (now - self._stop_sign_start)
                if remaining > 0:
                    target_v = 0.0
                    print(f"[TRAFFIC] Stop sign full stop: {remaining:.1f}s left, "
                          f"area={stop_sign_area:.0f}, max={getattr(self, '_stop_sign_max_area', 0.0):.0f}...", end="\r")
                else:
                    self._stop_sign_done = True
                    self._stop_sign_release_until = now + STOP_SIGN_COOLDOWN
                    self._stop_sign_armed = False
                    self._stop_sign_max_area = 0.0
                    target_v = max(target_v, MIN_SPEED)
                    print("[TRAFFIC] Stop sign done, proceeding...        ", end="\r")
        elif in_stop_sign_zone:
            # Stop sign is visible but not close enough yet. Arm once it passes the
            # pre-warning area and approach slowly so either close-box or out-of-frame
            # loss will still produce exactly one full stop.
            target_v = min(target_v, 0.22)
            armed_txt = "armed" if getattr(self, "_stop_sign_armed", False) else "visible"
            print(f"[TRAFFIC] Approaching stop sign ({armed_txt}) area={stop_sign_area:.0f}, "
                  f"max={getattr(self, '_stop_sign_max_area', 0.0):.0f}...", end="\r")
        elif in_yield_zone:
            # Yield does NOT require a full stop when the way is clear.
            # Keep moving slowly and be ready to brake only for real obstacles/red lights.
            yield_cap = max(MIN_SPEED, YIELD_SPEED * max(0.65, math.cos(steering)))
            target_v = min(target_v, yield_cap)
            print("[TRAFFIC] Yield sign, slowing but not stopping...  ", end="\r")
        elif in_roundabout_zone:
            # Roundabout entry/inside: slow down, but do not create a stop timer.
            # A full stop here causes rear-end/creep oscillations and is not needed in the ACC map.
            rb_cap = max(MIN_SPEED, ROUNDABOUT_SPEED * max(0.65, math.cos(steering)))
            target_v = min(target_v, rb_cap)
            print("[TRAFFIC] Roundabout, slowing but rolling...      ", end="\r")

        if (not in_stop_sign_zone) and self._stop_sign_active and now >= self._stop_sign_release_until:
            self._stop_sign_active = False
            self._stop_sign_done = False
            self._stop_sign_armed = False
            self._stop_sign_max_area = 0.0
        elif (not in_stop_sign_zone and
              not self._stop_sign_active and
              now - getattr(self, "_stop_sign_last_seen_time", 0.0) > STOP_SIGN_LOST_TIME):
            # Stale armed state that did not represent a pass-by event.
            self._stop_sign_armed = False
            self._stop_sign_max_area = 0.0
        if not in_yield_zone and hasattr(self, "yield_arrival_time"):
            delattr(self, "yield_arrival_time")
        if not in_roundabout_zone and hasattr(self, "roundabout_arrival_time"):
            delattr(self, "roundabout_arrival_time")

        # ---- [Enhancement] LiDAR obstacle: stop and wait, with anti-deadlock creep ----
        obstacle_detected, obstacle_y, min_front = \
            self.perception.process_lidar_for_obstacles(steering=steering)

        # In yield/roundabout areas the LiDAR often sees static map geometry
        # (curb, roundabout island, sign pole) inside the short forward corridor.
        # Do not convert those expected close returns into stop-creep-stop jitter.
        # Still brake if something is dangerously close directly in front.
        # Ignore expected static map geometry around signs/intersections. In the ACC
        # map the LiDAR often sees stop-sign poles, traffic-light poles, curb blocks,
        # or the roundabout island inside the forward box. Treat those as geometry,
        # not as a dynamic obstacle, unless they are literally at the bumper.
        in_traffic_geometry_zone = (in_yield_zone or in_roundabout_zone or
                                    in_stop_sign_zone or in_stop_zone or in_detection_zone)
        emergency_obstacle = obstacle_detected and min_front < OBSTACLE_EMERGENCY_DIST
        obstacle_hold_allowed = OBSTACLE_HOLD and (not in_traffic_geometry_zone or emergency_obstacle)

        if obstacle_detected and obstacle_hold_allowed:
            now = time.time()
            if not getattr(self, "_obstacle_holding", False):
                reason = "emergency close obstacle" if emergency_obstacle else "in-lane obstacle"
                print(f"\n[OBSTACLE] {reason} at y={obstacle_y:+.2f} m, d={min_front:.2f} m; holding...")
                self._obstacle_holding = True
                self._hold_since = now
            # Anti-deadlock: if we've been held for a while and there is still
            # frontal clearance, creep along the planned path instead of freezing forever.
            if now - self._hold_since > HOLD_CREEP_AFTER and min_front > CREEP_CLEARANCE:
                target_v = MIN_SPEED
                print(f"\n[OBSTACLE] Static for {HOLD_CREEP_AFTER:.0f}s with "
                      f"{min_front:.2f}m clearance -> creeping along path...")
                self._hold_since = now
            else:
                target_v = 0.0
        elif obstacle_detected and in_traffic_geometry_zone:
            # Expected static geometry near yield/roundabout: keep rolling at the already-capped speed.
            if getattr(self, "_obstacle_holding", False):
                print("\n[OBSTACLE] In yield/roundabout geometry zone; releasing hold and rolling.")
                self._obstacle_holding = False
        elif getattr(self, "_obstacle_holding", False):
            print("\n[OBSTACLE] Cleared, resuming.")
            self._obstacle_holding = False

        # ---- Actuate ----
        self.current_velocity = target_v
        self.last_steering_cmd = steering
        actual_steer = -float(steering)  # QLabs virtual car steering sign is opposite to map coordinates

        status, loc, rot, _, _ = self.car.set_velocity_and_request_state(
            forward=float(target_v), turn=actual_steer,
            headlights=255, leftTurnSignal=0, rightTurnSignal=0,
            brakeSignal=255 if target_v == 0.0 else 0, reverseSignal=0)
        if not status:
            return False

        # ---- [Gap 4] EKF: omega = v*tan(delta)/L ----
        omega = target_v * math.tan(steering) / self.pure_pursuit.L
        self.localization.predict(target_v, omega, self.dt)
        self.localization.update(np.array([loc[0], loc[1], rot[2]]))
        pose = self.localization.x

        # Roadmap debug visualization only. Does not affect vehicle control.
        try:
            self.map_window.update(
                pose=pose,
                route_path=self.pure_pursuit.target_waypoints,
                wp_index=self.pure_pursuit.current_wp_index,
                target_destination=target_destination,
                target_v=target_v,
                steering=steering,
                state=self.state,
                obstacle=getattr(self, "_obstacle_holding", False),
            )
        except Exception:
            pass

        print(f"Pose [X:{pose[0]:+.2f} Y:{pose[1]:+.2f} Yaw:{pose[2]:+.2f}] "
              f"v={target_v:.2f} d_goal={dist_to_target:.2f}    ", end="\r")

        return math.hypot(target_destination[0] - pose[0],
                          target_destination[1] - pose[1]) < STOP_RADIUS + 0.05

    # ---------------- Full stop + dwell ----------------
    def full_stop(self, seconds=0.0):
        self.car.set_velocity_and_request_state(0.0, 0.0, 255, 0, 0, 255, 0)
        self.current_velocity = 0.0
        if seconds > 0:
            time.sleep(seconds)

    # ---------------- Main loop ----------------
    def run(self):
        if not self.setup_sim():
            return

        print("\n==================================")
        print("  ACC TAXI MISSION INITIALIZED    ")
        print("==================================")

        try:
            while True:
                # ---------- Idle at hub: spawn passenger + patrol route ----------
                if self.state == FSMState.STATE_IDLE_HUB:
                    print("\n[IDLE_HUB] Preparing next ride...")
                    self.set_led(LED_IDLE)
                    time.sleep(2.0)

                    if not hasattr(self, "person"):
                        self.person = QLabsPerson(self.qlabs)
                    patrol_pool = [0, 4, 14, 20, 23] if DIRECT_TO_KNOWN_PASSENGER else [0, 2, 4, 8, 14, 20, 23]
                    if not self.first_person_spawned:
                        self.spawn_node = 4
                        self.first_person_spawned = True
                    else:
                        self.spawn_node = random.choice(patrol_pool)

                    nx, ny, yaw = self.node_pose(self.spawn_node)
                    off = 0.15
                    self.person.spawn_id(actorNumber=0,
                        location=[nx + off * math.cos(yaw - math.pi / 2),
                                  ny + off * math.sin(yaw - math.pi / 2), 0.1],
                        rotation=[0, 0, 0], scale=[0.1]*3,
                        configuration=0, waitForConfirmation=True)
                    print(f"[IDLE_HUB] Passenger spawned near Node {self.spawn_node}")

                    self.patrol_route = [0, 4, 14, 23, 20, 10]
                    self.patrol_index = 0
                    self.current_patrol_node = HUB_NODE
                    self.patrol_start_time = time.time()
                    self.set_led(LED_DRIVING)

                    if DIRECT_TO_KNOWN_PASSENGER:
                        # The passenger is spawned by this script, so do not wait for a potentially
                        # false YOLO person detection. Go directly to the known pickup node.
                        print(f"[IDLE_HUB] Direct shortest route to pickup Node {self.spawn_node}")
                        # Ask the roadmap for the shortest HUB -> pickup route first.
                        # Only fall back through Node 0 if the direct request is not usable.
                        try:
                            self.load_route([HUB_NODE, self.spawn_node])
                        except RuntimeError:
                            print(f"[IDLE_HUB] Direct route failed; falling back via Node 0 to Node {self.spawn_node}")
                            self.load_route([HUB_NODE, 0, self.spawn_node])
                        self.state = FSMState.STATE_NAV_TO_PICKUP
                    else:
                        self.current_patrol_node = self.patrol_route[0]
                        self.load_route([HUB_NODE, self.current_patrol_node])
                        self.state = FSMState.STATE_PATROLLING

                # ---------- Patrolling for passengers ----------
                elif self.state == FSMState.STATE_PATROLLING:
                    person_bbox = None
                    if self.perception:
                        with self.perception.vision_lock:
                            person_bbox = self.perception.latest_person_bbox

                    # [Enhancement] YOLO unavailable or patrol timeout -> route straight to the passenger spawn node
                    yolo_ok = self.perception.person_yolo_model is not None
                    timed_out = (time.time() - self.patrol_start_time) > 90.0
                    if (not yolo_ok) or timed_out:
                        reason = "no person-YOLO" if not yolo_ok else "patrol timeout"
                        print(f"\n[PATROLLING] Fallback ({reason}): routing to spawn "
                              f"Node {self.spawn_node} directly")
                        pose = self.localization.x
                        start = self.closest_node(pose[0], pose[1])
                        self.load_route([start, self.spawn_node])
                        self.state = FSMState.STATE_NAV_TO_PICKUP

                    elif USE_PERSON_YOLO_REROUTE and person_bbox is not None:
                        px1, py1, px2, py2 = person_bbox
                        center_u = (px1 + px2) / 2
                        angle_rad = -((center_u - 320) / 320.0) * (41.0 * math.pi / 180.0)

                        success, angles, distances = self.car.get_lidar(samplePoints=400)
                        if success:
                            # [Enhancement] median ranging over +/-3 degrees
                            dist = self.perception.lidar_median_distance(
                                angles, distances, angle_rad)
                            if dist is not None and dist < 30.0:
                                pose = self.localization.x
                                p_x = pose[0] + dist * math.cos(pose[2] + angle_rad)
                                p_y = pose[1] + dist * math.sin(pose[2] + angle_rad)
                                print(f"\n[PATROLLING] Passenger at ({p_x:.2f},{p_y:.2f}) "
                                      f"dist={dist:.2f}m -> rerouting")
                                target_node = self.closest_node(p_x, p_y)
                                pose_now = self.localization.x
                                start_node = self.closest_node(pose_now[0], pose_now[1])
                                self.load_route([start_node, target_node])
                                with self.perception.vision_lock:
                                    self.perception.latest_person_bbox = None
                                self.state = FSMState.STATE_NAV_TO_PICKUP

                    if self.state == FSMState.STATE_PATROLLING:
                        reached = self.execute_trajectory_tracking(self.target_destination)
                        if reached:
                            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_route)
                            nxt = self.patrol_route[self.patrol_index]
                            self.load_route([self.current_patrol_node, nxt])
                            self.current_patrol_node = nxt
                    time.sleep(self.dt)

                # ---------- Navigating to pickup ----------
                elif self.state == FSMState.STATE_NAV_TO_PICKUP:
                    self.execute_trajectory_tracking(self.target_destination)
                    pose = self.localization.x
                    d = math.hypot(self.target_destination[0] - pose[0],
                                   self.target_destination[1] - pose[1])
                    # The passenger standing at the curb often enters the
                    # obstacle corridor before we reach the exact node and
                    # would deadlock the pickup (the passenger blocks their
                    # own taxi). Being held by an obstacle close to the target
                    # IS arrival -- the obstacle is the passenger.
                    held_near_target = (getattr(self, "_obstacle_holding", False)
                                        and d < 0.8)
                    if d < PICKUP_SWITCH or held_near_target:
                        if held_near_target and d >= PICKUP_SWITCH:
                            print(f"\n[PICKUP] Held {d:.2f} m from the node by the "
                                  f"waiting passenger -> treating as arrival.")
                        self._obstacle_holding = False
                        self.state = FSMState.STATE_PASSENGER_LOADING
                    time.sleep(self.dt)

                # ---------- Passenger boarding ----------
                elif self.state == FSMState.STATE_PASSENGER_LOADING:
                    print("\n[LOADING] At pickup point. Parking brake ON, LED=GREEN.")
                    self.set_led(LED_PICKUP)
                    self.full_stop()
                    # [Enhancement] passenger AI-pathfinds to the door before boarding (includes the officially required dwell)
                    self.passenger_board()
                    time.sleep(3.0)

                    if RANDOM_DROPOFF:
                        pose = self.localization.x
                        here = self.closest_node(pose[0], pose[1])
                        self.current_dropoff_node = random.choice(
                            [n for n in VALID_NODES if n != here])
                    else:
                        self.current_dropoff_node = DROPOFF_NODE

                    pose = self.localization.x
                    start = self.closest_node(pose[0], pose[1])
                    print(f"[LOADING] Route to drop-off: Node {start} -> "
                          f"Node {self.current_dropoff_node}")
                    self.load_route([start, self.current_dropoff_node])
                    self.set_led(LED_DRIVING)
                    self.state = FSMState.STATE_NAV_TO_DROPOFF

                # ---------- Navigating to drop-off ----------
                elif self.state == FSMState.STATE_NAV_TO_DROPOFF:
                    reached = self.execute_trajectory_tracking(self.target_destination)
                    if reached:
                        print("\n[DROPOFF] At drop-off point. LED=BLUE.")
                        self.set_led(LED_DROPOFF)
                        self.full_stop(3.0)  # official scenario: stop 3 s to signal drop-off
                        # [Enhancement] passenger appears at the door and walks to the sidewalk
                        self.passenger_alight()
                        print("[DROPOFF] Returning to hub.")

                        if self.current_dropoff_node == HUB_NODE:
                            self.pure_pursuit.target_waypoints = [list(self.target_destination)]
                            self.pure_pursuit.current_wp_index = 0
                        else:
                            self.load_route([self.current_dropoff_node, HUB_NODE])
                        self.set_led(LED_DRIVING)
                        self.state = FSMState.STATE_RETURN_TO_HUB
                    time.sleep(self.dt)

                # ---------- Returning to hub ----------
                elif self.state == FSMState.STATE_RETURN_TO_HUB:
                    reached = self.execute_trajectory_tracking(self.target_destination)
                    if reached:
                        print("\n[HUB] Parked at Taxi Hub. Ride complete.")
                        self.full_stop()
                        self.set_led(LED_IDLE)
                        self.state = FSMState.STATE_IDLE_HUB
                    time.sleep(self.dt)

        except KeyboardInterrupt:
            print("\nMission aborted by user.")
        finally:
            print("Cleaning up vehicle state...")
            if self.car:
                try:
                    self.car.set_velocity_and_request_state(0.0, 0.0, 0, 0, 0, 255, 0)
                except Exception:
                    pass
            if SHOW_WINDOWS:
                cv2.destroyAllWindows()
            self.qlabs.close()
            print("Simulation closed.")


if __name__ == "__main__":
    AutonomousTaxiMission().run()