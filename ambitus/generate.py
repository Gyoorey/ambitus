"""
Synthetic validation suite for traj_filter.py and feature_extractor.py.

What changed compared with the previous generate.py
----------------------------------------------------
1. Gradual turns. Paths with turns are produced by a simple rat-like walker:
   cruising speed, a limited turn rate (a 180-degree turn takes about 0.6 s)
   and slowing down while turning. U-turns are no longer instantaneous.
2. Ground truth is measured on the generated path itself (distances,
   CW/CCW distances, number of completed U-turns), not hard-coded.
   CW/CCW truth: each frame's true direction comes from the intended route
   leg, and is then passed through the extractor's own 1-second majority vote
   and High-Active gate. The test then checks the per-frame direction rule
   (review item 2.3), not the voting design. The per-frame truth before
   voting is stored in the *_ground_truth.json files as "_Per_Frame_Truth_*".
   A U-turn counts once the walker's heading has reversed, even if a rest
   interrupts the turn (this happens in test3).
3. Noise variants: tracker noise + blob wobble, then the same smoothing as
   traj_filter (median 5 + Savitzky-Golay 31/3), then integer rounding.
4. New tests: U-turn speed sweep (test6) and a loop without reversals that
   must give 0 U-turns (test7, false-positive check).
5. CW/CCW rule comparison (review item 2.3): a copy of feature_extractor
   section 6 is run with the current rule (angle > 0.5 deg/frame) and with
   the proposed tangential rule, so you can see the effect before changing
   feature_extractor.py.
6. Tolerances per feature type. The old max(1.5, 3%) could never fail for
   values between 0 and 1 (e.g. the asymmetry index).
7. Fixed random seed, no files created on import, tests run from a list
   (not from whatever *_ground_truth.json happens to be in the folder).
8. Activity frame sync (review item 2.5, test8): a synthetic 20 s video in which
   the rat moves in place (Low Active) only at 10-12 s is classified by
   activity.py twice: with every row, and with the rows of 4-6 s removed (as
   traj_filter does for a rejected wall-crossing gap). Both must report Low
   Active at 10-12 s and the same state at every shared timestamp. Runs for
   .mp4 (OpenCV) and .mpg (imageio; skipped if ffmpeg is not available).
9. Zone background (tracker.py, bg_method='zone', test9): a synthetic 100 s
   video in which a hand places a food stick and drops the rat in the
   top-right corner; the rat walks to the bottom-right corner and sits there
   almost still, except for two short runs. Tracking starts at 30 s (as with
   --backtrack, the rat is already sitting). The tracker must find the rat in
   >= 99% of the tracked frames. The plain 'global' median (rat baked
   into the bottom-right corner) is shown for comparison only.

Usage
-----
    python generate.py                  # full suite, 3 noise seeds
    python generate.py --seeds 5        # more noise seeds
    python generate.py --no-noise       # clean paths only
    python generate.py --only test6     # only tests whose name contains "test6"
    python generate.py --only test8     # only the activity frame-sync test
    python generate.py --only test9     # only the zone-background tracking test (about 2 minutes)
    python generate.py --html           # also write the extractor's debug dashboards

Exit code is 1 if any feature check fails. The CW/CCW comparison table is
informational and does not affect the exit code.
"""

import argparse
import csv
import json
import math
import os
import sys

import shutil

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from traj_filter import process_trajectory
from feature_extractor import extract_features, get_arena_center, clean_poly, normalize_angle_diff, calculate_angle

OUTPUT_DIR = "./synthetic_test_suite"
SEED = 12345

# Given arena geometry
CONFIG = {
    "outer_poly": [[200, 44], [170, 450], [595, 470], [608, 68], [200, 44]],
    "inner_poly": [[254, 90], [232, 403], [547, 422], [560, 106], [254, 90]]
}

# Key waypoint nodes in the 4 corridors between inner and outer walls
WAYPOINTS = {
    "North_Corridor": np.array([400.0, 75.0]),
    "East_Corridor":  np.array([580.0, 260.0]),
    "South_Corridor": np.array([390.0, 440.0]),
    "West_Corridor":  np.array([200.0, 250.0]),
    "Top_Left_Corner": np.array([215.0, 65.0]),
    "Top_Right_Corner": np.array([585.0, 85.0]),
    "Bottom_Right_Corner": np.array([575.0, 445.0]),
    "Bottom_Left_Corner": np.array([190.0, 430.0])
}

FPS = 25.0
DT = 1.0 / FPS
TOTAL_PROTOCOL_TIME = 300.0  # Strict 5-minute standard
N_FRAMES = int(round(TOTAL_PROTOCOL_TIME * FPS))

# --- Walker (gradual turns) ---
TURN_RATE_RAD_S = math.pi / 0.6   # 180 degrees of pure rotation take ~0.6 s
MIN_TURN_SPEED_FACTOR = 0.4       # speed drops to 40% at the start of a reversal
REVERSAL_ANGLE_DEG = 135.0        # a target switch sharper than this is a U-turn
TURN_DONE_DEG = 20.0              # U-turn complete when heading is within this of the new direction

# --- Noise model (applied to the clean path) ---
NOISE_SIGMA_PX = 2.0              # white tracker noise
WOBBLE_AMP_PX = 3.0               # slow blob-centre drift, 1-3 Hz
MEDIAN_WINDOW = 5                 # traj_filter defaults
SAVGOL_WINDOW = 31

# --- CW/CCW rules compared in the replica (review item 2.3) ---
CURRENT_ANGLE_THRESHOLD_DEG = 0.5  # current feature_extractor rule, per frame
TANG_WINDOW = 5                    # proposed rule: frames
MIN_TANG_PX = 2.0                  # proposed rule: tangential movement over the window

# --- Tolerances ---
COUNT_KEYS = {"Total_Activity_Number", "Total_Inactivity_Number", "Short_Active_Number",
              "Long_Active_Number", "Short_Inactivity_Number", "Long_Inactivity_Number", "U_Turn_Count"}
INDEX_KEYS = {"Directional_Asymmetry_Index"}
DIRECTION_DISTANCE_KEYS = {"Clockwise_Distance_px", "CounterClockwise_Distance_px"}

ARENA_CENTER = get_arena_center(clean_poly(np.array(CONFIG["inner_poly"], dtype=np.int32)),
                                clean_poly(np.array(CONFIG["outer_poly"], dtype=np.int32)))

RNG = np.random.default_rng(SEED)


def tolerance(key, expected, noisy=False):
    """Counts must match exactly. Noisy variants get wider direction tolerances: noise can move a
    reversal across a 1-second voting block, which flips up to one block of movement (~60 px at
    60 px/s) between CW and CCW."""
    if key in COUNT_KEYS:
        return 0
    if key in INDEX_KEYS:
        return max(0.05 if noisy else 0.02, (0.10 if noisy else 0.05) * abs(expected))
    if key in DIRECTION_DISTANCE_KEYS:
        return max(1.5, (0.10 if noisy else 0.05) * abs(expected))
    return max(1.5, 0.03 * abs(expected))


# ==============================================================================
# PATH SYNTHESIS
# ==============================================================================
def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def leg_label(p_from, p_to):
    """+1 if the leg runs clockwise around the arena centre (on screen, y axis down), -1 otherwise."""
    mid = (np.asarray(p_from) + np.asarray(p_to)) / 2.0
    r = mid - np.array(ARENA_CENTER, float)
    d = np.asarray(p_to) - np.asarray(p_from)
    cross = r[0] * d[1] - r[1] * d[0]
    return 1 if cross > 0 else -1


class Walker:
    """Rat-like mover: cruising speed, limited turn rate, slows down while turning."""

    def __init__(self, pos, heading):
        self.pos = np.array(pos, float)
        self.heading = heading

    def step(self, target, speed):
        to_t = np.asarray(target) - self.pos
        desired = math.atan2(to_t[1], to_t[0])
        diff = wrap_angle(desired - self.heading)
        if abs(diff) > math.radians(170):
            # Near-reversal: turn towards the arena centre side (stays inside the corridor)
            c = np.array(ARENA_CENTER, float) - self.pos
            side = math.cos(self.heading) * c[1] - math.sin(self.heading) * c[0]
            diff = abs(diff) if side >= 0 else -abs(diff)
        max_turn = TURN_RATE_RAD_S * DT
        self.heading = wrap_angle(self.heading + max(-max_turn, min(max_turn, diff)))
        factor = MIN_TURN_SPEED_FACTOR + (1.0 - MIN_TURN_SPEED_FACTOR) * math.cos(diff / 2.0) ** 2
        self.pos = self.pos + speed * factor * DT * np.array([math.cos(self.heading), math.sin(self.heading)])


class Route:
    """Follows a list of targets with a Walker and records the ground truth of each step."""

    def __init__(self, start, targets, speed, cycle=False):
        self.targets = [np.asarray(t, float) for t in targets]
        self.speed = speed
        self.cycle = cycle
        self.idx = 0
        self.leg_start = np.asarray(start, float)
        first = self.targets[0] - self.leg_start
        self.walker = Walker(start, math.atan2(first[1], first[0]))
        self.finished = False
        self.pending_turn = False
        self.u_turn_times = []
        # Capture radius > turning diameter, so the walker never circles around a target
        self.capture = 2.2 * speed / TURN_RATE_RAD_S + speed * DT

    def _leg_dir(self):
        d = self.targets[self.idx] - self.leg_start
        return d / (np.linalg.norm(d) + 1e-12)

    def advance(self, t):
        """Moves one frame. Returns (step_vector, cw_label_of_this_step)."""
        if self.finished:
            return np.zeros(2), 0
        target = self.targets[self.idx]
        if np.linalg.norm(target - self.walker.pos) < self.capture:
            old_dir = self._leg_dir()
            nxt = self.idx + 1
            if nxt >= len(self.targets):
                if not self.cycle:
                    self.finished = True
                    return np.zeros(2), 0
                nxt = 0
            self.leg_start, self.idx = target, nxt
            new_dir = self._leg_dir()
            angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(old_dir, new_dir))))))
            if angle > REVERSAL_ANGLE_DEG:
                self.pending_turn = True
        prev = self.walker.pos.copy()
        self.walker.step(self.targets[self.idx], self.speed)
        step = self.walker.pos - prev
        leg_dir = self._leg_dir()
        label = leg_label(self.leg_start, self.targets[self.idx])
        if np.dot(step, leg_dir) < 0:
            label = -label  # still finishing the previous (opposite) leg during a reversal
        if self.pending_turn:
            to_t = self.targets[self.idx] - self.walker.pos
            if abs(wrap_angle(math.atan2(to_t[1], to_t[0]) - self.walker.heading)) < math.radians(TURN_DONE_DEG):
                self.u_turn_times.append(round(t, 2))
                self.pending_turn = False
        return step, label


class Recorder:
    """Builds the 4-column rows (Time, X, Y, State) and the CW/CCW ground truth."""

    def __init__(self, pos):
        self.rows = []
        self.labels = []   # true direction of the step that ends at this row (+1 CW, -1 CCW, 0 none)
        self.pos = np.array(pos, float)
        self.cw = 0.0      # per-frame truth (before the extractor's 1-s majority vote)
        self.ccw = 0.0

    def t(self):
        return len(self.rows) * DT

    def move(self, route, seconds=None, until_finished=False, state=2):
        n = int(round(seconds * FPS)) if seconds is not None else N_FRAMES
        for _ in range(n):
            if until_finished and route.finished:
                break
            if len(self.rows) >= N_FRAMES:
                break
            step, label = route.advance(self.t())
            if until_finished and route.finished:
                break
            self.pos = route.walker.pos.copy()
            if state == 2:
                if label > 0:
                    self.cw += float(np.linalg.norm(step))
                elif label < 0:
                    self.ccw += float(np.linalg.norm(step))
            self.rows.append([self.t(), self.pos[0], self.pos[1], state])
            self.labels.append(label if state == 2 else 0)

    def start(self, state=2):
        self.rows.append([0.0, self.pos[0], self.pos[1], state])
        self.labels.append(0)

    def rest(self, seconds=None, state=0):
        n = int(round(seconds * FPS)) if seconds is not None else N_FRAMES - len(self.rows)
        for _ in range(n):
            if len(self.rows) >= N_FRAMES:
                break
            self.rows.append([self.t(), self.pos[0], self.pos[1], state])
            self.labels.append(0)

    def jitter(self, seconds, state=1, amp=0.1):
        for _ in range(int(round(seconds * FPS))):
            j = RNG.uniform(-amp, amp, size=2)
            self.rows.append([self.t(), self.pos[0] + j[0], self.pos[1] + j[1], state])
            self.labels.append(0)


def rows_total_distance(rows, decimals=2):
    xy = np.round(np.array([[r[1], r[2]] for r in rows]), decimals)
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def asym(cw, ccw):
    return abs(cw - ccw) / (cw + ccw) if (cw + ccw) > 0 else 0.0


def direction_gt(rec, route):
    """CW/CCW ground truth = true per-frame direction (from the intended route legs), passed through the
    extractor's own 1-second majority vote and High-Active gate. This isolates the per-frame direction
    rule (review item 2.3) from the voting design, which can move up to ~0.5 s of movement around each
    reversal to the other direction. The per-frame truth before voting is stored for reference."""
    rows = np.round(np.array([[r[0], r[1], r[2]] for r in rec.rows]), 4)
    xs, ys = np.round(rows[:, 1], 2), np.round(rows[:, 2], 2)
    raw_states = np.array([int(r[3]) for r in rec.rows])
    step_labels = np.array(rec.labels[1:] + [0])  # label of step i -> i+1 sits on row i+1
    cw, ccw = direction_pipeline(rows[:, 0], xs, ys, raw_states, step_labels)
    return {
        "Clockwise_Distance_px": round(cw, 1),
        "CounterClockwise_Distance_px": round(ccw, 1),
        "Directional_Asymmetry_Index": round(asym(cw, ccw), 4),
        "U_Turn_Count": len(route.u_turn_times),
        "_Per_Frame_Truth_CW_px": round(rec.cw, 1),
        "_Per_Frame_Truth_CCW_px": round(rec.ccw, 1),
    }


# ==============================================================================
# TESTS
# ==============================================================================
def gen_test_1():
    """Absolute resting (300 s, distance 0). Unchanged."""
    pt = WAYPOINTS["Top_Left_Corner"]
    rows = [[i * DT, pt[0], pt[1], 0] for i in range(N_FRAMES)]
    gt = {
        "Total_Distance_px": 0.0, "Total_Combined_Activity_Time_s": 0.0, "Total_Inactivity_Time_s": 300.0,
        "Total_Activity_Number": 0, "Total_Inactivity_Number": 1, "Short_Active_Number": 0, "Long_Active_Number": 0,
        "Short_Inactivity_Number": 0, "Long_Inactivity_Number": 1, "Corner_Time_Ratio_pct": 100.0,
        "Directional_Asymmetry_Index": 0.0, "U_Turn_Count": 0
    }
    return {"name": "test1_still_resting", "rows": rows, "gt": gt}


def gen_test_2():
    """One CW lap, reversal at the top-left corner, CCW legs, reversal at the bottom-right corner, CW legs.
    Gradual turns; then rest until 300 s."""
    W = WAYPOINTS
    nodes = [W["Top_Left_Corner"], W["North_Corridor"], W["Top_Right_Corner"], W["East_Corridor"],
             W["Bottom_Right_Corner"], W["South_Corridor"], W["Bottom_Left_Corner"], W["West_Corridor"],
             W["Top_Left_Corner"], W["West_Corridor"], W["Bottom_Left_Corner"], W["South_Corridor"],
             W["Bottom_Right_Corner"], W["South_Corridor"], W["Bottom_Left_Corner"]]
    route = Route(nodes[0], nodes[1:], speed=60.0)
    rec = Recorder(nodes[0])
    rec.start(state=2)
    rec.move(route, until_finished=True, state=2)
    lap_time = rec.t()
    rec.rest()
    gt = {
        "Total_Distance_px": round(rows_total_distance(rec.rows), 1),
        "Total_Combined_Activity_Time_s": round(lap_time, 1),
        "Total_Activity_Number": 1, "Total_Inactivity_Number": 1,
        "Short_Active_Number": 0, "Long_Active_Number": 1,
        "Short_Inactivity_Number": 0, "Long_Inactivity_Number": 1,
        **direction_gt(rec, route),
    }
    return {"name": "test2_cw_continuous_circuits", "rows": rec.rows, "gt": gt,
            "u_turn_times": route.u_turn_times}


def gen_test_3():
    """10 cycles x 30 s: short active, short rest, long active (shuttle with U-turns), long rest,
    micro-movement, short rest. Same phase timing as before; turns are gradual now.
    Some U-turns end right at a phase boundary or are interrupted by a rest (e.g. one starts at
    ~167 s and finishes at 180 s); these check that the detector handles turns cut by stops."""
    p_left = np.array([215.0, 65.0])
    p_right = np.array([585.0, 85.0])
    speed = np.linalg.norm(p_right - p_left) / 6.0  # 61.75 px/s
    route = Route(p_left, [p_right, p_left], speed=speed, cycle=True)
    rec = Recorder(p_left)
    for _ in range(10):
        rec.move(route, 3.0, state=2)
        rec.rest(2.0)
        rec.move(route, 12.0, state=2)
        rec.rest(8.0)
        rec.jitter(3.0, state=1)
        rec.rest(2.0)
    gt = {
        "Total_Combined_Activity_Time_s": 180.0, "Total_Inactivity_Time_s": 120.0,
        "Total_Activity_Number": 30, "Total_Inactivity_Number": 30,
        "Short_Active_Number": 20, "Long_Active_Number": 10,
        "Short_Inactivity_Number": 20, "Long_Inactivity_Number": 10,
        "Total_Distance_px": round(rows_total_distance(rec.rows), 1),
        **direction_gt(rec, route),
    }
    return {"name": "test3_phase_clockwork", "rows": rec.rows, "gt": gt, "u_turn_times": route.u_turn_times}


def gen_test_4():
    """Corner preference & low activity. Unchanged (straight dashes, no U-turn checks)."""
    rows = []
    c_pt = WAYPOINTS["Bottom_Right_Corner"]
    dash_pt = WAYPOINTS["South_Corridor"]
    t = 0.0
    num_cycles = 6
    one_way_dist = np.linalg.norm(dash_pt - c_pt)
    expected_dist = num_cycles * (2.0 * one_way_dist)
    for _ in range(num_cycles):
        for _ in range(int(45.0 * FPS)):
            rows.append([t, c_pt[0], c_pt[1], 1])
            t += DT
        frames_dash = int(2.5 * FPS)
        for step_i in range(frames_dash):
            p = c_pt + (dash_pt - c_pt) * (step_i / frames_dash)
            rows.append([t, p[0], p[1], 2])
            t += DT
        for step_i in range(frames_dash):
            p = dash_pt + (c_pt - dash_pt) * (step_i / frames_dash)
            rows.append([t, p[0], p[1], 2])
            t += DT
    gt = {
        "Total_Combined_Activity_Time_s": 300.0, "Total_Inactivity_Time_s": 0.0,
        "Total_Low_Activity_Time_s": 270.0, "Total_High_Activity_Time_s": 30.0,
        "Total_Distance_px": round(float(expected_dist), 1), "Corner_Time_Ratio_pct": 90.0,
        "Total_Activity_Number": 1, "Long_Active_Number": 1, "Short_Active_Number": 0
    }
    return {"name": "test4_corner_preference", "rows": rows, "gt": gt}


def gen_test_5():
    """Trajectory filter stress test (input for traj_filter.py). Unchanged."""
    rows = []
    t = 0.0
    # 1. Insertion noise (0.0 s - 3.5 s)
    for _ in range(int(3.5 * FPS)):
        rows.append([t, 215.0 + RNG.uniform(-0.5, 0.5), 65.0 + RNG.uniform(-0.5, 0.5)])
        t += DT
    # 2. Continuous movement
    pos = WAYPOINTS["North_Corridor"].copy()
    for i in range(int(315.0 * FPS)):
        pos[0] += 0.8
        if pos[0] > 550.0:
            pos[0] = 250.0
        curr_x, curr_y = pos[0], pos[1]
        if i == 400:                      # ERROR 1: impossible speed spike (>1000 px/s)
            curr_x += 150.0
        elif i == 800:                    # ERROR 2: single-frame spike into the inner island
            curr_x, curr_y = 380.0, 260.0
        elif 1200 <= i <= 1220:           # ERROR 3: dropout; rat reappears in the south corridor
            t += DT
            if i == 1220:
                pos = WAYPOINTS["South_Corridor"].copy()
            continue
        rows.append([t, curr_x, curr_y])
        t += DT
    gt = {"Target_Module": "traj_filter.py", "Expected_Spikes_Removed": 2,
          "Expected_Wall_Jumps_Rejected": 1, "Min_Duration_Post_Crop_s": 300.0}
    return {"name": "test5_traj_filter_stress", "rows": rows, "gt": gt, "filter_test": True}


def gen_test_6(speed):
    """U-turn speed sweep: shuttle along the top corridor at a fixed cruising speed,
    gradual U-turns at both ends, 5 s rest at the end."""
    p_left = np.array([215.0, 65.0])
    p_right = np.array([585.0, 85.0])
    route = Route(p_left, [p_right, p_left], speed=speed, cycle=True)
    rec = Recorder(p_left)
    rec.move(route, TOTAL_PROTOCOL_TIME - 5.0, state=2)
    rec.rest()
    gt = {"Total_Distance_px": round(rows_total_distance(rec.rows), 1), **direction_gt(rec, route)}
    return {"name": f"test6_uturn_sweep_{int(speed)}pxs", "rows": rec.rows, "gt": gt,
            "u_turn_times": route.u_turn_times}


def gen_test_7():
    """Clockwise laps around all four corridors, 90-degree corners, NO reversals.
    Any U-turn found here is a false positive. 5 s rest at the end."""
    W = WAYPOINTS
    loop = [W["North_Corridor"], W["Top_Right_Corner"], W["East_Corridor"], W["Bottom_Right_Corner"],
            W["South_Corridor"], W["Bottom_Left_Corner"], W["West_Corridor"], W["Top_Left_Corner"]]
    route = Route(W["Top_Left_Corner"], loop, speed=60.0, cycle=True)
    rec = Recorder(W["Top_Left_Corner"])
    rec.move(route, TOTAL_PROTOCOL_TIME - 5.0, state=2)
    rec.rest()
    gt = {"Total_Distance_px": round(rows_total_distance(rec.rows), 1), **direction_gt(rec, route)}
    return {"name": "test7_cw_loop_no_reversal", "rows": rec.rows, "gt": gt, "u_turn_times": route.u_turn_times}


def make_noisy(test, seed):
    """Tracker noise + blob wobble -> traj_filter smoothing -> integer rounding. States unchanged.
    Ground truth (true movement) is kept; only direction/U-turn checks are applied."""
    rng = np.random.default_rng(seed)
    rows = test["rows"]
    t = np.array([r[0] for r in rows])
    x = np.array([r[1] for r in rows], float)
    y = np.array([r[2] for r in rows], float)
    f1, f2 = rng.uniform(1.0, 3.0, 2)
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    x = x + WOBBLE_AMP_PX * np.sin(2 * np.pi * f1 * t + p1) + rng.normal(0, NOISE_SIGMA_PX, len(x))
    y = y + WOBBLE_AMP_PX * np.sin(2 * np.pi * f2 * t + p2) + rng.normal(0, NOISE_SIGMA_PX, len(y))
    x = np.round(savgol_filter(median_filter(x, size=MEDIAN_WINDOW), SAVGOL_WINDOW, 3))
    y = np.round(savgol_filter(median_filter(y, size=MEDIAN_WINDOW), SAVGOL_WINDOW, 3))
    new_rows = [[r[0], xi, yi, r[3]] for r, xi, yi in zip(rows, x, y)]
    keys = ["U_Turn_Count", "Clockwise_Distance_px", "CounterClockwise_Distance_px", "Directional_Asymmetry_Index"]
    gt = {k: test["gt"][k] for k in keys if k in test["gt"]}
    return {"name": f"{test['name']}_noise_s{seed}", "rows": new_rows, "gt": gt,
            "u_turn_times": test.get("u_turn_times"), "noisy": True}


# ==============================================================================
# FILE OUTPUT
# ==============================================================================
def save_test_bundle(test):
    name = test["name"]
    with open(os.path.join(OUTPUT_DIR, f"{name}_config.json"), 'w') as f:
        json.dump(CONFIG, f, indent=4)
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        if test.get("filter_test"):
            writer.writerow(['Time', 'X', 'Y'])  # 3 columns for traj_filter.py
            for r in test["rows"]:
                writer.writerow([round(r[0], 4), round(r[1], 2), round(r[2], 2)])
        else:
            writer.writerow(['Time', 'X', 'Y', 'State'])  # 4 columns for feature_extractor.py
            for r in test["rows"]:
                writer.writerow([round(r[0], 4), round(r[1], 2), round(r[2], 2), int(r[3])])
    gt_out = dict(test["gt"])
    if test.get("u_turn_times") is not None:
        gt_out["_U_Turn_Times_s"] = test["u_turn_times"]
    with open(os.path.join(OUTPUT_DIR, f"{name}_ground_truth.json"), 'w') as f:
        json.dump(gt_out, f, indent=4)
    return csv_path


# ==============================================================================
# CW/CCW REPLICA (copy of feature_extractor sections 4 and 6)
# ==============================================================================
def smoothed_states_and_fps(times, raw_states, n_steps):
    """Copy of feature_extractor section 4 (1-second majority voting of states)."""
    total_time = times[-1] - times[0]
    avg_dt = total_time / len(times)
    fps = int(1.0 / avg_dt) if avg_dt > 0 else 25
    states = np.zeros(n_steps, dtype=int)
    for i in range(n_steps // fps + 1):
        start, end = i * fps, min((i + 1) * fps, n_steps)
        if start >= end:
            break
        w = raw_states[start:end]
        c_high, c_low, c_inact = np.sum(w == 2), np.sum(w == 1), np.sum(w == 0)
        if c_high > c_low and c_high > c_inact:
            states[start:end] = 2
        elif (c_low > c_high and c_low > c_inact) or (c_high + c_low > c_inact):
            states[start:end] = 1
        else:
            states[start:end] = 0
    return states, fps


def direction_pipeline(times, xs, ys, raw_states, raw_directions):
    """Copy of the second half of feature_extractor section 6: 1-s majority vote of the per-frame
    directions, High-Active gate, distance sums. raw_directions has one entry per row."""
    distances = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    states, fps = smoothed_states_and_fps(times, raw_states, len(distances))
    raw = np.array(raw_directions, dtype=int).copy()
    if len(raw) > 1:
        raw[-1] = raw[-2]
    smoothed = np.zeros(len(xs), dtype=int)
    for i in range(len(xs) // fps + 1):
        start, end = i * fps, min((i + 1) * fps, len(xs))
        if start >= end:
            break
        w = raw[start:end]
        cw_c, ccw_c = np.sum(w == 1), np.sum(w == -1)
        smoothed[start:end] = 1 if cw_c > ccw_c else (-1 if ccw_c > cw_c else 0)
    for i in range(len(smoothed)):
        if states[min(i, len(states) - 1)] != 2:
            smoothed[i] = 0
    cw = float(np.sum(distances[smoothed[:-1] == 1]))
    ccw = float(np.sum(distances[smoothed[:-1] == -1]))
    return cw, ccw


def raw_directions_by_rule(xs, ys, rule):
    """First half of feature_extractor section 6 with the current or the proposed rule."""
    raw = np.zeros(len(xs), dtype=int)
    if rule == "angle":
        angles = [calculate_angle(ARENA_CENTER, (x, y)) for x, y in zip(xs, ys)]
        for i in range(len(xs) - 1):
            d = normalize_angle_diff(angles[i + 1] - angles[i])
            if d > CURRENT_ANGLE_THRESHOLD_DEG:
                raw[i] = 1
            elif d < -CURRENT_ANGLE_THRESHOLD_DEG:
                raw[i] = -1
    else:
        cx, cy = ARENA_CENTER
        for i in range(len(xs) - 1):
            j = min(i + TANG_WINDOW, len(xs) - 1)
            rx, ry = xs[i] - cx, ys[i] - cy
            r = np.hypot(rx, ry)
            if r < 1e-6:
                continue
            tang = (rx * (ys[j] - ys[i]) - ry * (xs[j] - xs[i])) / r  # > 0 = clockwise on screen
            if tang > MIN_TANG_PX:
                raw[i] = 1
            elif tang < -MIN_TANG_PX:
                raw[i] = -1
    return raw


def replica_cw_ccw(csv_path, rule):
    times, xs, ys, raw_states = [], [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader, None)
        for row in reader:
            times.append(float(row[0]))
            xs.append(float(row[1]))
            ys.append(float(row[2]))
            raw_states.append(int(row[3]))
    times, xs, ys, raw_states = map(np.array, (times, xs, ys, raw_states))
    cw, ccw = direction_pipeline(times, xs, ys, raw_states, raw_directions_by_rule(xs, ys, rule))
    return cw, ccw, asym(cw, ccw)


# ==============================================================================
# RUNNER
# ==============================================================================
def check(key, expected, actual, noisy=False):
    if actual is None:
        return None
    if key in COUNT_KEYS:
        return actual == expected
    return abs(actual - expected) <= tolerance(key, expected, noisy)


def run_filter_test(test, csv_path):
    base = csv_path[:-4]
    out_filtered = f"{base}_filtered.csv"
    metrics = process_trajectory(csv_path=csv_path, config_path=f"{base}_config.json", output_path=out_filtered,
                                 show_plot=False, use_plotly=False, debug=False)
    results = []
    duration = 0.0
    if os.path.exists(out_filtered):
        with open(out_filtered, 'r') as f:
            r = list(csv.reader(f, delimiter=';'))
            if len(r) > 2:
                duration = float(r[-1][0]) - float(r[1][0])
    results.append(("Cropped duration (s)", 300.0, round(duration, 1), abs(duration - 300.0) <= 1.0))
    if isinstance(metrics, dict):
        # Planted spikes are removed either by the excursion step or by the spike rule
        s = metrics.get("Teleportation_Spikes_Removed", 0) + metrics.get("Tracking_Excursions_Removed", 0)
        j = metrics.get("Wall_Jumps_Rejected", 0)
        results.append(("Spikes + excursions removed (>=)", 2, s, s >= 2))
        results.append(("Wall-crossing gaps rejected (>=)", 1, j, j >= 1))
    return results


def run_feature_test(test, csv_path, html):
    base = csv_path[:-4]
    out_json = f"{base}_features.json"
    extract_features(csv_path, f"{base}_config.json", out_json, debug_html=html)
    with open(out_json, 'r') as f:
        feat = json.load(f)
    gen = feat["I_General_Locomotor"]
    p_short = feat["II_Phase_Specific"]["Short_Phases_LEQ_5s"]
    p_long = feat["II_Phase_Specific"]["Long_Phases_GT_5s"]
    spat = feat["III_Spatial_and_Geometric"]
    val_map = {
        "Total_Distance_px": gen.get("Total_Distance_px"),
        "Total_Combined_Activity_Time_s": gen.get("Total_Combined_Activity_Time_s"),
        "Total_Inactivity_Time_s": gen.get("Total_Inactivity_Time_s"),
        "Total_High_Activity_Time_s": gen.get("Total_High_Activity_Time_s"),
        "Total_Low_Activity_Time_s": gen.get("Total_Low_Activity_Time_s"),
        "Total_Activity_Number": gen.get("Total_Activity_Number"),
        "Total_Inactivity_Number": gen.get("Total_Inactivity_Number"),
        "Short_Active_Number": p_short.get("Short_Activity_Number"),
        "Long_Active_Number": p_long.get("Long_Activity_Number"),
        "Short_Inactivity_Number": p_short.get("Short_Inactivity_Number"),
        "Long_Inactivity_Number": p_long.get("Long_Inactivity_Number"),
        "Corner_Time_Ratio_pct": spat.get("Corner_Time_Ratio_pct"),
        "Clockwise_Distance_px": spat.get("Clockwise_Distance_px"),
        "CounterClockwise_Distance_px": spat.get("CounterClockwise_Distance_px"),
        "Directional_Asymmetry_Index": spat.get("Directional_Asymmetry_Index"),
        "U_Turn_Count": spat.get("U_Turn_Count"),
    }
    results = []
    for key, expected in test["gt"].items():
        ok = check(key, expected, val_map.get(key), test.get("noisy", False))
        if ok is not None:
            results.append((key, expected, val_map.get(key), ok))
    return results, val_map


# ==============================================================================
# TEST 8: ACTIVITY FRAME SYNC (review item 2.5)
# ==============================================================================
SYNC_FPS = 25
SYNC_DURATION_S = 20
SYNC_MOVE_S = (10.0, 12.0)   # in-place movement (Low Active)
SYNC_GAP_S = (4.0, 6.0)      # rows removed from the "gap" CSV
SYNC_EDGE_TOL_S = 0.25       # 5-frame SAD delay + debouncing


def gen_test_8(ext):
    return {"name": f"test8_frame_sync{ext.replace('.', '_')}", "activity_test": ext}


def _sync_frames():
    import cv2
    rng = np.random.default_rng(0)
    for f in range(SYNC_DURATION_S * SYNC_FPS):
        img = np.full((576, 768, 3), 40, np.uint8)
        cv2.circle(img, (300, 300), 18, (220, 220, 220), -1)
        if SYNC_MOVE_S[0] * SYNC_FPS <= f < SYNC_MOVE_S[1] * SYNC_FPS:
            for _ in range(6):  # texture changes inside the box, the centre does not move
                x, y = rng.integers(280, 320, 2)
                cv2.circle(img, (int(x), int(y)), 6, (0, 0, 0), -1)
        yield img


def _write_sync_video(path, ext):
    import cv2
    if ext == ".mpg":
        import imageio
        w = imageio.get_writer(path, format="FFMPEG", fps=SYNC_FPS, codec="mpeg2video", quality=10,
                               macro_block_size=16)
        for img in _sync_frames():
            w.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        w.close()
    else:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), SYNC_FPS, (768, 576), True)
        for img in _sync_frames():
            vw.write(img)
        vw.release()


def run_activity_sync_test(test):
    """Classify the same video with a complete CSV and with a CSV that has a 2 s gap.
    Returns (results, skipped_reason)."""
    import contextlib
    import io
    from activity import classify_activities

    ext = test["activity_test"]
    base = os.path.join(OUTPUT_DIR, test["name"])
    try:
        _write_sync_video(f"{base}{ext}", ext)
    except Exception as e:
        return [], f"could not write {ext} video ({e!r})"

    t = np.round(np.arange(SYNC_DURATION_S * SYNC_FPS) / SYNC_FPS, 3)
    states = {}
    for variant in ("nogap", "gap"):
        keep = ~((t >= SYNC_GAP_S[0]) & (t < SYNC_GAP_S[1])) if variant == "gap" else np.ones(len(t), bool)
        csv_path = f"{base}_{variant}.csv"
        with open(csv_path, "w", newline="") as fh:
            fh.write("Time_Sec;X;Y\n")
            for tt in t[keep]:
                fh.write(f"{tt};300;300\n")
        shutil.copy(f"{base}{ext}", f"{base}_{variant}{ext}")  # activity.py finds the video by CSV name
        out = f"{base}_{variant}_classified.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            classify_activities(csv_path, output_path=out, search_dir=OUTPUT_DIR)
        with open(out) as fh:
            rows = list(csv.reader(fh, delimiter=";"))[1:]
        states[variant] = {round(float(r[0]), 3): int(r[3]) for r in rows}

    results = []
    for variant in ("nogap", "gap"):
        low = sorted(tt for tt, st in states[variant].items() if st == 1)
        first, last = (low[0], low[-1]) if low else (None, None)
        results.append((f"Low Active start (s), {variant}", SYNC_MOVE_S[0], first,
                        first is not None and abs(first - SYNC_MOVE_S[0]) <= SYNC_EDGE_TOL_S))
        results.append((f"Low Active end (s), {variant}", SYNC_MOVE_S[1], last,
                        last is not None and abs(last - SYNC_MOVE_S[1]) <= SYNC_EDGE_TOL_S))
    shared = set(states["gap"]) & set(states["nogap"])
    differ = sum(states["gap"][tt] != states["nogap"][tt] for tt in shared)
    results.append(("States differing gap vs no-gap", 0, differ, differ == 0))
    return results, None


# ==============================================================================
# TEST 9: ZONE BACKGROUND (tracker.py, bg_method='zone')
# ==============================================================================
ZB_FPS = 25
ZB_DURATION_S = 100
ZB_W, ZB_H = 768, 576
ZB_DROP_POS = np.array([583.0, 82.0])              # top-right corridor corner (drop)
ZB_SEAT = np.array([572.0, 446.0])                 # bottom-right corridor corner (where the rat sits)
ZB_BL = np.array([202.0, 424.0])
ZB_MOVES = [(24.0, [ZB_DROP_POS, ZB_SEAT]),        # (start s, route); the rat sits at the end of each route
            (55.0, [ZB_SEAT, ZB_BL, ZB_SEAT]),
            (80.0, [ZB_SEAT, ZB_BL, ZB_SEAT])]
ZB_RUN_SPEED = 200.0
ZB_DROP_S = 18.0
ZB_TRACK_FROM_S = 30.0                             # tracking starts later, like --backtrack (rat already sitting)
ZB_MIN_FOUND_PCT = 99.0


def gen_test_9():
    return {"name": "test9_zone_background", "tracker_test": True}


def _zb_rat(t):
    """(centre, angle, moving) of the rat at time t, or None before the drop."""
    if t < ZB_DROP_S:
        return None
    seat = ZB_DROP_POS
    for start, route in ZB_MOVES:
        if t < start:
            break
        seg = [float(np.linalg.norm(b - a)) for a, b in zip(route, route[1:])]
        if t < start + sum(seg) / ZB_RUN_SPEED:
            d = (t - start) * ZB_RUN_SPEED
            for a, b, L in zip(route, route[1:], seg):
                if d <= L:
                    return a + (b - a) * d / L, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), True
                d -= L
        seat = route[-1]
    # sitting almost still (small breathing / head movement)
    return seat + np.array([0.5 * math.sin(t * 2.1), 0.5 * math.cos(t * 1.7)]), 80.0 + 3.0 * math.sin(t * 0.8), False


def _zb_write_video(path):
    import cv2
    rng = np.random.default_rng(1)
    base = np.full((ZB_H, ZB_W), 30, np.float32)
    cv2.fillPoly(base, [np.array(CONFIG["outer_poly"], np.int32)], 95)
    cv2.fillPoly(base, [np.array(CONFIG["inner_poly"], np.int32)], 150)
    base += cv2.GaussianBlur(rng.normal(0, 12, (ZB_H, ZB_W)).astype(np.float32), (0, 0), 2)  # floor texture
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), ZB_FPS, (ZB_W, ZB_H), True)
    truth = {}
    for f in range(ZB_DURATION_S * ZB_FPS):
        t = f / ZB_FPS
        img = base.copy()
        if t >= 12.0:                                           # food stick, stays in the corner
            cv2.line(img, (596, 56), (590, 100), 175, 3)
        if 8.0 <= t < 12.0 or 16.0 <= t < 19.0:                 # experimenter's hand
            k = (t - 8.0) / 4.0 if t < 12.0 else (t - 16.0) / 3.0
            c = (int(660 - 60 * math.sin(math.pi * k)), int(20 + 50 * math.sin(math.pi * k)))
            cv2.ellipse(img, c, (55, 30), 30, 0, 360, 185, -1)
        st = _zb_rat(t)
        if st is not None:
            (x, y), ang, _ = st
            cv2.ellipse(img, (int(round(x)), int(round(y))), (22, 11), ang, 0, 360, 210, -1)
            if t >= ZB_TRACK_FROM_S:
                truth[f] = (x, y)
        img += rng.normal(0, 3, img.shape)
        vw.write(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    vw.release()
    return truth


def _zb_found_pct(traj_csv, truth):
    """% of frames after the drop where the tracked position is within 10 px of the rat
    (best of the matching frame and its neighbours, to be independent of the time-stamp convention)."""
    got = {}
    if os.path.exists(traj_csv):
        with open(traj_csv) as fh:
            for r in list(csv.reader(fh, delimiter=";"))[1:]:
                got[int(round(float(r[0]) * ZB_FPS))] = (float(r[1]), float(r[2]))
    found = 0
    for f, (x, y) in truth.items():
        errs = [math.hypot(got[g][0] - x, got[g][1] - y) for g in (f - 1, f, f + 1) if g in got]
        found += int(bool(errs) and min(errs) <= 10.0)
    return round(100.0 * found / len(truth), 1)


def run_zone_background_test(test):
    import contextlib
    import io
    from tracker import track

    base = os.path.join(OUTPUT_DIR, test["name"])
    video = f"{base}.mp4"
    config_path = f"{base}_config.json"
    with open(config_path, "w") as fh:
        json.dump(CONFIG, fh)
    truth = _zb_write_video(video)
    stem = os.path.basename(base)
    pct, safe = {}, None
    for bg in ("zone", "global"):
        out_dir = f"{base}_{bg}"
        with contextlib.redirect_stdout(io.StringIO()):
            track(video, config_file=config_path, method="static", bg_method=bg, output_dir=out_dir,
                  skip_seconds=ZB_TRACK_FROM_S)
        pct[bg] = _zb_found_pct(os.path.join(out_dir, f"{stem}_static_trajectory.csv"), truth)
        stats = os.path.join(out_dir, f"{stem}_static_statistics.txt")
        if bg == "zone" and os.path.exists(stats):
            for line in open(stats):
                if line.startswith("Zone background:") and "top-right quarter" in line:
                    safe = int(line.split(" safe samples")[0].split("(")[-1])
    sitting = [not _zb_rat(f / ZB_FPS)[2] for f in range(int(ZB_TRACK_FROM_S * ZB_FPS), ZB_DURATION_S * ZB_FPS)]
    in_corner = 100.0 * sum(sitting) / len(sitting)
    results = [
        ("Top-right zone safe samples (>=5)", 5, safe, safe is not None and safe >= 5),
        ("Rat found <=10 px, zone (%)", ZB_MIN_FOUND_PCT, pct["zone"], pct["zone"] >= ZB_MIN_FOUND_PCT),
    ]
    info = (f"tracking from {ZB_TRACK_FROM_S:.0f} s; the rat sits still in the bottom-right corner {in_corner:.0f}% "
            f"of that time; for comparison, "
            f"bg_method='global' found it in {pct['global']}% of frames")
    return results, info


def print_results(name, results, u_turn_times=None):
    print(f"\n[{name}]")
    print(f"  {'Feature':<34} | {'Expected':>10} | {'Got':>10} | Status")
    print("  " + "-" * 70)
    for key, exp, got, ok in results:
        print(f"  {key:<34} | {str(exp):>10} | {str(got):>10} | {'PASS' if ok else 'FAIL'}")
        if key == "U_Turn_Count" and not ok and u_turn_times:
            print(f"  {'':<34}   ground-truth U-turn times (s): {u_turn_times}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3, help="Number of noise seeds per test (default 3)")
    parser.add_argument("--no-noise", action="store_true", help="Skip the noise variants")
    parser.add_argument("--only", default=None, help="Run only tests whose name contains this text")
    parser.add_argument("--html", action="store_true", help="Write the extractor's debug HTML dashboards")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating synthetic validation test suite...")

    clean = [gen_test_1(), gen_test_2(), gen_test_3(), gen_test_4(), gen_test_5()]
    # U-turns are defined during clear movement (>= 2 px/frame = 50 px/s, feature_extractor
    # UTURN_MIN_SPEED_PX_FRAME), so the sweep starts above that speed.
    clean += [gen_test_6(v) for v in (60.0, 100.0, 150.0)]
    clean.append(gen_test_7())
    tests = list(clean)
    sync_tests = [gen_test_8(".mp4"), gen_test_8(".mpg"), gen_test_9()]
    if not args.no_noise:
        noise_sources = [t for t in clean if t["name"].startswith(("test2", "test3", "test6", "test7"))]
        for t in noise_sources:
            for s in range(args.seeds):
                tests.append(make_noisy(t, seed=1000 + s))
    tests += sync_tests
    if args.only:
        tests = [t for t in tests if args.only in t["name"]]

    for t in tests:
        if not t.get("activity_test") and not t.get("tracker_test"):
            t["csv_path"] = save_test_bundle(t)
    print(f"{len(tests)} test files written to {os.path.abspath(OUTPUT_DIR)}")

    print("\n" + "=" * 90)
    print("      PIPELINE EXECUTION & GROUND TRUTH VERIFICATION")
    print("=" * 90)

    failures = []
    total_checks = 0
    direction_rows = []

    for t in tests:
        if t.get("tracker_test"):
            results, info = run_zone_background_test(t)
            print_results(t["name"], results)
            print(f"  ({info})")
            total_checks += len(results)
            failures += [(t["name"], r[0], r[1], r[2]) for r in results if not r[3]]
            continue
        if t.get("activity_test"):
            results, skipped = run_activity_sync_test(t)
            if skipped:
                print(f"\n[{t['name']}]\n  SKIPPED: {skipped}")
                continue
            extracted = None
        elif t.get("filter_test"):
            results = run_filter_test(t, t["csv_path"])
            extracted = None
        else:
            results, extracted = run_feature_test(t, t["csv_path"], args.html)
        print_results(t["name"], results, t.get("u_turn_times"))
        total_checks += len(results)
        failures += [(t["name"], r[0], r[1], r[2]) for r in results if not r[3]]

        if extracted is not None and "Clockwise_Distance_px" in t["gt"]:
            current = replica_cw_ccw(t["csv_path"], "angle")
            proposed = replica_cw_ccw(t["csv_path"], "tangential")
            direction_rows.append((t["name"], t["gt"], extracted, current, proposed, t.get("noisy", False)))

    # --- CW/CCW rule comparison ---
    if direction_rows:
        print("\n" + "=" * 90)
        print("      CW/CCW RULE COMPARISON (review item 2.3) - % of ground truth, asymmetry index")
        print("=" * 90)
        print(f"  {'Test':<38} | {'Rule':<22} | {'CW %':>6} | {'CCW %':>6} | {'Asym':>6} (truth)")
        print("  " + "-" * 100)
        rule_ok = {"extractor": 0, "current": 0, "proposed": 0}
        extractor_matches = {"current": 0, "proposed": 0, "neither": 0}
        for name, gt, ext, cur, prop, noisy in direction_rows:
            t_cw, t_ccw, t_as = gt["Clockwise_Distance_px"], gt["CounterClockwise_Distance_px"], gt["Directional_Asymmetry_Index"]
            entries = [("extractor", (ext["Clockwise_Distance_px"], ext["CounterClockwise_Distance_px"],
                                      ext["Directional_Asymmetry_Index"])),
                       ("current", cur), ("proposed", prop)]
            for label, (cw, ccw, a) in entries:
                ok = (check("Clockwise_Distance_px", t_cw, cw, noisy)
                      and check("CounterClockwise_Distance_px", t_ccw, ccw, noisy)
                      and check("Directional_Asymmetry_Index", t_as, a, noisy))
                rule_ok[label] += int(bool(ok))
                pct = lambda v, ref: f"{100.0 * v / ref:6.1f}" if ref > 0 else (f"{v:6.1f}" if v else "   0.0")
                rule_name = {"extractor": "your feature_extractor", "current": "current (0.5 deg)",
                             "proposed": "tangential (2 px/5 fr)"}[label]
                print(f"  {name if label == 'extractor' else '':<38} | {rule_name:<22} | {pct(cw, t_cw)} | "
                      f"{pct(ccw, t_ccw)} | {a:6.3f} ({t_as:.3f}) {'PASS' if ok else 'FAIL'}")
            same = lambda r: (abs(ext["Clockwise_Distance_px"] - round(r[0], 2)) < 0.05
                              and abs(ext["CounterClockwise_Distance_px"] - round(r[1], 2)) < 0.05)
            extractor_matches["current"] += int(same(cur))
            extractor_matches["proposed"] += int(same(prop))
            extractor_matches["neither"] += int(not same(cur) and not same(prop))
        n = len(direction_rows)
        print("  " + "-" * 100)
        print(f"  Within tolerance: your extractor {rule_ok['extractor']}/{n}, current rule {rule_ok['current']}/{n}, "
              f"tangential rule {rule_ok['proposed']}/{n}")
        if extractor_matches["proposed"] == n and extractor_matches["current"] < n:
            print("  Your feature_extractor.py gives the same CW/CCW as the tangential rule (fix applied).")
        elif extractor_matches["current"] == n and extractor_matches["proposed"] < n:
            print("  Your feature_extractor.py gives the same CW/CCW as the current (0.5 deg) rule.")
        else:
            print(f"  Your feature_extractor.py matches the current rule in {extractor_matches['current']}, "
                  f"the tangential rule in {extractor_matches['proposed']} and neither in "
                  f"{extractor_matches['neither']} of {n} tests (on some paths both rules give the same result).")

    print("\n" + "=" * 90)
    print(f"  {total_checks - len(failures)} of {total_checks} feature checks passed.")
    if failures:
        print("  Failed checks:")
        for name, key, exp, got in failures:
            print(f"    - {name}: {key} expected {exp}, got {got}")
    else:
        print("  All feature checks passed.")
    print("=" * 90 + "\n")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())