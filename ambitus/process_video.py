import cv2
import numpy as np
import json
import os
import csv

# --- CONSTANTS ---
MAX_JUMP_DIST = 100 

class SimpleRatTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "Simple (MOG2)"
        self.missed_frames = 0
        self.collision_detected = False
        
        self.is_initialized = False
        self.last_valid_pos = None 
        
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly

    def update(self, measured_coords, frame_gray=None):
        if measured_coords is not None:
            self.missed_frames = 0
            
            self.is_initialized = True
            self.last_valid_pos = measured_coords
            
            return measured_coords
        else:
            self.missed_frames += 1
            return None

    def get_state_debug(self):
        return 0.0, 0.0, 0.0


class DenseRatTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "Dense Optical Flow" # Print this in debug
        self.missed_frames = 0
        self.collision_detected = False
        
        self.is_initialized = False
        self.last_valid_pos = None 
        
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly

    def update(self, measured_coords, frame_gray=None):
        # This is also just "Pass-through", does not predict like Simple
        if measured_coords is not None:
            self.missed_frames = 0
            self.is_initialized = True
            self.last_valid_pos = measured_coords
            return measured_coords
        else:
            self.missed_frames += 1
            return None

    def get_state_debug(self):
        return 0.0, 0.0, 0.0

class KalmanRatTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "Kalman Filter"
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

        self.last_valid_pos = None
        self.is_initialized = False
        self.missed_frames = 0
        self.collision_detected = False 
        
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly

    def check_geometric_constraints(self, x, y):
        if cv2.pointPolygonTest(self.inner_poly, (float(x), float(y)), False) >= 0: return False 
        if cv2.pointPolygonTest(self.outer_poly, (float(x), float(y)), False) < 0: return False 
        return True

    def update(self, measured_coords, frame_gray=None): # frame_gray needed for compatibility
        pred = self.kalman.predict()
        pred_x, pred_y = int(pred[0, 0]), int(pred[1, 0])
        self.collision_detected = False

        if not self.is_initialized:
            if measured_coords is not None:
                self.kalman.statePre = np.array([[measured_coords[0]], [measured_coords[1]], [0], [0]], np.float32)
                self.kalman.statePost = np.array([[measured_coords[0]], [measured_coords[1]], [0], [0]], np.float32)
                self.last_valid_pos = measured_coords
                self.is_initialized = True
                return measured_coords
            return None

        if measured_coords is not None:
            measurement = np.array([[np.float32(measured_coords[0])], [np.float32(measured_coords[1])]])
            self.kalman.correct(measurement)
            self.last_valid_pos = measured_coords
            self.missed_frames = 0
            return measured_coords
        else:
            self.missed_frames += 1
            if not self.check_geometric_constraints(pred_x, pred_y):
                self.collision_detected = True
                if self.last_valid_pos:
                    self.kalman.statePost[0, 0] = self.last_valid_pos[0]
                    self.kalman.statePost[1, 0] = self.last_valid_pos[1]
                    self.kalman.statePost[2, 0] = 0.0 
                    self.kalman.statePost[3, 0] = 0.0 
                    return self.last_valid_pos
            
            decay_factor = 0.9 
            self.kalman.statePost[2, 0] *= decay_factor
            self.kalman.statePost[3, 0] *= decay_factor

            if self.missed_frames > 50: return None
            return (pred_x, pred_y)

    def get_state_debug(self):
        if not self.is_initialized: return 0.0, 0.0, 0.0
        vx, vy = self.kalman.statePost[2, 0], self.kalman.statePost[3, 0]
        return np.sqrt(vx**2 + vy**2), vx, vy


class OpticalFlowRatTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "Optical Flow"
        self.last_valid_pos = None
        self.is_initialized = False
        self.missed_frames = 0
        self.collision_detected = False
        
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly
        
        # Optical Flow variables
        self.prev_gray = None
        self.p0 = None # Points to track
        
        # Lucas-Kanade parameters
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Feature search parameters (GoodFeaturesToTrack)
        self.feature_params = dict(maxCorners=20, qualityLevel=0.3, minDistance=7, blockSize=7)

    def check_geometric_constraints(self, x, y):
        if cv2.pointPolygonTest(self.inner_poly, (float(x), float(y)), False) >= 0: return False 
        if cv2.pointPolygonTest(self.outer_poly, (float(x), float(y)), False) < 0: return False 
        return True

    def update(self, measured_coords, frame_gray):
        """
        If measured_coords (MOG2): Reinitialize the points on the rat.
        If not: Track old points with Optical Flow.
        """
        self.collision_detected = False
        current_pos = None

        # CASE 1: Strong detection (MOG2) -> This is the "Ground Truth"
        if measured_coords is not None:
            self.last_valid_pos = measured_coords
            self.missed_frames = 0
            self.is_initialized = True
            current_pos = measured_coords
            
            # Search for new tracking points around the rat so we can prepare for the next (possibly incomplete) frame
            # Create a small mask around the rat
            mask_roi = np.zeros_like(frame_gray)
            cx, cy = measured_coords
            # About 40x40 box around the rat for feature search
            cv2.circle(mask_roi, (cx, cy), 20, 255, -1) 
            
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask_roi, **self.feature_params)
            self.prev_gray = frame_gray.copy()
            
            return current_pos

        # CASE 2: No detection -> Optical Flow
        if self.is_initialized and self.p0 is not None and len(self.p0) > 0:
            self.missed_frames += 1
            
            # Calculate displacement
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.p0, None, **self.lk_params)
            
            # Select good points (which were successfully tracked)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]
                
                if len(good_new) > 0:
                    # Calculate average displacement (dx, dy)
                    displacement = good_new - good_old
                    dx = np.median(displacement[:, 0])
                    dy = np.median(displacement[:, 1])
                    
                    # If very few points or jumping, assume zero motion (stop)
                    if abs(dx) < 0.5: dx = 0
                    if abs(dy) < 0.5: dy = 0
                    
                    # Update position
                    old_x, old_y = self.last_valid_pos
                    new_x = int(old_x + dx)
                    new_y = int(old_y + dy)
                    
                    # Geometric constraint check
                    if self.check_geometric_constraints(new_x, new_y):
                        self.last_valid_pos = (new_x, new_y)
                        current_pos = (new_x, new_y)
                        
                        # Update points for next iteration
                        self.p0 = good_new.reshape(-1, 1, 2)
                        self.prev_gray = frame_gray.copy()
                    else:
                        self.collision_detected = True
                        # Hit wall -> Keep old position
                        current_pos = self.last_valid_pos
                else:
                    # Lost the points
                    current_pos = self.last_valid_pos # Best we can do is stay in place
            else:
                current_pos = self.last_valid_pos

            if self.missed_frames > 50: return None
            return current_pos

        return None # Not initialized yet

    def get_state_debug(self):
        # OF does not store velocity directly, but we can return 0 or dummy data
        return 0.0, 0.0, 0.0

class PureSparseOFTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "Pure Sparse OF"
        self.missed_frames = 0
        self.collision_detected = False
        self.is_initialized = False
        self.last_valid_pos = None 
        
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly
        
        self.prev_gray = None
        self.p0 = None # Tracked points
        
        # Lucas-Kanade parameters
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Feature search (only during initialization)
        self.feature_params = dict(maxCorners=50, qualityLevel=0.1, minDistance=5, blockSize=5)

    def check_geometric_constraints(self, x, y):
        if cv2.pointPolygonTest(self.inner_poly, (float(x), float(y)), False) >= 0: return False 
        if cv2.pointPolygonTest(self.outer_poly, (float(x), float(y)), False) < 0: return False 
        return True

    def update(self, measured_coords, frame_gray):
        """
        measured_coords: Position detected by MOG2.
        LOGIC:
        - If we have enough trackable points (self.p0), we ignore measured_coords and only use OF.
        - If we have no points (lost them), we reinitialize based on measured_coords.
        """
        self.collision_detected = False
        
        # 1. Do we have enough trackable points from the previous frame?
        if self.is_initialized and self.p0 is not None and len(self.p0) > 5:
            # OPTICAL FLOW TRACKING
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.p0, None, **self.lk_params)
            
            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                
                if len(good_new) > 5:
                    # Rat's new position is the centroid of the points (average)
                    cx = int(np.mean(good_new[:, 0]))
                    cy = int(np.mean(good_new[:, 1]))
                    
                    # Geometric constraint check
                    if self.check_geometric_constraints(cx, cy):
                        self.last_valid_pos = (cx, cy)
                        self.missed_frames = 0
                        
                        # Update points
                        self.p0 = good_new.reshape(-1, 1, 2)
                        self.prev_gray = frame_gray.copy()
                        return (cx, cy)
                    else:
                        # Hit wall with the points -> Reset
                        self.collision_detected = True
                        # Let the branch below (re-init) run next time, for now return the old position
                        self.p0 = None # Forced reset
                        return self.last_valid_pos
            
            # If we reach here, we lost the points
            self.p0 = None 

        # 2. INITIALIZATION / RESCUE
        # Only process MOG2 if OF failed
        if measured_coords is not None:
            self.missed_frames = 0
            self.is_initialized = True
            self.last_valid_pos = measured_coords
            
            # Search for new points on the rat (around MOG2 position)
            mask_roi = np.zeros_like(frame_gray)
            cx, cy = measured_coords
            cv2.circle(mask_roi, (cx, cy), 25, 255, -1) # Slightly larger radius for search
            
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask_roi, **self.feature_params)
            self.prev_gray = frame_gray.copy()
            
            return measured_coords

        # 3. Complete darkness (neither OF nor MOG2)
        self.missed_frames += 1
        return None

    def get_state_debug(self):
        # Optional: Could return the number of tracked points instead of speed
        num_points = len(self.p0) if self.p0 is not None else 0
        return float(num_points), 0.0, 0.0 # Return point count instead of speed

class FusionRatTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "MOG2 + DOF Fusion"
        self.missed_frames = 0
        self.collision_detected = False
        
        self.is_initialized = False
        self.last_valid_pos = None 
        
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly

    def update(self, measured_coords, frame_gray=None):
        if measured_coords is not None:
            self.missed_frames = 0
            self.is_initialized = True
            self.last_valid_pos = measured_coords
            return measured_coords
        else:
            self.missed_frames += 1
            return None

    def get_state_debug(self):
        return 0.0, 0.0, 0.0
    

class MotionHistoryTracker:
    def __init__(self, inner_poly, outer_poly):
        self.name = "Motion History (MHI)"
        self.missed_frames = 0
        self.collision_detected = False
        self.is_initialized = False
        self.last_valid_pos = None 
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly
        
        self.mhi = None
        self.prev_gray = None
        self.timestamp = 0
        self.duration = None # Duration for MHI in seconds

    def update(self, measured_coords, frame_gray=None):
        if measured_coords is not None:
            self.missed_frames = 0
            self.is_initialized = True
            self.last_valid_pos = measured_coords
            return measured_coords
        else:
            self.missed_frames += 1
            return None

    def get_state_debug(self):
        return 0.0, 0.0, 0.0

def load_config(config_file):
    if not os.path.exists(config_file):
        print(f"HIBA: Nem található a konfigurációs fájl: {config_file}")
        return None
    with open(config_file, 'r') as f:
        return json.load(f)

def create_poly_mask(config, frame_shape):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    outer_pts = np.array(config["outer_poly"], dtype=np.int32)
    inner_pts = np.array(config["inner_poly"], dtype=np.int32)
    cv2.fillPoly(mask, [outer_pts], 255)
    cv2.fillPoly(mask, [inner_pts], 0)
    return mask

def save_to_csv(data, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Time_Sec", "X", "Y"])
            writer.writerows(data)
        print(f"\nSuccessful save: {filename} ({len(data)} points)")
    except Exception as e:
        print(f"Error during save: {e}")

def get_best_contour(contours, tracker, inner_pts_array, min_area):
    valid_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if cv2.pointPolygonTest(inner_pts_array, (cx, cy), False) >= 0: continue
        valid_candidates.append((c, (cx, cy), area))

    if not valid_candidates: return None, None

    if not tracker.is_initialized or tracker.last_valid_pos is None:
        best = max(valid_candidates, key=lambda x: x[2])
        return best[0], best[1]

    last_x, last_y = tracker.last_valid_pos
    best_candidate = None
    min_dist = float('inf')

    for c, (cx, cy), area in valid_candidates:
        dist = np.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_candidate = (c, (cx, cy))

    if min_dist > MAX_JUMP_DIST: return None, None
    return best_candidate[0], best_candidate[1]

def clean_trajectory(data, max_drift_dist=50):
    """
    Filters out Kalman filter "overshoots".
    data format: [(time, x, y, is_predicted), ...]
    
    Logic: If there is a predicted segment (all True), we check where it started
    and where the real measurement returned. If the rat actually didn't move much
    (start and end are close), but the prediction drifted, then delete the prediction.
    """
    if not data: return []
    
    cleaned_data = []
    n = len(data)
    i = 0
    
    while i < n:
        # If this is a measured point (not prediction), keep it
        if not data[i][3]: # is_predicted == False
            cleaned_data.append(data[i])
            i += 1
            continue
            
        # If we find a prediction, find the end of the segment
        j = i
        while j < n and data[j][3]: # While prediction
            j += 1
            
        # Now the segment runs from i to (j-1).
        # The j-th element (if it exists) is the first new REAL measurement.
        # The (i-1)-th element is the last REAL measurement.
        
        start_valid = data[i-1] if i > 0 else None
        end_valid = data[j] if j < n else None
        
        keep_segment = True
        
        # Check the "drift"
        if start_valid and end_valid:
            print("Start Valid:", start_valid, "End Valid:", end_valid)
            sx, sy = start_valid[1], start_valid[2]
            ex, ey = end_valid[1], end_valid[2]
            
            # Actual displacement in reality
            real_displacement = np.sqrt((ex-sx)**2 + (ey-sy)**2)
            
            # If the rat barely moved in reality, but the tracker predicted much,
            # then this is just "drift during a stop". Discard it.
            if real_displacement < max_drift_dist:
                keep_segment = False
        
        # If there's no end to the segment (end of video), or no beginning,
        # then we can decide to discard the uncertain end.
        elif start_valid and not end_valid:
             print("No end valid point; discarding trailing drift segment.")
             keep_segment = False # Drifted into wall at end of video -> trash
             
        if keep_segment:
            print("Keeping drift segment from", i, "to", j-1)
            # Keep the points (optionally we could smooth, but just copy for now)
            for k in range(i, j):
                cleaned_data.append(data[k])
        else:
            print("Discarding drift segment from", i, "to", j-1)
            # Don't keep (delete the drift). 
            # Optional: we could interpolate between start and end,
            # but for "stopping" case, deletion is correct because it won't draw a line.
            pass
            
        i = j # Continue after the segment

    return cleaned_data

# --- DEBUG VIEW BUILDER ---
def create_debug_view(frame, fgmask, tracker, measured_coords, final_pos, extra_debug_mask=None):
    """
    Assembles the combined image:
    [ Original Video ] [ Mask ]
                      [ Info Panel ]
    """
    h, w = frame.shape[:2]
    panel_width = 400 
    target_h = int(h / 2)
    
    if tracker.name == "MOG2 + DOF Fusion" and extra_debug_mask is not None:
        # FUSION VIEW: Colored composite
        # fgmask = MOG2 (should be Green)
        # extra_debug_mask = DOF (should be Red)
        
        composite = np.zeros((h, w, 3), dtype=np.uint8)
        composite[:, :, 1] = fgmask           # Green channel (MOG2)
        composite[:, :, 2] = extra_debug_mask # Red channel (DOF)
        
        # Resize for the panel
        mask_resized = cv2.resize(composite, (panel_width, target_h))
        label = "Fusion: Green=MOG2, Red=DOF"
    else:
        # Traditional view (Grayscale)
        mask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        mask_resized = cv2.resize(mask_bgr, (panel_width, target_h))
        label = tracker.name
    
    cv2.rectangle(mask_resized, (0,0), (panel_width-1, target_h-1), (100,100,100), 2)    
    cv2.putText(mask_resized, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    info_panel = np.zeros((h - target_h, panel_width, 3), dtype=np.uint8)
    
    val1, val2, val3 = tracker.get_state_debug()
    missed = tracker.missed_frames
    
    line_h = 30
    y_off = 30
    
    status_color = (0, 255, 0)
    status_text = "TRACKING"
    
    if missed > 0:
        status_color = (0, 255, 255)
        status_text = "COASTING"
    if missed > 40:
        status_color = (0, 0, 255)
        status_text = "LOST"

    # For Simple and Dense trackers there is no prediction, so missed > 0 means NO DETECTION
    if (tracker.name.startswith("Simple") or tracker.name.startswith("Dense")) and missed > 0:
        status_text = "NO DETECTION"
        status_color = (0, 0, 255)
        
    if tracker.collision_detected:
        status_text = "WALL COLLISION"
        status_color = (0, 0, 255)

    cv2.putText(info_panel, f"METHOD: {tracker.name}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y_off += line_h
    cv2.putText(info_panel, f"STATUS: {status_text}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    y_off += line_h
    
    meas_str = f"Meas: {measured_coords}" if measured_coords else "Meas: None"
    cv2.putText(info_panel, meas_str, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1)
    y_off += line_h
    
    track_str = f"{tracker.name}: {final_pos}" if final_pos else f"{tracker.name}: None"
    cv2.putText(info_panel, track_str, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    y_off += line_h * 2
    
    if tracker.name == "Pure Sparse OF":
        cv2.putText(info_panel, f"Tracked Pts: {int(val1)}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    else:
        cv2.putText(info_panel, f"Speed: {val1:.2f} px/fr", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

    y_off += line_h
    cv2.putText(info_panel, f"Vel X: {val2:.2f}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    y_off += line_h
    cv2.putText(info_panel, f"Vel Y: {val3:.2f}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    y_off += line_h * 2
    
    cv2.putText(info_panel, f"Missed Frames: {missed}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)

    right_sidebar = np.vstack([mask_resized, info_panel])
    if right_sidebar.shape[0] != h:
        right_sidebar = cv2.resize(right_sidebar, (panel_width, h))
        
    combined = np.hstack([frame, right_sidebar])
    return combined

def process_video(video_path, config_file="arena_config_poly.json", method='simple', show_traj=False, debug=False, skip_seconds=0.0, min_area=200):
    """
    Process a video file to track rat movement within defined polygonal regions.
    This function applies various tracking methods to detect and follow a rat's trajectory
    throughout a video. It supports multiple detection and tracking algorithms, including
    background subtraction (MOG2), optical flow, Kalman filtering, and dense optical flow.
    The function can process videos with configurable parameters for detection sensitivity,
    region of interest, and visualization options.
    Args:
        video_path (str): Path to the input video file to process.
        config_file (str, optional): Path to the JSON configuration file containing polygon
            coordinates for inner and outer regions. Defaults to "arena_config_poly.json".
        method (str, optional): Tracking algorithm to use. Options are:
            - 'simple': MOG2 background subtraction only
            - 'of': Optical flow-based tracking
            - 'kf': Kalman filter tracking
            - 'dof': Dense optical flow tracking
            - 'sub_dof_fusion': Fusion of MOG2 and Dense Optical Flow (advanced)
            - 'mhi': Motion History Image-based tracking (experimental)
            Defaults to 'simple'.
        show_traj (bool, optional): If True, display the video with trajectory visualization
            in real-time. Defaults to False.
        debug (bool, optional): If True, enable debug visualizations and messages during processing.
            Defaults to False.
        skip_seconds (float, optional): Number of seconds to skip from the start of the video
            before beginning processing. Defaults to 0.0.
        min_area (int, optional): Minimum contour area (in pixels) to consider as a valid detection.
            Smaller detections are ignored. Defaults to 200.
    Returns:
        None. Outputs are saved to files:
            - CSV file: "{video_filename}_{method}_trajectory.csv" containing (timestamp, x, y)
            - Image file: "{video_filename}_{method}_trajectory_map.jpg" showing the trajectory map
    """
    config = load_config(config_file)
    if config is None: return

    inner_pts_array = np.array(config["inner_poly"], dtype=np.int32)
    outer_pts_array = np.array(config["outer_poly"], dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(skip_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, first_frame = cap.read()
    if not ret: return

    final_background_image = first_frame.copy()
    roi_mask = create_poly_mask(config, first_frame.shape)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # Background subtractor (only needed if NOT in 'dof' mode)
    fgbg = None
    if method != 'dof':
        fgbg = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=30, detectShadows=False)
    
    # --- TRACKER SELECTION ---
    if method == 'mhi':
        print("Using Motion History Image Tracker")
        tracker = MotionHistoryTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)
        tracker.duration = 20.0 # Duration for MHI
    elif method == 'sub_dof_fusion':
        print("Using Fusion Tracker (MOG2 + DOF)")
        tracker = FusionRatTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)
    elif method == 'pure_of':
        print("Using PURE Sparse Optical Flow Tracker")
        tracker = PureSparseOFTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)
    elif method == 'of':
        print("Using Optical Flow Tracker")
        tracker = OpticalFlowRatTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)
    elif method == 'simple':
        print("Using Simple Tracker (MOG2 Only)")
        tracker = SimpleRatTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)
    elif method == 'kf':
        print("Using Kalman Filter Tracker")
        tracker = KalmanRatTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)
    elif method == 'dof':
        print("Using Dense Optical Flow Detector")
        tracker = DenseRatTracker(inner_poly=inner_pts_array, outer_poly=outer_pts_array)       
    # ----------------------------

    trajectory_data = []
    frame_counter = start_frame
    visualization_active = show_traj or debug

    # Dense Optical Flow needs previous frame in grayscale
    prev_gray_dof = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_gray_dof = clahe.apply(prev_gray_dof)
    prev_gray_dof = cv2.GaussianBlur(prev_gray_dof, (7, 7), 0)

    print(f"Starting processing from {skip_seconds} seconds...")
    if not visualization_active:
        print("Fast mode active...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_counter += 1
        if not visualization_active and frame_counter % 100 == 0:
            print(f"\rProcessed: {frame_counter}/{total_frames} frames", end="")

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = current_time_ms / 1000.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0) # For MOG2

        # --- DETECTION LOGIC SELECTION ---
        fgmask = None
        extra_mask_for_debug = None
        if method == 'mhi':
            if 'mhi' not in locals():
                mhi = np.zeros_like(gray, dtype=np.float32)
                prev_gray_mhi = gray.copy()
            
            # 2. Frame Difference Calculation
            diff = cv2.absdiff(gray, prev_gray_mhi)
            
            # 3. Binary mask (Threshold) - Where there is movement, it is 1
            _, motion_mask = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
            
            # 4. MHI Update (Manual)
            # - We decrease the value everywhere a little (Fade out / Decay)
            #   E.g. -1 every frame (if we are on a 0-255 scale)
            #   Here we use float32 for the timestamp
            
            timestamp = float(frame_counter)
            duration = tracker.duration
            
            # Where there is movement (motion_mask == 1), we write the current timestamp
            mhi[motion_mask > 0] = timestamp
            
            # Where the movement is too old (< timestamp - duration), we reset to zero
            # (This step is optional if only visualization is needed, but helps in contour detection)
            mhi[mhi < (timestamp - duration)] = 0
            
            # 5. Conversion to 0-255 uint8 mask for contour detection
            # Normalize: (val - (now - duration)) / duration -> 0..1 -> 0..255
            # Where the movement is fresh, it is bright (255). Where it is old, it darkens.
            
            vis_mhi = (mhi - (timestamp - duration)) / duration
            vis_mhi = np.clip(vis_mhi, 0, 1) # Just to be safe
            fgmask = np.uint8(vis_mhi * 255)
            # Update
            prev_gray_mhi = gray.copy()
        elif method == 'sub_dof_fusion':
            # 1. MOG2
            mask_mog2 = fgbg.apply(gray_blurred, learningRate=0.0005)
            
            # 2. DOF
            flow = cv2.calcOpticalFlowFarneback(prev_gray_dof, gray_blurred, None, 
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            _, mask_dof = cv2.threshold(mag, 2.0, 255, cv2.THRESH_BINARY)
            mask_dof = mask_dof.astype(np.uint8)
            
            # 3. FÚZIÓ (VAGY kapcsolat)
            fgmask = cv2.bitwise_or(mask_mog2, mask_dof)
            
            # Frissítjük a köv. körhöz
            prev_gray_dof = gray_blurred.copy()
            extra_mask_for_debug = mask_dof
        elif method == 'dof':
            # DENSE OPTICAL FLOW DETECTOR
            # 1. Flow calculation (Farneback)
            # cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            # winsize=15 (averaging window, smaller = better detail, but more noise)
            # iterations=3 (more iterations = more accurate, slower)
            # poly_n=5, poly_sigma=1.1 (sharper edges, less blur)
            flow = cv2.calcOpticalFlowFarneback(prev_gray_dof, gray_blurred, None, 
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            
            # 2. Magnitude calculation (Speed)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # 3. Thresholding: Anything moving faster than 2 pixels/frame is the rat
            # This 2.0 can be tuned for sensitivity
            _, fgmask = cv2.threshold(mag, 2.5, 255, cv2.THRESH_BINARY)
            fgmask = fgmask.astype(np.uint8)
            
            # Update previous frame for next iteration
            prev_gray_dof = gray_blurred.copy()
            
        else:
            # MOG2 DETECTOR (Traditional)
            fgmask = fgbg.apply(gray_blurred, learningRate=0.0005)

        # ------------------------------------

        fgmask = cv2.bitwise_and(fgmask, fgmask, mask=roi_mask)

        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour, measured_coords = get_best_contour(contours, tracker, inner_pts_array, min_area)
        
        # IMPORTANT: update now also requires the gray frame (because of OF)
        # Kalman tracker simply ignores it
        final_pos = tracker.update(measured_coords, frame_gray=gray)

        final_cx, final_cy = None, None
        if final_pos is not None:
            final_cx, final_cy = int(final_pos[0]), int(final_pos[1])
            is_predicted = (measured_coords is None)
            
            # Save
            if cv2.pointPolygonTest(inner_pts_array, (final_cx, final_cy), False) < 0:
                trajectory_data.append((round(timestamp_sec, 3), final_cx, final_cy, is_predicted))
            else:
                final_cx, final_cy = None, None

        # --- VISUALIZATION ---
        if visualization_active:
            # 1. Raw detection box (Green) - Only if contour exists
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 2. Tracker position (Crosshair)
            if final_cx is not None:
                # COLORING: If measured_coords is None, this is PREDICTION -> YELLOW
                # If measured_coords exists, this is MEASUREMENT -> RED
                is_prediction = (measured_coords is None)
                color = (0, 255, 255) if is_prediction else (0, 0, 255) # Yellow vs Red
                
                cv2.drawMarker(frame, (final_cx, final_cy), color, cv2.MARKER_CROSS, 20, 2)
                
                # Optional: print "Pred" next to it
                if is_prediction and debug:
                    cv2.putText(frame, "Pred", (final_cx+10, final_cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 3. Trajectory "trail" (Past points)
            # Only draw last 100 points for speed
            for point in trajectory_data[-100:]:
                # point structure: (time, x, y, is_predicted)
                p_x, p_y = point[1], point[2]
                p_is_pred = point[3]
                
                pt_color = (0, 255, 255) if p_is_pred else (0, 0, 255)
                cv2.circle(frame, (p_x, p_y), 2, pt_color, -1)

            # Draw polygons (Yellow-blue)
            pts_out = outer_pts_array.reshape((-1, 1, 2))
            pts_in = inner_pts_array.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_out], True, (255, 255, 0), 1)
            cv2.polylines(frame, [pts_in], True, (255, 255, 0), 1)

            if debug:
                if method == 'sub_dof_fusion':
                    debug_mog2 = cv2.bitwise_and(mask_mog2, mask_mog2, mask=roi_mask)
                    debug_dof = cv2.bitwise_and(extra_mask_for_debug, extra_mask_for_debug, mask=roi_mask)
                    display_frame = create_debug_view(
                        frame, 
                        debug_mog2,
                        tracker, 
                        measured_coords, 
                        (final_cx, final_cy), 
                        extra_debug_mask=debug_dof
                    )
                else:
                    display_frame = create_debug_view(frame, fgmask, tracker, measured_coords, (final_cx, final_cy))
            else:
                display_frame = frame

            cv2.imshow("Tracking", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nInterrupted.")
                break

    cap.release()
    if visualization_active: cv2.destroyAllWindows()

    if len(trajectory_data) > 0:
        # trajectory_data = clean_trajectory(trajectory_data, max_drift_dist=100)
        final_csv_data = [(t, x, y) for (t, x, y, is_pred) in trajectory_data]

        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        # Add method to filename (kf or of)
        csv_output_file = f"{video_filename}_{method}_trajectory.csv"
        output_path = os.path.join("results", csv_output_file)
        
        save_to_csv(final_csv_data, output_path)

        # Draw map...
        print("Drawing final trajectory...")
        pts_out = outer_pts_array.reshape((-1, 1, 2))
        pts_in = inner_pts_array.reshape((-1, 1, 2))
        cv2.polylines(final_background_image, [pts_out], True, (255, 0, 0), 2)
        cv2.polylines(final_background_image, [pts_in], True, (255, 0, 0), 2)

        # DRAW COLORED POINTS ON MAP
        for point in trajectory_data: # Here use either cleaned data or raw, depending on code state
            p_x, p_y = point[1], point[2]
            
            # Check if 4th element exists (is_predicted)
            if len(point) > 3:
                p_is_pred = point[3]
                color = (0, 255, 255) if p_is_pred else (0, 0, 255) # Yellow vs Red
            else:
                color = (0, 0, 255) # If no info, use red
                
            cv2.circle(final_background_image, (p_x, p_y), 2, color, -1)

        output_image = os.path.join("results", f"{video_filename}_{method}_trajectory_map.jpg")
        cv2.imwrite(output_image, final_background_image)
        print(f"Done. ({method})")
    else:
        print("No data recorded.")