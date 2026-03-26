import cv2
import numpy as np
import json
import os
import csv
import imageio

# --- CONSTANTS ---
MAX_JUMP_DIST = 100 

class RatTracker:
    """
    Simplified state machine for rat tracking.
    No estimation (no Kalman), only tracks position and missed frames.
    """
    def __init__(self, name):
        self.name = name
        self.missed_frames = 0
        self.is_initialized = False
        self.last_valid_pos = None 

    def update(self, measured_coords):
        if measured_coords is not None:
            self.is_initialized = True
            self.last_valid_pos = measured_coords
            return measured_coords
        else:
            self.missed_frames += 1
            return None

# --- HELPER FUNCTIONS ---
def load_config(config_file):
    if not os.path.exists(config_file):
        print(f"ERROR: Configuration file not found: {config_file}")
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
        print(f"\nSuccessfully saved: {filename} ({len(data)} points)")
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
        
        # Exclude inner polygon
        if cv2.pointPolygonTest(inner_pts_array, (cx, cy), False) >= 0: continue
        valid_candidates.append((c, (cx, cy), area))

    if not valid_candidates: return None, None

    # Amnesia (Release Teleportation Lock after 20 missed frames)
    if not tracker.is_initialized or tracker.last_valid_pos is None or tracker.missed_frames > 20:
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

def get_robust_center(fgmask):
    """
    Calculates a robust center (median) and standard deviation of the foreground mask
    using Interquartile Range (IQR) to filter out outliers.
    Returns:
        iqr_coords (tuple or None): (x_median, y_median)
        devs (tuple): (x_dev, y_dev)
    """
    y_coords, x_coords = np.where(fgmask == 255)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None, (0, 0)

    # Calculate quartiles and IQR
    x_q1, x_q3 = np.percentile(x_coords, [25, 75])
    y_q1, y_q3 = np.percentile(y_coords, [25, 75])
    x_iqr = x_q3 - x_q1
    y_iqr = y_q3 - y_q1
    
    # Define bounds to filter outliers
    x_lower_bound = x_q1 - 1.5 * x_iqr
    x_upper_bound = x_q3 + 1.5 * x_iqr
    y_lower_bound = y_q1 - 1.5 * y_iqr
    y_upper_bound = y_q3 + 1.5 * y_iqr

    valid_indices = np.where(
        (x_coords >= x_lower_bound) & (x_coords <= x_upper_bound) &
        (y_coords >= y_lower_bound) & (y_coords <= y_upper_bound)
    )[0]

    if len(valid_indices) == 0:
        return None, (0, 0)

    x_median = int(np.median(x_coords[valid_indices]))
    y_median = int(np.median(y_coords[valid_indices]))
    
    x_dev = np.std(x_coords[valid_indices])
    y_dev = np.std(y_coords[valid_indices])

    return (x_median, y_median), (x_dev, y_dev)

# ==============================================================================
# BACKGROUND GENERATION (Quiet Window Detection)
# ==============================================================================
def find_quiet_window(video_path, is_mpg, reader, cap, fps, search_seconds=60.0, window_sec=5.0, method='sad'):
    """
    Scans the beginning of the video to find the most stable, motionless period.
    
    Args:
        method (str): 'sad' for Sum of Absolute Differences (default), or 'intensity' for intensity variance
    
    Returns:
        int: Start frame index of the quietest window
    """
    print(f"Scanning the first {search_seconds}s for a clean background window using {method} method...")
    
    scale = 0.2 
    max_frames = int(search_seconds * fps)
    window_frames = int(window_sec * fps)
    
    # Ensure we start from frame 0
    if is_mpg:
        try:
            reader.set_image_index(0)
        except:
            pass
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if method == 'intensity':
        intensity_array = []
        
        for i in range(max_frames):
            if is_mpg:
                try:
                    frame = reader.get_next_data()
                    ret = True
                except:
                    ret = False
            else:
                ret, frame = cap.read()
                
            if not ret: break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if not is_mpg else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
            intensity_array.append(np.sum(small))

        if len(intensity_array) < window_frames:
            print("WARNING: Video is too short to find a quiet window. Using frame 0.")
            return 0

        # Signal processing pipeline: Mean subtraction -> Median filter -> Abs -> Sliding STD
        mean_intensity = np.mean(intensity_array)
        normalized_intensity = np.array(intensity_array) - mean_intensity
        
        normalized_intensity = np.array([
            np.median(normalized_intensity[max(0, i-3):min(len(normalized_intensity), i+4)]) 
            for i in range(len(normalized_intensity))
        ])
        
        normalized_intensity = np.abs(normalized_intensity)  
        
        std_window = 35
        normalized_intensity = np.array([
            np.std(normalized_intensity[max(0, i-std_window//2):min(len(normalized_intensity), i+(std_window//2)+1)]) 
            for i in range(len(normalized_intensity))
        ])

        motion_profile = normalized_intensity
        
    else:  # method == 'sad' (default)
        motion_profile = []
        prev_gray = None
        
        for i in range(max_frames):
            if is_mpg:
                try:
                    frame = reader.get_next_data()
                    ret = True
                except:
                    ret = False
            else:
                ret, frame = cap.read()
                
            if not ret: break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if not is_mpg else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
            
            if prev_gray is not None:
                diff = cv2.absdiff(small, prev_gray)
                motion_score = np.sum(diff)
                motion_profile.append(motion_score)
            else:
                motion_profile.append(0)
                
            prev_gray = small

        if len(motion_profile) < window_frames:
            print("WARNING: Video is too short to find a quiet window. Using frame 0.")
            return 0

    # --- FIND THE QUIETEST WINDOW ---
    min_avg_motion = float('inf')
    best_start_frame = 0
    
    for i in range(len(motion_profile) - window_frames):
        window_slice = motion_profile[i : i + window_frames]
        score = np.mean(window_slice) + np.std(window_slice)
        
        if score < min_avg_motion:
            min_avg_motion = score
            best_start_frame = i
            
    best_start_sec = best_start_frame / fps
    print(f"Found quietest {window_sec}s window starting at {round(best_start_sec, 2)}s.")
    
    return best_start_frame

def generate_median_background(video_path, is_mpg, reader, cap, start_frame, fps, window_sec=5.0, num_samples=30):
    """
    Extracts 'num_samples' frames evenly spaced across the quiet window 
    and calculates their pixel-wise median to create a flawless background.
    """
    print("Generating perfect median background from the quiet window...")
    frames = []
    
    # Calculate the step size to get exactly 'num_samples' frames within the window
    window_frames = int(window_sec * fps)
    step = max(1, window_frames // num_samples)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    
    for i in range(0, window_frames, step):
        target_frame = start_frame + i
        
        if is_mpg:
            try:
                reader.set_image_index(target_frame)
                frame = reader.get_next_data()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret = True
            except:
                ret = False
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
        if ret:
            # Apply the exact same preprocessing as in the main tracking loop
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            frames.append(gray_blurred)
            
    if not frames:
        print("ERROR: Failed to extract frames for median background!")
        return None
        
    # Calculate the pixel-wise median across all collected frames
    median_bg = np.median(frames, axis=0).astype(np.float32)
    print("Median background generation complete.")
    
    return median_bg

def create_debug_view(frame, fgmask, tracker, measured_coords, final_pos):
    """ Clean debug view builder. """
    h, w = frame.shape[:2]
    panel_width = 400 
    target_h = int(h / 2)
    
    mask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    mask_resized = cv2.resize(mask_bgr, (panel_width, target_h))
    
    cv2.rectangle(mask_resized, (0,0), (panel_width-1, target_h-1), (100,100,100), 2)    
    cv2.putText(mask_resized, tracker.name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    info_panel = np.zeros((h - target_h, panel_width, 3), dtype=np.uint8)
    missed = tracker.missed_frames
    
    y_off, line_h = 30, 30
    status_color = (0, 255, 0)
    status_text = "TRACKING"
    if missed > 0 and final_pos[0] is None:
        status_color = (0, 0, 255)
        status_text = "LOST / NO DETECT"

    cv2.putText(info_panel, f"METHOD: {tracker.name}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2); y_off += line_h
    cv2.putText(info_panel, f"STATUS: {status_text}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2); y_off += line_h
    cv2.putText(info_panel, f"Meas: {measured_coords}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1); y_off += line_h
    cv2.putText(info_panel, f"Pos:  {final_pos}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1); y_off += line_h * 2
    cv2.putText(info_panel, f"Missed Frames: {missed}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)

    right_sidebar = np.vstack([mask_resized, info_panel])
    if right_sidebar.shape[0] != h:
        right_sidebar = cv2.resize(right_sidebar, (panel_width, h))
        
    return np.hstack([frame, right_sidebar])

# --- MAIN TRACKING LOGIC ---
def track(video_path, config_file="arena_config_poly.json", method='simple', show_traj=False, debug=False, skip_seconds=0.0, min_area=200, output_dir=None):
    config = load_config(config_file)
    if config is None: return

    inner_pts_array = np.array(config["inner_poly"], dtype=np.int32)
    outer_pts_array = np.array(config["outer_poly"], dtype=np.int32)

    is_mpg = video_path.lower().endswith('.mpg')

    if is_mpg:
        input_params = []
        output_params = [
            '-vf', 'yadif,scale=768:576,setsar=1',
        ]
        reader = imageio.get_reader(video_path, 'ffmpeg', 
                                    input_params=input_params, 
                                    output_params=output_params)
        meta = reader.get_meta_data()
        fps = meta.get('fps', 25.0)
        try:
            total_frames = reader.count_frames()
        except:
            total_frames = 0
            
        start_frame = int(skip_seconds * fps)
        
        try:
            if start_frame > 0:
                reader.set_image_index(start_frame)
            first_frame = reader.get_next_data()
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            ret = True
        except:
            ret = False
    else:
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
    
    # --- TRACKER INITIALIZATION ---
    if method == 'simple':
        print("Starting: Simple (MOG2) method")
        tracker = RatTracker("Simple (MOG2)")
        fgbg = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=30, detectShadows=False)
        running_avg_bg = None
        rat_mask_for_bg_update = None
        
    elif method == 'static':
        print("Starting: Static (Selective Averaging) method")
        tracker = RatTracker("Static Diff")

        quiet_start_frame = find_quiet_window(
            video_path, is_mpg,
            reader if is_mpg else None,
            cap if not is_mpg else None,
            fps)

        running_avg_bg = generate_median_background(
            video_path, is_mpg, 
            reader if is_mpg else None, 
            cap if not is_mpg else None, 
            quiet_start_frame, fps
        )
        if running_avg_bg is None:
            gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            gray_first = clahe.apply(gray_first)
            gray_first = cv2.GaussianBlur(gray_first, (7, 7), 0)
            running_avg_bg = np.float32(gray_first)

        rat_mask_for_bg_update = np.zeros_like(running_avg_bg, dtype=np.uint8)
        if is_mpg:
                try:
                    reader.set_image_index(start_frame)
                except:
                    print("WARNING: Failed to set start frame for MPG. Starting from the beginning.")
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    trajectory_data = []
    frame_counter = start_frame
    visualization_active = show_traj or debug

    while True:
        if is_mpg:
            try:
                frame = reader.get_next_data()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret = True
                timestamp_sec = frame_counter / fps
            except (IndexError, Exception):
                ret = False
        else:
            ret, frame = cap.read()
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if not ret: break

        frame_counter += 1
        if not visualization_active and frame_counter % 100 == 0:
            print(f"\rProcessed: {frame_counter}/{total_frames} frames", end="")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # --- DETECTION ---
        if method == 'static':
            # Create a protective mask around the confirmed rat
            kernel_protect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            dilated_rat = cv2.dilate(rat_mask_for_bg_update, kernel_protect, iterations=1)
            
            # Invert: Update where there is no rat (255), freeze where rat is (0)
            bg_update_mask = cv2.bitwise_not(dilated_rat)
            
            cv2.accumulateWeighted(gray_blurred, running_avg_bg, 0.005, mask=bg_update_mask)
            bg_uint8 = cv2.convertScaleAbs(running_avg_bg)
            
            diff = cv2.absdiff(gray_blurred, bg_uint8)
            _, fgmask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
        else: # simple
            fgmask = fgbg.apply(gray_blurred, learningRate=0.0005)

        # --- POST-PROCESSING ---
        fgmask = cv2.bitwise_and(fgmask, fgmask, mask=roi_mask)
        
        # Special morphological filtering against elongated shapes (walls)
        kernel = np.ones((3, 3), np.uint8)
        kernel1 = np.ones((9, 1), np.uint8)
        kernel2 = np.ones((1, 9), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel1, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel2, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # --- ROBUST IQR-BASED CENTER DETECTION ---
        iqr_coords, (x_dev, y_dev) = get_robust_center(fgmask)

        # --- CONTOUR SEARCH (For fallback and mask update) ---
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour, contour_coords = get_best_contour(contours, tracker, inner_pts_array, min_area)
        
        # DECISION: Use IQR median if available, otherwise fallback to contour.
        measured_coords = iqr_coords if iqr_coords is not None else contour_coords

        # Update static mask for the NEXT frame (requires best_contour)
        if method == 'static':
            rat_mask_for_bg_update.fill(0) 
            if best_contour is not None:
                cv2.drawContours(rat_mask_for_bg_update, [best_contour], -1, 255, thickness=cv2.FILLED)

        # --- UPDATE TRACKER ---
        final_pos = tracker.update(measured_coords)

        # --- DATA RECORDING ---
        final_cx, final_cy = None, None
        if final_pos is not None:
            final_cx, final_cy = int(final_pos[0]), int(final_pos[1])
            is_predicted = False # Always measured in this version
            
            if cv2.pointPolygonTest(inner_pts_array, (final_cx, final_cy), False) < 0:
                trajectory_data.append((round(timestamp_sec, 3), final_cx, final_cy, is_predicted))
            else:
                final_cx, final_cy = None, None

        # --- VISUALIZATION ---
        if visualization_active:
            # 1. Green box (Contour)
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 2. Blue box (IQR median and deviation)
            if iqr_coords is not None:
                cx, cy = iqr_coords
                cv2.rectangle(frame, 
                              (int(cx - x_dev), int(cy - y_dev)),
                              (int(cx + x_dev), int(cy + y_dev)), 
                              (255, 0, 0), 2)

            # 3. Crosshair (Final tracker position)
            if final_cx is not None:
                cv2.drawMarker(frame, (final_cx, final_cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # 4. Trajectory trail
            for point in trajectory_data[-100:]:
                cv2.circle(frame, (point[1], point[2]), 2, (0, 0, 255), -1)

            # Draw polygons
            pts_out = outer_pts_array.reshape((-1, 1, 2))
            pts_in = inner_pts_array.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_out], True, (255, 255, 0), 1)
            cv2.polylines(frame, [pts_in], True, (255, 255, 0), 1)

            display_frame = create_debug_view(frame, fgmask, tracker, measured_coords, (final_cx, final_cy)) if debug else frame

            cv2.imshow("Tracking", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nInterrupted.")
                break

    if is_mpg:
        reader.close()
    else:
        cap.release()
    
    if visualization_active: cv2.destroyAllWindows()

    if len(trajectory_data) > 0:
        video_filename = os.path.splitext(os.path.basename(video_path))[0]

        if output_dir is None:
            output_dir = os.path.join("results", video_filename)

        os.makedirs(output_dir, exist_ok=True)
        csv_output_file = f"{video_filename}_{method}_trajectory.csv"
        output_path = os.path.join(output_dir, csv_output_file)
        
        # Drop the 4th is_predicted element for saving
        final_csv_data = [(t, x, y) for (t, x, y, is_pred) in trajectory_data]
        save_to_csv(final_csv_data, output_path)

        print("Drawing map...")
        pts_out = outer_pts_array.reshape((-1, 1, 2))
        pts_in = inner_pts_array.reshape((-1, 1, 2))
        cv2.polylines(final_background_image, [pts_out], True, (255, 0, 0), 2)
        cv2.polylines(final_background_image, [pts_in], True, (255, 0, 0), 2)

        for point in trajectory_data:
            cv2.circle(final_background_image, (point[1], point[2]), 2, (0, 0, 255), -1)

        output_image = os.path.join(output_dir, f"{video_filename}_{method}_trajectory_map.jpg")
        cv2.imwrite(output_image, final_background_image)
        print(f"Done! ({method}), missed frames: {tracker.missed_frames}, trajectory points: {len(trajectory_data)}")
        stats_output = os.path.join(output_dir, f"{video_filename}_{method}_statistics.txt")
        with open(stats_output, 'w') as f:
            f.write(f"Tracking Statistics\n")
            f.write(f"===================\n\n")
            f.write(f"Video: {video_filename}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Total Frames Processed: {frame_counter - start_frame}\n")
            f.write(f"Trajectory Points: {len(trajectory_data)}\n")
            f.write(f"Missed Frames: {tracker.missed_frames}\n")
            f.write(f"Duration (sec): {round(timestamp_sec, 2)}\n")
        print(f"Statistics saved to: {stats_output}")
    else:
        print("\nNo data recorded.")