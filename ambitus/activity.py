import cv2
import numpy as np
import csv
import os
import json
from pathlib import Path
import click
import imageio
from itertools import groupby

def find_video_for_csv(csv_path):
    """ Attempts to automatically find the original video file based on the CSV filename. """
    basename = os.path.basename(csv_path)
    name_without_ext = basename.replace('.csv', '')
    for suffix in ['_trajectory_filtered', '_filtered', '_trajectory']:
        if name_without_ext.endswith(suffix):
            name_without_ext = name_without_ext.replace(suffix, '')
            break
    for method in ['_simple', '_static', '_dof', '_kf', '_of', '_sub_dof_fusion', '_pure_of', '_mhi']:
        if name_without_ext.endswith(method):
            name_without_ext = name_without_ext[:-len(method)]
            break
            
    extensions = ['.mp4', '.mpg', '.avi', '.mov']
    csv_dir = Path(csv_path).resolve().parent
    
    parts = list(csv_dir.parts)
    if "results" in parts:
        idx = parts.index("results")
        project_root = Path(*parts[:idx])
        sub_folders = parts[idx+1 : -1] 
        expected_video_dir = project_root / "data" / Path(*sub_folders)
        if expected_video_dir.exists():
            for ext in extensions:
                p1 = expected_video_dir / (name_without_ext + ext)
                if p1.exists(): return str(p1)
                p2 = expected_video_dir / (name_without_ext + ext.upper())
                if p2.exists(): return str(p2)
                
    for ext in extensions:
        p1 = csv_dir / (name_without_ext + ext)
        if p1.exists(): return str(p1)
        p2 = csv_dir.parent / (name_without_ext + ext)
        if p2.exists(): return str(p2)
    return None

def classify_activities(csv_path, output_path=None, box_size=80, sad_threshold=20, high_active_speed=None, debug=False):
    """
    Video-based activity classifier distinguishing Locomotion, Micro-movements, and Resting.
    States: 2 = High Active (Locomotion), 1 = Low Active (Grooming/Sniffing), 0 = Inactive (Resting)
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    # --- 0. FIND GLOBAL THRESHOLDS (High Active Speed) ---
    if high_active_speed is None:
        csv_dir = Path(csv_path).resolve().parent
        parts = list(csv_dir.parts)
        
        # Determine the project root to find global_thresholds.json
        if "results" in parts:
            idx = parts.index("results")
            project_root = Path(*parts[:idx+1]) # Up to 'results' folder
        else:
            project_root = csv_dir # Fallback
            
        global_thresh_file = project_root / "global_thresholds.json"
        
        if global_thresh_file.exists():
            try:
                with open(global_thresh_file, 'r') as f:
                    glob_data = json.load(f)
                    if "T_upper_px_s" in glob_data:
                        high_active_speed = glob_data["T_upper_px_s"]
                        print(f"Loaded Global High Active Speed Threshold: {high_active_speed} px/s")
            except Exception as e:
                print(f"WARNING: Could not read global thresholds ({e}).")
                
        # If still None (file missing or missing key), use a reasonable fallback
        if high_active_speed is None:
            high_active_speed = 35.0
            print(f"WARNING: Global thresholds not found. Using fallback High Active Speed: {high_active_speed} px/s")

    # --- 1. LOAD TRAJECTORY DATA ---
    times, xs, ys = [], [], []
    with open(csv_path, 'r') as f:
        reader_csv = csv.reader(f, delimiter=';')
        header = next(reader_csv, None)
        for row in reader_csv:
            try:
                times.append(float(row[0]))
                xs.append(int(row[1]))
                ys.append(int(row[2]))
            except (ValueError, IndexError):
                continue

    if len(times) < 10:
        print("ERROR: Not enough data points.")
        return

    times = np.array(times)
    xs = np.array(xs)
    ys = np.array(ys)
    
    dt = np.diff(times)
    dt = np.append(dt, dt[-1]) 
    dx = np.append(np.diff(xs), 0)
    dy = np.append(np.diff(ys), 0)
    speeds = np.sqrt(dx**2 + dy**2) / dt

    states = np.zeros(len(times), dtype=int)
    sad_values = np.zeros(len(times), dtype=float) 

    # --- 2. FIND AND OPEN VIDEO ---
    video_path = find_video_for_csv(csv_path)
    if video_path is None:
        print(f"ERROR: Could not find video for CSV: {csv_path}")
        return

    print(f"Analyzing micro-movements on video: {video_path}")
    is_mpg = video_path.lower().endswith('.mpg')
    
    if is_mpg:
        input_params = []
        output_params = ['-vf', 'yadif,scale=768:576,setsar=1']
        reader = imageio.get_reader(video_path, 'ffmpeg', input_params=input_params, output_params=output_params)
        meta = reader.get_meta_data()
        fps = meta.get('fps', 25.0)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return
        fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame_idx = int(times[0] * fps)
    
    print(f"Fast-forwarding video to T={times[0]:.2f}s (Frame {start_frame_idx})...")
    current_frame_idx = 0
    while current_frame_idx < start_frame_idx:
        if is_mpg:
            try: reader.get_next_data()
            except: break
        else:
            ret, _ = cap.read()
            if not ret: break
        current_frame_idx += 1

    # --- 3. FRAME-BY-FRAME ANALYSIS (Intelligent Common-Window) ---
    half_box = box_size // 2
    fps_int = int(fps) if fps > 0 else 25
    frame_delay = 5
    
    frame_history = [] 
    
    print(f"Extracting pixel variance (SAD) using an Intelligent Spatial Window with a {frame_delay}-frame delay...")
    
    for i in range(len(times)):
        if is_mpg:
            try:
                frame = reader.get_next_data()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret = True
            except: ret = False
        else:
            ret, frame = cap.read()

        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        frame_history.append(gray)
        
        current_sad = 0.0
        diff_thresh = None
        
        cx_current, cy_current = xs[i], ys[i]

        if len(frame_history) <= frame_delay:
            states[i] = 2 if speeds[i] > high_active_speed else 0 
            sad_values[i] = 0
            continue

        delayed_frame = frame_history.pop(0)

        if speeds[i] > high_active_speed:
            states[i] = 2 
            sad_values[i] = 0 
        else:
            cx_delayed, cy_delayed = xs[i - frame_delay], ys[i - frame_delay]
            
            min_x = min(cx_current, cx_delayed) - half_box
            max_x = max(cx_current, cx_delayed) + half_box
            min_y = min(cy_current, cy_delayed) - half_box
            max_y = max(cy_current, cy_delayed) + half_box
            
            crop_x1, crop_y1 = max(0, min_x), max(0, min_y)
            crop_x2, crop_y2 = min(w, max_x), min(h, max_y)
            
            current_box = gray[crop_y1:crop_y2, crop_x1:crop_x2]
            delayed_box = delayed_frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            diff = cv2.absdiff(current_box, delayed_box)
            _, diff_thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((3, 3), np.uint8)
            diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            current_sad = np.sum(diff_thresh) / 255.0 
            sad_values[i] = current_sad
            
            if current_sad > sad_threshold:
                states[i] = 1 # Low Active
            else:
                states[i] = 0 # Inactive

        # --- 4. DEBUG VISUALIZATION (Real-time Playback) ---
        if debug:
            disp = frame.copy()
            
            color = (0, 0, 0)
            text = "UNKNOWN"
            if states[i] == 2:
                color = (0, 255, 0)
                text = "HIGH ACTIVE (Speed)"
            elif states[i] == 1:
                color = (0, 255, 255)
                text = "LOW ACTIVE (Pixel Var)"
            elif states[i] == 0:
                color = (0, 0, 255)
                text = "INACTIVE (Resting)"
                
            if states[i] != 2 and diff_thresh is not None:
                cv2.rectangle(disp, (crop_x1, crop_y1), (crop_x2, crop_y2), color, 2)
            else:
                cv2.rectangle(disp, (cx_current - half_box, cy_current - half_box), 
                              (cx_current + half_box, cy_current + half_box), color, 2)
                
            cv2.drawMarker(disp, (cx_current, cy_current), color, cv2.MARKER_CROSS, 10, 2)
            
            cv2.putText(disp, f"State: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(disp, f"Speed: {speeds[i]:.1f} px/s (Thresh: {high_active_speed:.1f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if states[i] != 2:
                sad_val = sad_values[i]
                sad_color = (0, 255, 255) if sad_val > sad_threshold else (0, 0, 255)
                
                cv2.putText(disp, f"{frame_delay}-Frame Delay SAD: {int(sad_val)} px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sad_color, 2)
                cv2.putText(disp, f"SAD Threshold: {sad_threshold}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                if diff_thresh is not None:
                    diff_vis = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2BGR)
                    
                    dh_orig, dw_orig = diff_vis.shape[:2]
                    scale = min(200 / dw_orig, 200 / dh_orig) if dw_orig > 0 and dh_orig > 0 else 1.0
                    new_w, new_h = int(dw_orig * scale), int(dh_orig * scale)
                    
                    if new_w > 0 and new_h > 0:
                        diff_vis = cv2.resize(diff_vis, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                        disp[10:10+new_h, w-new_w-10:w-10] = diff_vis
                        cv2.rectangle(disp, (w-new_w-10, 10), (w-10, 10+new_h), color, 2)
                        cv2.putText(disp, f"Smart Diff Window", (w-new_w-10, 10+new_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Activity Classifier Debug", disp)
            
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == 32: 
                cv2.waitKey(0)

    if is_mpg: reader.close()
    else: cap.release()
    if debug: cv2.destroyAllWindows()

    # --- 5. DEBOUNCING (Smooth out 1-frame spikes) ---
    min_gap_frames = int(1.0 * fps_int)
    for state_val in [1, 2]:
        start_idx = 0
        for key, group in groupby(states):
            length = sum(1 for _ in group)
            if key == 0 and length <= min_gap_frames:
                prev_st = states[max(0, start_idx-1)]
                if prev_st == state_val:
                    states[start_idx : start_idx + length] = state_val
            start_idx += length

    # --- 6. SAVE TO NEW CSV ---
    if output_path is None:
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(os.path.dirname(csv_path), f"{filename}_classified.csv")

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Time_Sec", "X", "Y", "State"])
        for t, x, y, s in zip(times, xs, ys, states):
            writer.writerow([round(t, 3), x, y, s])
            
    print(f"Classification successful! Saved to: {output_path}")

@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--output", default=None, help="Output CSV filename")
@click.option("--box-size", default=80, help="Bounding box size in pixels around the rat")
@click.option("--sad-thresh", default=20, help="Number of changed pixels to consider Low Active")
@click.option("--speed-thresh", type=float, default=None, help="Speed in px/s to consider High Active (Auto-detected if not provided)")
@click.option("--debug", is_flag=True, help="Show a real-time visual playback of the classification")
def cli(csv_path, output, box_size, sad_thresh, speed_thresh, debug):
    """
    Analyzes the video at the tracked positions to distinguish between 
    High Active, Low Active (Micro-movements), and Inactivity (Resting).
    Saves a new *_classified.csv file with a 'State' column (0, 1, 2).
    """
    classify_activities(
        csv_path, 
        output_path=output, 
        box_size=box_size, 
        sad_threshold=sad_thresh, 
        high_active_speed=speed_thresh, 
        debug=debug
    )

if __name__ == '__main__':
    cli()