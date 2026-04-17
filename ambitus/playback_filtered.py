import cv2
import numpy as np
import csv
import os
import click
import imageio
from pathlib import Path

def find_video_for_csv(csv_path):
    """ 
    Attempts to automatically find the original video file based on the CSV filename 
    and the mirrored folder structure (results/ -> data/).
    """
    basename = os.path.basename(csv_path)
    name_without_ext = basename.replace('.csv', '')
    if name_without_ext.endswith('_trajectory_filtered'):
        name_without_ext = name_without_ext.replace('_trajectory_filtered', '')
    elif name_without_ext.endswith('_trajectory'):
        name_without_ext = name_without_ext.replace('_trajectory', '')
        
    methods = ['_simple', '_static', '_dof', '_kf', '_of', '_sub_dof_fusion', '_pure_of', '_mhi']
    for method in methods:
        if name_without_ext.endswith(method):
            name_without_ext = name_without_ext[:-len(method)]
            break
            
    extensions = ['.mp4', '.mpg', '.avi', '.mov']
    csv_dir = Path(csv_path).resolve().parent
    
    # 2. DECONSTRUCTING THE STRUCTURE (Mirroring)
    parts = list(csv_dir.parts)
    
    if "results" in parts:
        idx = parts.index("results")
        project_root = Path(*parts[:idx])
        sub_folders = parts[idx+1 : -1] 
        expected_video_dir = project_root / "data" / Path(*sub_folders)
        
        print(f"Expected original video directory: {expected_video_dir}")
        
        if expected_video_dir.exists():
            for ext in extensions:
                possible_path = expected_video_dir / (name_without_ext + ext)
                if possible_path.exists():
                    return str(possible_path)

                possible_path_upper = expected_video_dir / (name_without_ext + ext.upper())
                if possible_path_upper.exists():
                    return str(possible_path_upper)

    # 3. FALLBACK 
    print("WARNING: Could not reconstruct path from 'results/'. Falling back to local directory search.")
    
    for ext in extensions:
        possible_path = csv_dir / (name_without_ext + ext)
        if possible_path.exists():
            return str(possible_path)

    for ext in extensions:
        possible_path = csv_dir.parent / (name_without_ext + ext)
        if possible_path.exists():
            return str(possible_path)

    return None

def play_filtered_trajectory(csv_path, tail_length_sec=3.0, video_dir="data", video_file=None, speed=1.0, save_video=False):
    """
    Plays the original video and overlays the filtered trajectory with a fading tail.
    Visually represents the Behavioral State (Active/Inactive), Spatial Context, 
    and draws a geometric tangent vector (arrow) showing CW/CCW direction.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: Trajectory CSV file not found: {csv_path}")
        return

    # --- 1. KERESSÜK MEG A HOZZÁ TARTOZÓ DEBUG CSV-T (Állapotok és Irányok) ---
    base_dir = os.path.dirname(csv_path)
    filename_base = os.path.splitext(os.path.basename(csv_path))[0].replace('_trajectory_filtered', '').replace('_trajectory', '')
    
    debug_csv_path = os.path.join(base_dir, f"{filename_base}_features_debug.csv")
    
    has_debug_data = False
    states_dict = {}     
    directions_dict = {} 
    corners_dict = {}    
    
    if os.path.exists(debug_csv_path):
        print(f"Found Feature Debug data: {debug_csv_path}")
        has_debug_data = True
        with open(debug_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',') 
            next(reader, None) 
            for row in reader:
                try:
                    t = round(float(row[0]), 3) 
                    state = int(row[4])
                    direction = int(row[5])
                    is_corner = int(row[6])
                    
                    states_dict[t] = state
                    directions_dict[t] = direction
                    corners_dict[t] = is_corner
                except (ValueError, IndexError):
                    continue
    else:
        print(f"WARNING: Feature Debug CSV not found at {debug_csv_path}. States and directions will not be visualized.")

    # --- 2. LOAD TRAJECTORY DATA (A szűrt útvonal) ---
    trajectory = []
    with open(csv_path, 'r') as f:
        reader_csv = csv.reader(f, delimiter=';')
        next(reader_csv, None)
        for row in reader_csv:
            try:
                t = round(float(row[0]), 3)
                x = int(row[1])
                y = int(row[2])
                
                state = states_dict.get(t, -1) if has_debug_data else -1
                direction = directions_dict.get(t, 0) if has_debug_data else 0
                is_corner = corners_dict.get(t, 0) if has_debug_data else 0
                
                trajectory.append((t, x, y, state, direction, is_corner))
            except (ValueError, IndexError):
                continue

    if not trajectory:
        print("ERROR: No valid data found in CSV.")
        return

    print(f"Loaded {len(trajectory)} trajectory points. CSV starts at T={trajectory[0][0]:.2f}s")

    times = np.array([pt[0] for pt in trajectory])
    xs = np.array([pt[1] for pt in trajectory])
    ys = np.array([pt[2] for pt in trajectory])
    states = np.array([pt[3] for pt in trajectory])
    directions = np.array([pt[4] for pt in trajectory])
    corners = np.array([pt[5] for pt in trajectory])

    # Kiszámítjuk az aréna durva középpontját a vektorok pontos rajzolásához
    approx_center_x = np.median(xs)
    approx_center_y = np.median(ys)

    # --- 3. FIND AND OPEN THE VIDEO ---
    if video_file is not None and os.path.exists(video_file):
        video_path = video_file
    else:
        video_path = find_video_for_csv(csv_path)
        if video_path is None: return

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

    # --- 4. VIDEO WRITER INITIALIZATION ---
    out_video = None
    if save_video:
        if is_mpg:
            try:
                reader.set_image_index(0)
                temp_frame = reader.get_next_data()
                h, w = temp_frame.shape[:2]
                reader.set_image_index(0)
            except:
                print("ERROR: Failed to initialize MPG for saving.")
                return
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_t, temp_frame = cap.read()
            h, w = temp_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        out_path = os.path.join(base_dir, f"{filename_base}_playback.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"Recording output video to: {out_path}")

    tail_frames = int(tail_length_sec * fps)
    print(f"Video detected with {fps} FPS. Starting sequential playback...")
    print("Press 'q' or 'ESC' to stop. Press 'space' to pause/resume.")
    
    frame_idx = 0 
    paused = False

    while True:
        if not paused:
            # --- 5. SEQUENTIAL READ & FRAME DROPPING (Fast Forward) ---
            frames_to_read = int(max(1, speed))
            for _ in range(frames_to_read):
                if is_mpg:
                    try:
                        frame = reader.get_next_data()
                        ret = True
                        current_time = frame_idx / fps
                    except:
                        ret = False
                else:
                    ret, frame = cap.read()
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                frame_idx += 1
                if not ret: break
                
            if not ret:
                print("End of video reached.")
                break

            if is_mpg: frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # --- 6. SYNCHRONIZATION AND DRAWING ---
            if current_time >= times[0] and current_time <= times[-1] + 1.0:
                current_csv_idx = np.argmin(np.abs(times - current_time))
                
                # A. Draw the Fading Tail (Üstökös csóva)
                start_tail_idx = max(0, current_csv_idx - tail_frames)
                for i in range(start_tail_idx, current_csv_idx):
                    p1 = (xs[i], ys[i])
                    p2 = (xs[i+1], ys[i+1])
                    age_ratio = (i - start_tail_idx) / float(current_csv_idx - start_tail_idx + 1)
                    b, g, r = 255, int(255 * age_ratio), 0
                    thickness = int(1 + 2 * age_ratio)
                    cv2.line(frame, p1, p2, (b, g, r), thickness)

                # Current Position Variables
                cx = xs[current_csv_idx]
                cy = ys[current_csv_idx]
                c_state = states[current_csv_idx]
                c_direction = directions[current_csv_idx]
                c_corner = corners[current_csv_idx]
                
                if c_state == 2:
                    cross_color = (0, 255, 0)  # BGR Green
                    state_text = "HIGH ACTIVE"
                elif c_state == 1:
                    cross_color = (255, 0, 0)  # BGR Blue (Low Active)
                    state_text = "LOW ACTIVE"
                elif c_state == 0:
                    cross_color = (0, 0, 255)  # BGR Red (Inactive)
                    state_text = "INACTIVE"
                else:
                    cross_color = (0, 0, 0)    # BGR Black (Fallback)
                    state_text = "UNKNOWN"

                # Draw the Crosshair
                cv2.circle(frame, (cx, cy), 4, cross_color, -1)
                cv2.drawMarker(frame, (cx, cy), cross_color, cv2.MARKER_CROSS, 15, 2)
                
                # Draw the Status Text next to the rat
                cv2.putText(frame, state_text, (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cross_color, 2)

                # B. VECTOR DRAWING (Haladási irány nyíl)
                if has_debug_data and c_direction != 0:
                    # Relatív pozíció a középponthoz képest
                    rx = cx - approx_center_x
                    ry = cy - approx_center_y
                    
                    # Matematikai érintő vektorok (OpenCV Y tengely lefelé nő)
                    if c_direction == 1: # CW (Óramutató járásával megegyező)
                        tx, ty = -ry, rx
                    else: # CCW (Óramutató járásával ellentétes)
                        tx, ty = ry, -rx
                        
                    v_len = np.sqrt(tx**2 + ty**2)
                    if v_len > 0.1:
                        arrow_len = 50 # A nyíl hossza pixelben
                        end_x = int(cx + (tx / v_len) * arrow_len)
                        end_y = int(cy + (ty / v_len) * arrow_len)
                        
                        # Narancs a CW, Sárgás/Cián a CCW
                        arrow_color = (0, 165, 255) if c_direction == 1 else (255, 255, 0)
                        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), arrow_color, 4, tipLength=0.3)

            elif current_time < times[0]:
                cv2.putText(frame, f"Waiting for rat insertion... (T={current_time:.2f}s)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # On-Screen Information Panel (Top Left)
            cv2.putText(frame, f"Time: {current_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            speed_text = f"Speed: {speed}x" if speed != 1.0 else "Speed: 1x"
            cv2.putText(frame, speed_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if has_debug_data and (current_time >= times[0] and current_time <= times[-1] + 1.0):
                corner_txt = "CORNER" if c_corner == 1 else "CORRIDOR"
                cv2.putText(frame, f"Sector: {corner_txt}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Yellow

        if paused:
            h, w = frame.shape[:2]
            cv2.putText(frame, "PAUSED", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Filtered Trajectory Playback", frame)
        if out_video is not None:
            out_video.write(frame)
        
        # --- 7. KEYBOARD CONTROLS ---
        base_wait = int(1000 / fps) if fps > 0 else 40
        current_wait = int(base_wait / speed) if speed < 1.0 else base_wait
        
        key = cv2.waitKey(current_wait if not paused else 100) & 0xFF
        
        if key == ord('q') or key == 27: 
            break
        elif key == 32: 
            paused = not paused
        elif key == ord(']') or key == ord('+') or key == ord('f'): 
            speed *= 2.0
            if speed > 16.0: speed = 16.0 
        elif key == ord('[') or key == ord('-') or key == ord('s'): 
            speed /= 2.0
            if speed < 0.25: speed = 0.25 
        elif key == ord('1'): 
            speed = 1.0

    if is_mpg: reader.close()
    else: cap.release()
        
    if out_video is not None:
        out_video.release()
        print(f"\nPlayback video successfully saved to disk!")
        
    cv2.destroyAllWindows()


@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--video-file", default=None, help="Direct path to the video file (bypasses directory search)")
@click.option("--tail", default=3.0, help="Length of the visible trajectory tail in seconds (default: 3.0)")
@click.option("--speed", default=1.0, help="Playback speed multiplier (e.g., 2.0 for double speed)")
@click.option("--save", is_flag=True, help="Save the playback with overlaid trajectory as a new video file") 
def cli(csv_path, video_dir, video_file, tail, speed, save):
    """
    Plays the original video and dynamically overlays the filtered trajectory 
    with a fading tail to visualize the rat's movement over time.
    """
    play_filtered_trajectory(csv_path, tail_length_sec=tail, video_file=video_file, speed=speed, save_video=save)

if __name__ == '__main__':
    cli()