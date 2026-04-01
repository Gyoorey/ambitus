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
    # Typical path of the CSV file: [PROJECT_ROOT]/results/[SUBFOLDERS...]/[VIDEO_NAME]/video_static_trajectory.csv
    # Typical video path: [PROJECT_ROOT]/data/[SUBFOLDERS...]/video.mp4

    # Locate the "results" folder in the path
    parts = list(csv_dir.parts)
    
    if "results" in parts:
        idx = parts.index("results")
        
        # The root of the project is the part before "results"
        project_root = Path(*parts[:idx])
        
        # The internal path (subfolders) following "results", but with the last element (which is the video's name) removed
        # For example, if parts = (..., 'results', 'Experiment_1', 'Video_01')
        # Then the subfolder is: ('Experiment_1',)
        sub_folders = parts[idx+1 : -1] 
        
        # Let's set up the expected original video folder:
        # [PROJECT_ROOT] / "data" / [SUBFOLDERS...]
        expected_video_dir = project_root / "data" / Path(*sub_folders)
        
        print(f"Expected original video directory: {expected_video_dir}")
        
        # Let's see if the video exists here with the possible file extensions
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
    Uses sequential reading to avoid MPG seeking bugs.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    # --- 1. LOAD TRAJECTORY DATA ---
    trajectory = []
    with open(csv_path, 'r') as f:
        reader_csv = csv.reader(f, delimiter=';')
        next(reader_csv, None) # Skip header
        for row in reader_csv:
            try:
                t = float(row[0])
                x = int(row[1])
                y = int(row[2])
                trajectory.append((t, x, y))
            except (ValueError, IndexError):
                continue

    if not trajectory:
        print("ERROR: No valid data found in CSV.")
        return

    times = np.array([pt[0] for pt in trajectory])
    xs = np.array([pt[1] for pt in trajectory])
    ys = np.array([pt[2] for pt in trajectory])

    print(f"Loaded {len(trajectory)} points. CSV starts at T={times[0]:.2f}s")

    # --- 2. FIND AND OPEN THE VIDEO ---
    if video_file is not None and os.path.exists(video_file):
        video_path = video_file
    else:
        video_path = find_video_for_csv(csv_path)
        if video_path is None:
            return

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

    wait_time = int((1000 / fps) / speed) if fps > 0 else 40
    tail_frames = int(tail_length_sec * fps)
    
    print(f"Video detected with {fps} FPS. Starting sequential playback...")

    # --- 2.5 INITIALIZE VIDEO WRITER (If saving is requested) ---
    out_video = None
    if save_video:
        # Get frame dimensions by reading the first frame, then reset to 0
        if is_mpg:
            try:
                reader.set_image_index(0)
                temp_frame = reader.get_next_data()
                h, w = temp_frame.shape[:2]
                reader.set_image_index(0) # Reset
            except:
                print("ERROR: Failed to initialize MPG for saving.")
                return
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_t, temp_frame = cap.read()
            h, w = temp_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset
            
        base_dir = os.path.dirname(csv_path)
        filename_base = os.path.splitext(os.path.basename(csv_path))[0].replace('_trajectory_filtered', '')
        out_path = os.path.join(base_dir, f"{filename_base}_playback.mp4")
        
        # 'mp4v' is widely supported in OpenCV for MP4 containers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"Recording output video to: {out_path}")
    
    frame_idx = 0 
    paused = False

    while True:
        if not paused:
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
            
            if is_mpg:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if current_time >= times[0] and current_time <= times[-1] + 1.0:
                
                current_csv_idx = np.argmin(np.abs(times - current_time))
                start_tail_idx = max(0, current_csv_idx - tail_frames)
                
                for i in range(start_tail_idx, current_csv_idx):
                    p1 = (xs[i], ys[i])
                    p2 = (xs[i+1], ys[i+1])
                    
                    age_ratio = (i - start_tail_idx) / float(current_csv_idx - start_tail_idx + 1)
                    b, g, r = 255, int(255 * age_ratio), 0
                    thickness = int(1 + 2 * age_ratio)
                    
                    cv2.line(frame, p1, p2, (b, g, r), thickness)

                current_x = xs[current_csv_idx]
                current_y = ys[current_csv_idx]
                
                cv2.circle(frame, (current_x, current_y), 4, (0, 255, 255), -1)
                cv2.drawMarker(frame, (current_x, current_y), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            
            elif current_time < times[0]:
                cv2.putText(frame, f"Waiting for rat insertion... (T={current_time:.2f}s)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"Time: {current_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            speed_text = f"Speed: {speed}x" if speed != 1.0 else "Speed: 1x"
            cv2.putText(frame, speed_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if paused:
                cv2.putText(frame, "PAUSED", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Filtered Trajectory Playback", frame)

        if out_video is not None:
            out_video.write(frame)
        
        # --- DYNAMIC KEYBOARD CONTROLS ---
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

    if is_mpg:
        reader.close()
    else:
        cap.release()

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