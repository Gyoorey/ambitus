import cv2
import random
import csv
from pathlib import Path

# --- Configuration ---
# Update this to the root folder of your video dataset
DATASET_DIR = "Z:\\dataset\\videos" 
# This ensures the log file is created in the exact same folder as this Python script
LOG_FILE = Path(__file__).parent / "annotation_log.csv"
SUPPORTED_EXTS = {'.mp4'}

# Global variable to store click coordinates
clicked_coords = None

def mouse_callback(event, x, y, flags, param):
    """Callback function triggered by OpenCV window mouse events."""
    global clicked_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords = (x, y)

def get_video_files(directory):
    """Recursively search for video files in the given directory tree."""
    print(f"Scanning for videos in {directory}...")
    video_files = []
    for path in Path(directory).rglob('*'):
        if path.suffix.lower() in SUPPORTED_EXTS:
            video_files.append(str(path))
    return video_files

def main():
    global clicked_coords
    
    videos = get_video_files(DATASET_DIR)
    if not videos:
        print("No videos found. Please check your DATASET_DIR path.")
        return

    print(f"Found {len(videos)} videos.")

    # Create the CSV log file and write the header if it doesn't exist
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["video_path", "frame_number", "x", "y"])

    # Initialize the OpenCV window and attach the mouse listener
    window_name = "Random Annotator"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- Controls ---")
    print("Left Click : Log coordinates and load next random video")
    print("Press 'a'  : Move 0.5s backward")
    print("Press 'd'  : Move 0.5s forward")
    print("Press 's'  : Skip (load next random video)")
    print("Press 'q'  : Quit the program")
    print("----------------\n")

    while True:
        # 1. Grab a random video
        video_path = random.choice(videos)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            continue
            
        # 2. Get video properties to calculate frames for a 0.5s jump
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0 or fps <= 0:
            cap.release()
            continue
            
        jump_frames = max(1, int(fps * 0.5)) # Frames in 0.5 seconds
        current_frame_idx = random.randint(0, total_frames - 1)
        
        # 3. Read the initial random frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            cap.release()
            continue

        # 4. Display frame and wait for interaction
        clicked_coords = None
        cv2.imshow(window_name, frame)

        quit_program = False
        
        # Inner loop to handle clicks and navigation
        while True:
            key = cv2.waitKey(10) & 0xFF
            
            # If the callback registered a click
            if clicked_coords is not None:
                x, y = clicked_coords
                
                # Append data to the log file
                with open(LOG_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([video_path, current_frame_idx, x, y])
                
                print(f"Logged -> Video: {Path(video_path).name} | Frame: {current_frame_idx} | Pos: ({x}, {y})")
                break # Break inner loop to fetch a new random video
            
            # Navigation Controls
            if key == ord('q'):
                quit_program = True
                break
            elif key == ord('s'):
                break # Skip to next video
            elif key == ord('a') or key == ord('d'):
                # Update frame index based on direction
                if key == ord('a'):
                    current_frame_idx = max(0, current_frame_idx - jump_frames)
                else:
                    current_frame_idx = min(total_frames - 1, current_frame_idx + jump_frames)
                
                # Seek to the new frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    cv2.imshow(window_name, frame)
                    clicked_coords = None # Reset click coordinates to avoid accidental logging

        # Release the video capture object before opening a new one
        cap.release()

        if quit_program:
            print(f"Exiting. Logs saved to: {LOG_FILE}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
