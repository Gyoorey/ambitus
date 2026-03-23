import cv2
import numpy as np
import csv
import os
import click
import imageio

def find_video_for_csv(csv_path, search_dir="data"):
    """
    Attempts to automatically find the original video file based on the CSV filename.
    It strips the known tracking suffixes to isolate the original video name.
    """
    basename = os.path.basename(csv_path)
    
    # Remove the "_trajectory.csv" part
    name_without_ext = basename.replace('.csv', '')
    if name_without_ext.endswith('_trajectory_filtered'):
        name_without_ext = name_without_ext.replace('_trajectory_filtered', '')
    elif name_without_ext.endswith('_trajectory'):
        name_without_ext = name_without_ext.replace('_trajectory', '')
        
    # Remove the tracking method suffix to get the raw video filename
    methods = ['_simple', '_static', '_dof', '_kf', '_of', '_sub_dof_fusion', '_pure_of', '_mhi']
    for method in methods:
        if name_without_ext.endswith(method):
            name_without_ext = name_without_ext[:-len(method)]
            break
            
    # Possible video extensions and directories to search
    extensions = ['.mp4', '.mpg'] 
    search_dirs = [search_dir, '.', '..'] 
    
    for d in search_dirs:
        for ext in extensions:
            possible_path = os.path.join(d, name_without_ext + ext)
            if os.path.exists(possible_path):
                return possible_path
                
    return None

def generate_heatmap(csv_path, output_path=None, sigma=15, alpha=0.8, show=True, video_dir="data"):
    """
    Generates a heatmap from the CSV trajectory data and overlays it on the first frame of the video.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    # 1. Auto-discover the original video file
    video_path = find_video_for_csv(csv_path, search_dir=video_dir)
    
    if video_path is None:
        print(f"ERROR: Could not automatically find the video for CSV: {csv_path}")
        print(f"Make sure the video is located in the '{video_dir}' folder.")
        return
        
    print(f"Found corresponding video: {video_path}")

    # 2. Read the first frame of the video to use as the background
    if video_path.lower().endswith('.mpg'):
        input_params = []
        output_params = [
            '-vf', 'yadif,scale=768:576,setsar=1',
        ]
        reader = imageio.get_reader(video_path, 'ffmpeg', 
                                    input_params=input_params, 
                                    output_params=output_params )
        frame = reader.get_data(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        reader.close()
        ret = True
    else:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
    
    if not ret:
        print("ERROR: Failed to read the first frame of the video.")
        return

    h, w = frame.shape[:2]

    # 3. Create an empty 2D array to store the density (point accumulation)
    density_map = np.zeros((h, w), dtype=np.float32)

    # 4. Parse the CSV and accumulate points
    point_count = 0
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader, None) # Skip the header
        for row in reader:
            try:
                # Expected columns: Time_Sec; X; Y
                x = int(row[1])
                y = int(row[2])
                
                # Boundary check to prevent IndexError
                if 0 <= x < w and 0 <= y < h:
                    density_map[y, x] += 1
                    point_count += 1
            except (ValueError, IndexError):
                continue
                
    if point_count == 0:
        print("ERROR: No valid coordinates found in the CSV file.")
        return

    print(f"Loaded {point_count} points. Generating heatmap...")

    # 5. Apply Gaussian Blur to create a continuous density map
    # The sigma parameter controls the "thickness" or "spread" of the path
    smoothed_map = cv2.GaussianBlur(density_map, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 6. Normalize values between 0.0 and 1.0
    max_density = np.max(smoothed_map)
    if max_density > 0:
        normalized_map = smoothed_map / max_density
    else:
        normalized_map = smoothed_map

    # 7. Apply the colormap (JET scale: Blue -> Red)
    heatmap_8u = np.uint8(255 * normalized_map)
    color_map = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)

    # 8. Blend the heatmap with the original video frame
    overlay = frame.copy()
    
    # Create a mask to only apply color where there is significant density (> 2%)
    # This keeps the untouched areas of the arena in their original color
    mask = normalized_map > 0.02 

    # Alpha blending on the masked areas
    for c in range(3): # Loop over B, G, R channels
        overlay[mask, c] = frame[mask, c] * (1 - alpha) + color_map[mask, c] * alpha

    # 9. Save the resulting image
    if output_path is None:
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        base_dir = os.path.dirname(csv_path)
        output_path = os.path.join(base_dir, f"{filename}_heatmap.jpg")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    print(f"Success! Heatmap saved to: {output_path}")

    # 10. Display the result
    if show:
        cv2.imshow("Heatmap Result", overlay)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--video-dir", default="data", help="Directory where the video files are stored (default: data)")
@click.option("--output", default=None, help="Output image filename (optional)")
@click.option("--sigma", default=15.0, help="Blur amount / thickness of the path (default: 15)")
@click.option("--alpha", default=0.6, help="Transparency level between 0.0 and 1.0 (default: 0.6)")
@click.option("--show", is_flag=True, help="Display the image after generation")
def cli(csv_path, video_dir, output, sigma, alpha, show):
    """
    Generates a color heatmap from the trajectory CSV data and overlays it on the original video's first frame.
    The script automatically finds the corresponding video file.
    """
    generate_heatmap(csv_path, output_path=output, sigma=sigma, alpha=alpha, show=show, video_dir=video_dir)

if __name__ == '__main__':
    cli()