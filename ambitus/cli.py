import click
import cv2
from process_video import process_video as process_video_legacy
from matplotlib import pyplot as plt
from roi_selector import select_roi
from tracker import track
from align_roi import align_video_roi
from heatmap_gen import generate_heatmap
from traj_filter import process_trajectory
import os
from pathlib import Path


@click.group()
def cli():
    pass


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--show", is_flag=True, help="Show the loaded image")
def load_video(video_path, show):
    """Load a video and display the first frame."""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        click.echo("Failed to open video.")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        click.echo("Failed to load video.")
        return

    # Safe conversion
    if frame.ndim == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    if show:
        plt.imshow(gray_frame, cmap="gray")
        plt.title("First Frame")
        plt.axis("off")
        plt.show()

    click.echo("Video loaded successfully.")


@cli.command(name="analyze")
@click.argument("path", type=click.Path(exists=True))
@click.option("--config", default="arena_config_poly.json", help="Path to ROI config JSON file")
@click.option("--show", is_flag=True, help="Detect the rat and show trajectory on the loaded video")
@click.option("--debug", is_flag=True, help="Show debug view")
@click.option("--skip-seconds", default=0.0, help="Seconds to skip at the start")
@click.option("--min-area", default=100, help="Minimum area")
@click.option("--method", type=click.Choice(['simple', 'static'], case_sensitive=False), default='simple', help="Tracking method: simple (MOG2) or static (Masked Running Average).")
def analyze(path, config, show, debug, skip_seconds, min_area, method):
    """ Process video using the optimized, clean tracking methods. """
    track(
        video_path=path,
        config_file=config,
        method=method,
        show_traj=show,
        debug=debug,
        skip_seconds=skip_seconds,
        min_area=min_area,
    )

@cli.command(name="analyze-exp")
@click.argument("path", type=click.Path(exists=True))
@click.option("--config", default="arena_config_poly.json", help="Path to ROI config JSON file")
@click.option("--show", is_flag=True, help="Detect the rat and show trajectory on the loaded video")
@click.option("--debug", is_flag=True, help="Show debug view")
@click.option("--skip-seconds", default=0.0, help="Seconds to skip at the start")
@click.option("--min-area", default=100, help="Minimum area")
@click.option("--method", type=click.Choice(['static','simple', 'dof', 'kf', 'of', 'sub_dof_fusion', 'pure_of', 'mhi'], case_sensitive=False), default='simple', help="Tracking method: static(Median Backround Sub), simple (MOG2 Background Subtractor), simple + Kalman Filter (kf), simple + Optical Flow (of), dense optical flow (dof), fusion of MOG2 and DOF (sub_dof_fusion), pure sparse optical flow (pure_of), or motion history image (mhi).")
def analyze(path, config, show, debug, skip_seconds, min_area, method):
    """ Process the video to detect and track rat trajectories."""
    process_video_legacy(
        video_path=path,
        config_file=config,
        method=method,
        show_traj=show,
        debug=debug,
        skip_seconds=skip_seconds,
        min_area=min_area,
    )

@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output", default="arena_config_poly.json", help="Output config file path")
def roi(video_path, output):
    """
    Open GUI to define the Region of Interest (ROI) polygons.
    """
    select_roi(video_path, config_file=output)

@cli.command(name="align_roi")
@click.argument("new_video", type=click.Path(exists=True))
@click.option("--base-image", default="base_reference.jpg", type=click.Path(exists=True), help="The original reference image (first frame of the base video)")
@click.option("--base-config", default="arena_config_poly.json", help="The original ROI JSON file")
@click.option("--debug", is_flag=True, help="Display the visual result")
def align_roi_cmd(new_video, base_image, base_config, debug):
    """
    Calculate the camera offset and rotate the ROI mask for the new video.
    Creates a [video_name]_config.json file as output.
    """
    print("DBG: Starting ROI alignment...")
    align_video_roi(new_video, base_image, base_config, debug)

@cli.command(name="heatmap")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--video-dir", default="data", help="Directory where the video files are stored (default: data)")
@click.option("--output", default=None, help="Output image filename (optional)")
@click.option("--sigma", default=15.0, help="Blur amount / thickness of the path (default: 15)")
@click.option("--alpha", default=0.6, help="Transparency level between 0.0 and 1.0 (default: 0.6)")
@click.option("--show", is_flag=True, help="Display the image after generation")
def heatmap_cmd(csv_path, video_dir, output, sigma, alpha, show):
    """
    Generates a color heatmap from the trajectory CSV data and overlays it on the original video's first frame.
    The script automatically finds the corresponding video file.
    """
    generate_heatmap(csv_path, output_path=output, sigma=sigma, alpha=alpha, show=show, video_dir=video_dir)

@cli.command(name="filter")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--output", default=None, help="Output CSV filename (optional)")
@click.option("--median-window", default=5, help="Window size for Outlier Removal (default: 5)")
@click.option("--savgol-window", default=31, help="Window size for Low-Pass Filtering (default: 31)")
@click.option("--plot", is_flag=True, help="Show the comparison graph")
@click.option("--plotly", is_flag=True, help="Use interactive Plotly graph instead of OpenCV")
def filter_cmd(csv_path, output, median_window, savgol_window, plot, plotly):
    """
    Interpolates missing frames, removes tracking outliers, and applies a low-pass filter to the trajectory CSV.
    Outputs integer coordinates.
    """
    process_trajectory(
        csv_path, 
        output_path=output, 
        median_window=median_window, 
        savgol_window=savgol_window, 
        show_plot=plot,
        use_plotly=plotly
    )

@cli.command(name="run")
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--base-image", default="base_reference.jpg", type=click.Path(exists=True), help="The original reference image")
@click.option("--base-config", default="arena_config_poly.json", help="The original ROI JSON file")
@click.option("--method", type=click.Choice(['simple', 'static'], case_sensitive=False), default='static', help="Tracking method")
@click.option("--min-area", default=100, help="Minimum area")
@click.option("--skip-seconds", default=0.0, help="Seconds to skip at the start")
@click.option("--median-window", default=5, help="Window size for outlier removal")
@click.option("--savgol-window", default=31, help="Window size for low-pass filtering")
@click.option("--sigma", default=15.0, help="Blur amount for heatmap")
@click.option("--alpha", default=0.6, help="Transparency level for heatmap")
@click.option("--show", is_flag=True, help="Display results")
@click.option("--debug", is_flag=True, help="Show debug views")
def run_pipeline(video_path, base_image, base_config, method, min_area, skip_seconds, median_window, savgol_window, sigma, alpha, show, debug):
    """Run the complete pipeline: align ROI, track, filter, and generate heatmap."""

    # Example: ambitus run data/LE_17_1_45.mp4 --method static --show --debug
    # Options: --skip-seconds 0 --median-window 5 --savgol-window 31 --sigma 15 --alpha 0.6 
    
    video_name = Path(video_path).stem
    result_dir = os.path.join("results", video_name)
    os.makedirs(result_dir, exist_ok=True)
    aligned_config = os.path.join(result_dir, f"{video_name}_config.json")
    csv_file = os.path.join(result_dir, f"{video_name}_{method}_trajectory.csv")
    filtered_csv = os.path.join(result_dir, f"{video_name}_{method}_trajectory_filtered.csv")
    heatmap_file = os.path.join(result_dir, f"{video_name}_{method}_heatmap.jpg")
    
    click.echo("Step 1: Aligning ROI...")
    align_video_roi(video_path, base_image, base_config, debug)
    
    click.echo("Step 2: Tracking...")
    track(video_path, aligned_config, method, show, debug, skip_seconds, min_area)
    
    click.echo("Step 3: Filtering trajectory...")
    process_trajectory(csv_file, None, median_window, savgol_window, show, True)
    
    click.echo("Step 4: Generating heatmap...")
    generate_heatmap(filtered_csv, None, sigma, alpha, show, "data")
    
    click.echo(f"Pipeline complete! Results saved to {result_dir}")

@cli.command(name="run_dir")
@click.argument("video_dir", type=click.Path(exists=True))
@click.option("--base-image", default="base_reference.jpg", type=click.Path(exists=True), help="The original reference image")
@click.option("--base-config", default="arena_config_poly.json", help="The original ROI JSON file")
@click.option("--method", type=click.Choice(['simple', 'static'], case_sensitive=False), default='static', help="Tracking method")
@click.option("--min-area", default=100, help="Minimum area")
@click.option("--skip-seconds", default=0.0, help="Seconds to skip at the start")
@click.option("--median-window", default=5, help="Window size for outlier removal")
@click.option("--savgol-window", default=31, help="Window size for low-pass filtering")
@click.option("--sigma", default=15.0, help="Blur amount for heatmap")
@click.option("--alpha", default=0.6, help="Transparency level for heatmap")
@click.option("--show", is_flag=True, help="Display results")
@click.option("--debug", is_flag=True, help="Show debug views")
def run_dir_pipeline(video_dir, base_image, base_config, method, min_area, skip_seconds, median_window, savgol_window, sigma, alpha, show, debug):
    """Run the complete pipeline on all videos in a directory."""
    video_extensions = ['.mp4']
    video_files = [f for f in Path(video_dir).iterdir() if f.suffix.lower() in video_extensions]
    
    if not video_files:
        click.echo(f"No video files found in {video_dir}")
        return
    
    for video_path in sorted(video_files):
        click.echo(f"\nProcessing: {video_path.name}")
        run_pipeline.callback(str(video_path), base_image, base_config, method, min_area, skip_seconds, median_window, savgol_window, sigma, alpha, show, debug)