import click
import cv2
from process_video import process_video as process_video_legacy
from matplotlib import pyplot as plt
from roi_selector import select_roi
from tracker import track
from align_roi import align_video_roi
from heatmap_gen import generate_heatmap
from traj_filter import process_trajectory


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
@click.option("--min-area", default=200, help="Minimum area")
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
@click.option("--min-area", default=200, help="Minimum area")
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
def filter_cmd(csv_path, output, median_window, savgol_window, plot):
    """
    Interpolates missing frames, removes tracking outliers, and applies a low-pass filter to the trajectory CSV.
    Outputs integer coordinates.
    """
    process_trajectory(
        csv_path, 
        output_path=output, 
        median_window=median_window, 
        savgol_window=savgol_window, 
        show_plot=plot
    )
