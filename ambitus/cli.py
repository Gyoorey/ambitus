import click
import cv2
from process_video import process_video
from matplotlib import pyplot as plt
from roi_selector import select_roi


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


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--config", default="arena_config_poly.json", help="Path to ROI config JSON file")
@click.option("--show", is_flag=True, help="Detect the rat and show trajectory on the loaded video")
@click.option("--debug", is_flag=True, help="Show debug view")
@click.option("--skip-seconds", default=0.0, help="Seconds to skip at the start")
@click.option("--min-area", default=200, help="Minimum area")
@click.option("--method", type=click.Choice(['simple', 'dof', 'kf', 'of', 'sub_dof_fusion', 'pure_of', 'mhi'], case_sensitive=False), default='simple', help="Tracking method: simple (Background Subtractor), simple + Kalman Filter (kf), simple + Optical Flow (of), dense optical flow (dof), fusion of MOG2 and DOF (sub_dof_fusion), pure sparse optical flow (pure_of), or motion history image (mhi).")
def analyze(path, config, show, debug, skip_seconds, min_area, method):
    """ Process the video to detect and track rat trajectories."""
    process_video(
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