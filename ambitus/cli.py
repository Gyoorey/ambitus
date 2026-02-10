import click
import cv2
from process_video import process_video
from matplotlib import pyplot as plt


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
@click.option("--show", is_flag=True, help="Detect the rat and show trajectory on the loaded video")
@click.option("--debug", is_flag=True, help="Show debug view")
@click.option("--skip-seconds", default=0.0, help="Seconds to skip at the start")
@click.option("--min-area", default=200, help="Minimum area")
@click.option("--method", type=click.Choice(['simple', 'dof', 'kf', 'of'], case_sensitive=False), default='simple', help="Tracking method: Kalman Filter (kf) or Optical Flow (of)")
def analyze(path, show, debug, skip_seconds, min_area, method):
    """ Process the video to detect and track rat trajectories."""
    process_video(
        video_path=path,
        method=method,
        show_traj=show,
        debug=debug,
        skip_seconds=skip_seconds,
        min_area=min_area,
    )