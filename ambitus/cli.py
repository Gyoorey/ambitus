import click
import cv2
from process_video import process_video as process_video_legacy
from matplotlib import pyplot as plt
from roi_selector import select_roi
from tracker import track
from align_roi import align_video_roi
from heatmap_gen import generate_heatmap
from traj_filter import process_trajectory
from playback_filtered import play_filtered_trajectory
from feature_extractor import extract_features
from threshold_calibrator import calibrate_global_thresholds
from activity import classify_activities
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
@click.option("--method", type=click.Choice(['simple', 'static'], case_sensitive=False), default='static', help="Tracking method: simple (MOG2) or static (Masked Running Average).")
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
@click.option("--config", default="arena_config_poly.json", help="Path to ROI config JSON file")
def filter_cmd(csv_path, config, output, median_window, savgol_window, plot, plotly):
    """
    Interpolates missing frames, removes tracking outliers, and applies a low-pass filter to the trajectory CSV.
    Outputs integer coordinates.
    """
    process_trajectory(
        csv_path,
        config_path=config, 
        output_path=output, 
        median_window=median_window, 
        savgol_window=savgol_window, 
        show_plot=plot,
        use_plotly=plotly
    )


@cli.command(name="filter_dir")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--output-dir", default=None, help="Output directory for filtered CSVs (optional)")
@click.option("--median-window", default=5, help="Window size for Outlier Removal (default: 5)")
@click.option("--savgol-window", default=31, help="Window size for Low-Pass Filtering (default: 31)")
@click.option("--plot", is_flag=True, help="Show the comparison graph")
@click.option("--plotly", is_flag=True, help="Use interactive Plotly graph instead of OpenCV")
@click.option("--heatmap", is_flag=True, help="Generate heatmap for filtered trajectories")
@click.option("--config", default="arena_config_poly.json", help="Path to ROI config JSON file")
def filter_dir_cmd(results_dir, output_dir, median_window, savgol_window, plot, plotly, heatmap, config):
    """Apply trajectory filtering to all CSV files in results subdirectories."""
    root_dir = Path(results_dir).resolve()
    csv_files = list(root_dir.rglob('*_trajectory.csv'))
    
    if not csv_files:
        click.echo(f"No trajectory CSV files found in {results_dir}.")
        return
    
    click.echo(f"Found {len(csv_files)} CSV files.")
    
    for csv_file in sorted(csv_files):
        click.echo(f"\nFiltering: {csv_file.name}")
        video_name = csv_file.parent.name
        base_dir = os.path.dirname(str(csv_file))
        output_path = os.path.join(base_dir, f"{csv_file.stem}_filtered.csv")
        aligned_config = os.path.join(base_dir, f"{video_name}_config.json")
        process_trajectory(
            str(csv_file),
            config_path=aligned_config,
            output_path=output_path,
            median_window=median_window,
            savgol_window=savgol_window,
            show_plot=plot,
            use_plotly=plotly
        )
        
        if heatmap:
            click.echo(f"Generating heatmap for: {csv_file.name}")
            heatmap_output = os.path.join(os.path.dirname(output_path), f"{csv_file.stem}_filtered_heatmap.jpg")
            # Extract video name by removing method suffix and trajectory info
            csv_name = csv_file.stem  # e.g., "video_name_simple_trajectory"
            video_name = None
            for method_suffix in ['_simple', '_static', '_dof', '_kf', '_of', '_sub_dof_fusion', '_pure_of', '_mhi']:
                if csv_name.endswith(method_suffix + '_trajectory'):
                    video_name = csv_name[:-len(method_suffix + '_trajectory')]
                    break
            
            if video_name:
                # Search for the video file in root folder and all subfolders
                video_files = list(root_dir.parent.rglob(f'{video_name}.mp4')) + list(root_dir.parent.rglob(f'{video_name}.mpg'))
                if video_files:
                    video_dir = str(video_files[0].parent)
                    generate_heatmap(output_path, output_path=heatmap_output, sigma=15.0, alpha=0.6, show=False, video_dir=video_dir)
                else:
                    click.echo(f"Warning: Video file for {video_name} not found.")
    
    click.echo(f"\nFiltering complete! Results saved to {output_dir}")


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
@click.option("--out-dir", default=None, help="Output directory override")
def run_pipeline(video_path, base_image, base_config, method, min_area, skip_seconds, median_window, savgol_window, sigma, alpha, show, debug, out_dir):
    """Run the complete pipeline: align ROI, track, filter, and generate heatmap."""

    # Example: ambitus run data/LE_17_1_45.mp4 --method static --show --debug
    # Options: --skip-seconds 0 --median-window 5 --savgol-window 31 --sigma 15 --alpha 0.6 
    
    v_path = Path(video_path).resolve()
    video_name = v_path.stem
    if out_dir is None:
        result_dir = str(v_path.parent.parent / "results" / video_name)
    else:
        result_dir = out_dir
    os.makedirs(result_dir, exist_ok=True)
    aligned_config = os.path.join(result_dir, f"{video_name}_config.json")
    csv_file = os.path.join(result_dir, f"{video_name}_{method}_trajectory.csv")
    filtered_csv = os.path.join(result_dir, f"{video_name}_{method}_trajectory_filtered.csv")
    
    click.echo("Step 1: Aligning ROI...")
    align_video_roi(video_path, base_image, base_config, debug, output_dir=result_dir)
    
    click.echo("Step 2: Tracking...")
    track(video_path, aligned_config, method, show, debug, skip_seconds, min_area, output_dir=result_dir)
    
    click.echo("Step 3: Filtering trajectory...")
    process_trajectory(csv_file, aligned_config, None, median_window, savgol_window, show, True)
    
    click.echo("Step 4: Generating heatmap...")
    generate_heatmap(filtered_csv, None, sigma, alpha, show, video_dir=str(v_path.parent))

    click.echo("Step 5: Classifying activity...")
    classified_csv = os.path.join(result_dir, f"{video_name}_{method}_trajectory_filtered_classified.csv")
    classify_activities(filtered_csv, output_path=classified_csv, debug=debug)

    click.echo("Step 6: Extracting features...")
    features_json = os.path.join(result_dir, f"{video_name}_{method}_features.json")
    extract_features(classified_csv, aligned_config, features_json, debug_html=debug, video_path=video_path)

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


    root_data_dir = Path(video_dir).resolve()
    results_root_dir = root_data_dir.parent / "results"
    video_files = list(root_data_dir.rglob('*.mp4'))

    if not video_files:
        click.echo(f"No .mp4 files found in {video_dir}. Searching for .mpg files...")
        video_files = list(root_data_dir.rglob('*.mpg'))
    
    if not video_files:
        click.echo(f"No .mp4 or .mpg files found in {video_dir} or its subdirectories.")
        return
    
    click.echo(f"Found {len(video_files)} videos.")

    for video_path in sorted(video_files):
        click.echo(f"\nProcessing: {video_path.name}")
        rel_path = video_path.parent.relative_to(root_data_dir)
        target_result_dir = str(results_root_dir / rel_path / video_path.stem)
        run_pipeline.callback(str(video_path), base_image, base_config, method, min_area, skip_seconds,
                               median_window, savgol_window, sigma, alpha, show, debug, target_result_dir)


@cli.command(name="playback")
@click.argument("csv_dir", type=click.Path(exists=True))
@click.option("--video-dir", default="data", help="Directory where the video files are stored (default: data)")
@click.option("--video-file", default=None, help="Direct path to the video file (bypasses directory search)")
@click.option("--method", type=click.Choice(['simple', 'static'], case_sensitive=False), default='static', help="Tracking method")
@click.option("--tail", default=3.0, help="Length of the visible trajectory tail in seconds (default: 3.0)")
@click.option("--speed", default=1.0, help="Playback speed multiplier (e.g., 2.0 for double speed)")
@click.option("--save", is_flag=True, help="Save the playback with overlaid trajectory as a new video file")
def playback_cmd(csv_dir, video_dir, video_file, method, tail, speed, save):
    """
    Plays the original video and dynamically overlays the filtered trajectory 
    with a fading tail to visualize the rat's movement.
    """
    csv_first_part = Path(csv_dir).name
    csv_path = os.path.join(csv_dir, f"{csv_first_part}_{method}_trajectory_filtered.csv")
    play_filtered_trajectory(csv_path, tail_length_sec=tail, video_file=video_file, speed=speed, save_video=save)

@cli.command(name="calibrate")
@click.argument("results_dir", type=click.Path())
@click.option("--output", default="global_thresholds.json", help="The name of the output JSON file that will store the global thresholds (default: global_thresholds.json)")
def calibrate_cmd(results_dir, output):
    """
    Calculates population-level global speed thresholds for Active/Inactive classification.
    """
    calibrate_global_thresholds(results_dir, output)

@cli.command(name="activity")
@click.argument("csv_path", type=click.Path())
@click.option("--output", default=None, help="Output CSV filename (optional)")
@click.option("--box-size", default=120, help="Size of the box around the current position to calculate SAD (default: 80)")
@click.option("--sad", default=20, help="SAD threshold for classifying Low Active vs Inactive (default:100)")
@click.option("--speed", default=None, help="Speed threshold (px/s) for classifying High Active vs Low Active (default: None - global threshold)")
@click.option("--debug", is_flag=True, help="show debug window")
def activity_cmd(csv_path, output, box_size, sad, speed, debug):
    """
    Classifies each frame into High Active, Low Active, or Inactive based on speed and local pixel changes.
    Outputs a new CSV with the activity state for each frame.
    """
    classify_activities(csv_path, output_path=output, box_size=box_size, sad_threshold=sad, high_active_speed=speed, debug=debug)


@cli.command(name="extract")
@click.argument("csv_path", type=click.Path())
@click.option("--config", default=None, type=click.Path(), help="Path to the aligned ROI config JSON file (Optional, auto-detected if not provided)")
@click.option("--video", default=None, type=click.Path(), help="Path to original video file for generating the sector debug image (Optional)")
@click.option("--output", default=None, help="Output JSON filename (Optional)")
@click.option("--debug", is_flag=True, help="Generate debug HTML dashboard")
def extract_cmd(csv_path, config, video, output, debug):
    """
    Extracts behavioral and spatial features from a filtered trajectory CSV.
    Uses adaptive thresholds to classify active/inactive phases.
    Saves the extracted features to a human-readable JSON file.
    """
    if not os.path.exists(csv_path):
        click.echo(f"ERROR: CSV file not found: {csv_path}")
        return

    base_dir = os.path.dirname(csv_path)
    filename = os.path.splitext(os.path.basename(csv_path))[0]

    base_name = filename
    if base_name.endswith('_trajectory_filtered_classified'):
        base_name = base_name.replace('_trajectory_filtered_classified', '')
    elif base_name.endswith('_trajectory_filtered'):
        base_name = base_name.replace('_trajectory_filtered', '')
    elif base_name.endswith('_filtered'):
        base_name = base_name.replace('_filtered', '')
    elif base_name.endswith('_trajectory'):
        base_name = base_name.replace('_trajectory', '')
        
    methods = ['_simple', '_static', '_dof', '_kf', '_of', '_sub_dof_fusion', '_pure_of', '_mhi']
    for method in methods:
        if base_name.endswith(method):
            base_name = base_name[:-len(method)]
            break

    if config is None:
        config = os.path.join(base_dir, f"{base_name}_config.json")
        
        if not os.path.exists(config):
            click.echo(f"ERROR: Auto-detected Config file not found: {config}")
            click.echo("Please provide the config file manually using the --config option.")
            return
        else:
            click.echo(f"Auto-detected config file: {config}")
    else:
        if not os.path.exists(config):
            click.echo(f"ERROR: Provided Config file not found: {config}")
            return

    if output is None:
        method_suffix = filename.replace(base_name, '').replace('_trajectory_filtered_classified', '').replace('_trajectory_filtered', '').replace('_trajectory', '')
        output = os.path.join(base_dir, f"{base_name}{method_suffix}_features.json")

    extract_features(csv_path, config_path=config, output_json_path=output, debug_html=debug, video_path=video)

@cli.command(name="extract_dir")
@click.argument("results_dir", type=click.Path())
@click.option("--force", is_flag=True, help="Force overwrite existing feature JSON files")
@click.option("--debug", is_flag=True, help="Generate debug HTML dashboards")
def extract_dir_cmd(results_dir, force, debug):
    """
    Recursively finds all trajectory CSVs in the results directory and 
    extracts features into JSON files next to them.
    Automatically detects the corresponding config files and applies global thresholds if available.
    """
    root_dir = Path(results_dir).resolve()
    
    if not root_dir.exists():
        click.echo(f"ERROR: Results directory not found: {root_dir}")
        return
        
    csv_files = list(root_dir.rglob('*_trajectory_filtered_classified.csv'))
    
    if not csv_files:
        csv_files = list(root_dir.rglob('*_trajectory.csv'))
        
    if not csv_files:
        click.echo(f"No trajectory CSV files found in {root_dir} or its subdirectories.")
        return
        
    click.echo(f"Found {len(csv_files)} CSV files for feature extraction.")
    
    global_thresh_file = root_dir / "global_thresholds.json"
    global_path = str(global_thresh_file) if global_thresh_file.exists() else None
    
    if global_path:
        click.echo(f"Found global thresholds: {global_path}")
    else:
        click.echo("No global thresholds found. Using local per-video calibration.")

    success_count = 0

    for csv_file in sorted(csv_files):
        base_dir = csv_file.parent
        filename = csv_file.stem
        
        base_name = filename
        if base_name.endswith('_trajectory_filtered_classified'):
            base_name = base_name.replace('_trajectory_filtered_classified', '')
        elif base_name.endswith('_trajectory_filtered'):
            base_name = base_name.replace('_trajectory_filtered', '')
        elif base_name.endswith('_filtered'):
            base_name = base_name.replace('_filtered', '')
        elif base_name.endswith('_trajectory'):
            base_name = base_name.replace('_trajectory', '')
            
        methods = ['_simple', '_static', '_dof', '_kf', '_of', '_sub_dof_fusion', '_pure_of', '_mhi']
        method_suffix = ""
        for method in methods:
            if base_name.endswith(method):
                method_suffix = method
                base_name = base_name[:-len(method)]
                break

        config_path = base_dir / f"{base_name}_config.json"
        output_json = base_dir / f"{base_name}{method_suffix}_features.json"

        if output_json.exists() and not force:
            click.echo(f"Skipping {csv_file.name} (Features already extracted. Use --force to overwrite).")
            continue

        if not config_path.exists():
            click.echo(f"WARNING: Skipping {csv_file.name} (Missing config file: {config_path.name})")
            continue
            
        click.echo(f"\nExtracting features for: {csv_file.name}")
        
        try:
            extract_features(str(csv_file), config_path=str(config_path), output_json_path=str(output_json), debug_html=debug)
            success_count += 1
        except Exception as e:
            click.echo(f"ERROR during extraction of {csv_file.name}: {e}")
            
    click.echo(f"\n========================================")
    click.echo(f"Extraction complete! Successfully processed {success_count}/{len(csv_files)} files.")