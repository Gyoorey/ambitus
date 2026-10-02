# Ambitus Analysis CLI

Top-down infrared video tracking and behavioural feature extraction for rats in the Ambitus arena.

## Setup

1. Clone the repository  
   `git clone <repository-url>`

2. Navigate into the project directory  
   `cd <repository-name>`

3. Create a virtual environment  
   `python3 -m venv venv`

4. Activate the virtual environment  
   `source venv/bin/activate`

5. Install the package in editable mode  
   `pip install -e .`  
   *(Only the editable install is supported. Keep the cloned folder in place, because the `ambitus` command runs the code from it.)*

6. Add data files  
   Paste the shared image and video files into:  
   `ambitus/data/`

7. Reference files  
   `align_roi`, `run` and `run_dir` read `base_reference.jpg` and `arena_config_poly.json` from the **current folder**. Run the commands from the folder that contains these two files, or pass them explicitly with `--base-image` and `--base-config`. The examples below are run from inside `ambitus/`.

## Video Conversion

Converting `.mpg` files to `.mp4` before processing is optional but recommended for speed.

Example:  
`ffmpeg -i data/LE_17_1_45.mpg -vf "yadif,scale=768:576,setsar=1" -pix_fmt yuv420p data/45.mp4`

*(Note: The software also directly supports `.mpg` format via `imageio`, automatically applying the `yadif,scale=768:576,setsar=1` de-interlacing parameters.)*

---

## Usage Guide & Commands

### 1. Basic Testing
Run a basic test command to verify everything works by displaying the first frame:
`ambitus load-video data/45.mp4 --show`

### 2. ROI Alignment
Calculate the camera offset (shift and rotation, -10° to +10°) and transform the Region of Interest (ROI) polygons for a new video, based on the reference image (`base_reference.jpg`). The result is saved as `results/45/45_config.json`. Use `--debug` to see the aligned ROI.
`ambitus align_roi data/45.mp4 --debug`

*(You can manually define the ROI polygons via a GUI if needed: `ambitus roi data/45.mp4 --output my_config.json`. Without `--output` it overwrites `arena_config_poly.json`, the base config used by every alignment.)*

*Known limitation: upward camera shifts are not detected, because the matching template touches the top of the image.*

### 3. Tracking (Individual)
Track the rat using fast methods: `static` (Masked Running Average) or `simple` (MOG2).
`ambitus analyze data/45.mp4 --show --skip-seconds 0.0 --debug --method static --bg-method zone --config results/45/45_config.json`

Background methods for the `static` tracker:
*   **`--bg-method quiet`**: (Default) Finds the most stable 5-second window in the first 45 s and uses its median as background.
*   **`--bg-method global`**: Median of 150 frames spread over the entire video. Fails when the rat sits in one place for most of the video (the rat becomes part of the background).
*   **`--bg-method zone`** (recommended): The rat is always dropped in the top-right corner, so the other three quarters of the arena are empty at the start. They are taken from the start of the video (until the rat first enters them, at most 40 s). The top-right quarter (hand, food stick, rat drop) is taken from the first frames in which the rat is clearly in another quarter. The two parts are blended at the border. If the rat never leaves the top-right quarter, that quarter is taken from the first 2 s of the video and a warning is printed.

The background used is saved as `<video>_DEBUG_BACKGROUND.jpg` in the result folder; check it for a rat or a hand. The statistics file records the background method (and for `zone`, which time windows were used).

### 4. Trajectory Filtering
Crops the trajectory to a 300-second window, removes short tracking excursions and teleportation spikes, interpolates missing frames (rejecting gaps whose straight line would cross the inner wall), and applies a median + Savitzky-Golay low-pass filter.
`ambitus filter results/45/45_static_trajectory.csv --config results/45/45_config.json --plot`

*   `--config`: Use the **aligned** config of the video. Without it, the unaligned `arena_config_poly.json` is used for the wall checks.
*   `--plot`: Opens the comparison dashboard in a window. The dashboard file is always saved (`_plot.jpg`).
*   `--plotly`: Saves the dashboard as interactive HTML (`_plotly.html`) instead of JPG.
*   `--no-excursion-filter`: Keeps short tracking excursions (a jump of more than 25 px in one frame that returns within 15 frames). By default they are removed and interpolated.
*   Window start: if the CSV starts later than 1 s (`--skip-seconds` or `--backtrack`), the window starts at the first row; otherwise the rat insertion is detected automatically and 2 s are added.

Filtering results (window, excursions, spikes, rejected gaps) are appended to `<video>_static_statistics.txt`. Do not re-run `filter`/`filter_dir` on finished results without re-tracking: the statistics are appended again and `check` will count the rejected gaps twice.

### 5. Heatmap Generation
Generate a color density heatmap from the filtered CSV and overlay it on the original video's first frame:
`ambitus heatmap results/45/45_static_trajectory_filtered.csv --show`

### 6. Activity Classification & Feature Extraction
Classify every frame as High Active (2), Low Active (1) or Inactive (0):
`ambitus activity results/45/45_static_trajectory_filtered.csv --debug`

*   **High Active** is decided by speed only: speed over a centred 5-frame window above `--speed` (default **35 px/s**).
*   **Low Active vs Inactive** is decided by local pixel change: number of changed pixels in a box around the rat (`--box-size`, default 80) between the current frame and the frame 5 frames earlier, above `--sad` (default 20) = Low Active.
*   Video frames are matched to the CSV rows by timestamp, so rows missing after rejected gaps do not shift the video.

Extract the behavioural, spatial and kinematic features into a JSON file:
`ambitus extract results/45/45_static_trajectory_filtered_classified.csv --debug`

The config `results/45/45_config.json` is detected automatically. Feature groups:
*   **I. General locomotor**: total distance, High/Low-Active distance and time, inactivity time, average speeds, number and mean duration of activity/inactivity phases. States are smoothed by a 1-second majority vote; all time, distance and phase features use these smoothed states.
*   **II. Phase specific**: short (≤ 5 s) and long (> 5 s) activity/inactivity phases: count, time, distance, speed.
*   **III. Spatial and geometric**: corner time and ratio, clockwise/counter-clockwise distance and asymmetry index, U-turn count, spatial entropy.
*   **IV. Kinematic**: meander (tortuosity) and acceleration variance.
*   **V. Habituation**: per-minute statistics for minutes 1–5 and the habituation index.

Notes:
*   **CW/CCW** is decided by the sideways (tangential) movement around the arena centre over 5 frames (at least 2 px), counted during High Active only.
*   **U-turns** are counted from clear-movement headings: a U-turn is a change of heading of at least 120° between runs of clear movement (≥ 2 px/frame). In-place turns are counted once the rat moves off in the opposite direction. The turn times are written to `<name>_features_uturns.txt` as intervals (the turn happened inside the interval); the count is more reliable than the exact time.
*   **Meander** and **acceleration variance** are dominated by the whole-pixel coordinates (1-px flicker of a resting rat) and should not be interpreted as behavioural measures yet.
*   Every extraction checks the invariants (time conservation, phase counts, 300 s window) and prints a validation report.

Output files: `<name>_features.json`, `<name>_features_debug.csv` (per-frame speed, state, direction, corner flag), `<name>_features_uturns.txt`, and with `--debug` an HTML dashboard (and a sector image if `--video` is given).

### 7. Playback & Visualization
Run a playback on the filtered results. It dynamically draws the rat's state, fading comet tail, and directional vectors on the original video:
`ambitus playback results/resulted_csv_folder --video-dir videos`
*(Use `--save` to record the playback to an MP4 file).*

---

## Fully Automated Pipelines

### Run Pipeline on a Single Video
Executes the entire workflow (Align ROI -> Track -> Filter -> Heatmap -> Classify -> Extract Features) on a single video.

`ambitus run data/45.mp4 --backtrack --bg-method zone`

**Advanced Run Options:**
*   `--skip-seconds 40`: Skip the first 40 seconds manually (e.g., to ignore the researcher's hand).
*   `--bg-method zone|global|quiet`: Background method of the static tracker (see Tracking).
*   `--backtrack`: Finds the light-on event in the last 120 s of the video and tracks the 300 seconds before it (ending 1 s before the light-on). Forces the `global` background, except with `--bg-method zone`, which is kept (the zone background then only uses frames before the light-on). If no light-on is found (or it is less than 330 s into the video), tracking starts at 0 s, the rat insertion is detected automatically and `quiet` (or `zone`) is used.
*   `--no-excursion-filter`: Keep short tracking excursions (see Filtering).
*   `--min-area`: Minimum blob area in pixels (default 100).
*   `--out-dir`: Write the results to this folder.
*   `--export`: Updates the Master Excel with the newly extracted features.
*   `--show` / `--debug`: Display results and debug views.

*Two videos have a fixed start time hard-coded in `tracker.py` (inconsistent light source), which overrides `--skip-seconds` and `--backtrack`: `LI_G19_N2_D3_T1` (33 s) and `LI_G17_N33_D1_T1` (48 s).*

### Run Pipeline on a Directory
Runs the complete pipeline recursively on the videos found in a directory and mirrors the folder structure into a `results/` folder next to the video directory. At the end, all feature files are exported to Excel (master and folder sheets).

`ambitus run_dir somewhere/videos_dir --backtrack --bg-method zone`

*   All `.mp4` files are processed. `.mpg` files are processed **only if the directory tree contains no `.mp4` at all**; keep one format per run.
*   Supports `--backtrack`, `--bg-method`, `--skip-seconds`, `--min-area`, `--no-excursion-filter`, the filter/heatmap options, `--show` and `--debug`. `--export` and `--out-dir` are not available (the export always runs).

### Output Files per Video
`results/<subfolders>/<video>/`:
*   `<video>_config.json` – aligned ROI
*   `<video>_DEBUG_BACKGROUND.jpg` – background used by the static tracker
*   `<video>_static_trajectory.csv`, `_trajectory_map.jpg`, `_statistics.txt` – raw tracking
*   `<video>_static_trajectory_filtered.csv` – filtered trajectory (300 s), plus the filter dashboard and heatmap
*   `<video>_static_trajectory_filtered_classified.csv` – filtered trajectory with activity state
*   `<video>_static_features.json`, `_features_debug.csv`, `_features_uturns.txt` – features

---

## Data Aggregation & Quality Control

### Export to Excel
Aggregates all extracted feature JSON files into flat Excel spreadsheets.
`ambitus export results/`
*   Always creates an `Ambitus_Master_Features.xlsx` in the root of the results directory.
*   Use `--all-excels` to additionally create individual summary sheets for every subfolder and single video.

### Audit Results (Quality Check)
Scans all processed results and lists videos that need a closer look:
`ambitus check results/`
*   **Too short**: less than 280 s of evaluated time.
*   **Tracking error**: more than 5 rejected wall-crossing gaps.
*   **Escaped/lost**: more than 15% missed frames.
*   **Early start**: the 300 s window starts before 20 s (e.g. the light-on was not found and the hand was detected as the rat). Re-run these with `--skip-seconds`.
*   Generates a human-readable `Problematic_Videos_Report.txt` in the results directory.

---

## Validation

`generate.py` is a synthetic test suite for the filtering, activity, feature extraction and tracking steps. It generates rat-like paths (gradual turns, noise, rests) and synthetic videos with known ground truth and compares the pipeline output with it.

`python generate.py` (full suite, about 5 minutes)
*   `--only test6`: run only tests whose name contains the text (e.g. `test8` activity frame sync, `test9` zone background).
*   `--no-noise`: skip the noise variants; `--seeds 5`: more noise seeds; `--html`: write the debug dashboards.

Test files are written to `./synthetic_test_suite`. The exit code is 1 if any check fails.

---

*Available legacy tracking methods for `analyze-exp`: dense optical flow with Farneback (dof), simple+Kalman-Filter (kf), simple+Optical-flow (of), pure optical flow (pure_of), Motion History Image (mhi).*