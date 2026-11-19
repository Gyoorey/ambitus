# AmbitusVision

Top-down infrared video tracking and behavioral feature extraction for rats in the Ambitus arena.

AmbitusVision is an open-source Python command-line pipeline that turns overhead videos of the Ambitus, a square-corridor test apparatus for rats, into frame-by-frame trajectories, movement states and 106 per-trial movement features. It aligns the arena outline to each recording, finds the 5-min trial from the room-light switch, tracks the animal by background subtraction, removes tracking artifacts, classifies every frame as locomotion, micro-movement or rest, and writes per-trial feature files, an Excel summary and a quality report. Every stage writes its results and statistics to plain CSV, JSON or text files, so each step can be inspected and rerun on its own. Detection uses classical image processing, so no labeled training frames and no GPU are needed.

The software produced the public dataset of 709 Ambitus trials described in [Data and citation](#data-and-citation). The package and its command are named `ambitus`.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Example data](#example-data)
- [Quick start](#quick-start)
- [Usage guide and commands](#usage-guide-and-commands)
- [Fully automated pipelines](#fully-automated-pipelines)
- [Data aggregation and quality control](#data-aggregation-and-quality-control)
- [Validation](#validation)
- [Versions](#versions)
- [Data and citation](#data-and-citation)
- [License](#license)
- [Contact](#contact)

## Requirements

- Python 3.8 or later on Windows, Linux or macOS
- The Python dependencies listed in `pyproject.toml` (NumPy, OpenCV, SciPy, click, Matplotlib, scikit-image, Plotly, imageio with FFmpeg, pandas, openpyxl). They are installed automatically in the next step.
- A display is needed only for the interactive options (`--show`, `--debug`, `roi`, `playback`).

## Installation

1. Clone the repository  
   `git clone https://github.com/Gyoorey/ambitus.git`

2. Navigate into the project directory  
   `cd ambitus`

3. Create a virtual environment  
   `python3 -m venv venv`

4. Activate the virtual environment  
   `source venv/bin/activate` (Linux and macOS)  
   `venv\Scripts\activate` (Windows)

5. Install the package in editable mode  
   `pip install -e .`  
   *(Only the editable install is supported. Keep the cloned folder in place, because the `ambitus` command runs the code from it.)*

6. Check the installation  
   `ambitus --help`

## Repository structure

```
ambitus/
├── LICENSE.txt
├── README.md
├── pyproject.toml
└── src/
    └── ambitus/
        ├── cli.py                  command-line interface (the ambitus command)
        ├── align_roi.py            stage 1: arena alignment
        ├── tracker.py              stage 2: synchronization and tracking
        ├── traj_filter.py          stage 3: trajectory filtering
        ├── activity.py             stage 4: movement-state classification
        ├── feature_extractor.py    stage 5: feature extraction
        ├── export_excel.py         stage 6: export to Excel
        ├── check_results.py        stage 7: quality audit
        ├── heatmap_gen.py          occupancy heatmap
        ├── playback_filtered.py    video playback with the filtered track
        ├── roi_selector.py         manual drawing of the arena polygons
        ├── process_video.py        legacy tracking methods (analyze-exp)
        ├── generate.py             synthetic validation suite
        ├── annot.py                manual annotation of the rat center on random frames
        ├── base_reference.jpg      reference frame of the arena
        └── arena_config_poly.json  arena polygons drawn on the reference frame
```

## Example data

The videos, trajectories, movement states and features of 709 Ambitus trials are available in the SZTE Data Repository: https://doi.org/10.82570/SZTE_Datarepo_81fgh-zcr72. Every video in the dataset comes with the outputs of this software, so any of them can be used as example input and the results can be compared with the published files.

The videos are 768 × 576 px at 25 frames per second, named as `<strain>_G<generation>_N<animal>_D<day>_T<trial>`, for example `LE_G17_N3_D0_T1`.

## Quick start

The commands read `base_reference.jpg` and `arena_config_poly.json` from the **current folder**. Create a working folder, copy these two files into it from `src/ambitus/`, and put the downloaded videos into a `videos/` subfolder. All examples below are run from this working folder.

```
my_study/
├── base_reference.jpg
├── arena_config_poly.json
└── videos/
    └── LE_G17_N3_D0_T1.mp4
```

Process every video in the folder tree, then audit the results:

```
ambitus run_dir videos --backtrack --bg-method zone
ambitus check results
```

The results appear in `results/`, next to `videos/`, with one folder per video, a `summary.xlsx` with one row per video, and a `Problematic_Videos_Report.txt`.

## Video conversion

Converting `.mpg` files to `.mp4` before processing is optional but recommended for speed.

Example:  
`ffmpeg -i videos/LE_G17_N3_D0_T1.mpg -vf "yadif,scale=768:576,setsar=1" -pix_fmt yuv420p videos/LE_G17_N3_D0_T1.mp4`

*(Note: The software also directly supports `.mpg` format via `imageio`, automatically applying the `yadif,scale=768:576,setsar=1` de-interlacing parameters.)*

---

## Usage guide and commands

The examples use the video `videos/LE_G17_N3_D0_T1.mp4`. Its results are written to `results/LE_G17_N3_D0_T1/`.

### 1. Basic testing
Run a basic test command to verify everything works by displaying the first frame:  
`ambitus load-video videos/LE_G17_N3_D0_T1.mp4 --show`

### 2. ROI alignment
Calculate the camera offset (shift and rotation, -10° to +10°) and transform the Region of Interest (ROI) polygons for a new video, based on the reference image (`base_reference.jpg`). The result is saved as `results/LE_G17_N3_D0_T1/LE_G17_N3_D0_T1_config.json`. Use `--debug` to see the aligned ROI.  
`ambitus align_roi videos/LE_G17_N3_D0_T1.mp4 --debug`

*(You can manually define the ROI polygons via a GUI if needed: `ambitus roi videos/LE_G17_N3_D0_T1.mp4 --output my_config.json`. Without `--output` it overwrites `arena_config_poly.json`, the base config used by every alignment.)*

*Known limitation: upward camera shifts are not detected, because the matching template touches the top of the image.*

### 3. Tracking (individual)
Track the rat using fast methods: `static` (Masked Running Average) or `simple` (MOG2).  
`ambitus analyze videos/LE_G17_N3_D0_T1.mp4 --show --skip-seconds 0.0 --debug --method static --bg-method zone --config results/LE_G17_N3_D0_T1/LE_G17_N3_D0_T1_config.json`

Background methods for the `static` tracker:
*   **`--bg-method quiet`**: (Default) Finds the most stable 5-second window in the first 45 s and uses its median as background.
*   **`--bg-method global`**: Median of 150 frames spread over the entire video. Fails when the rat sits in one place for most of the video (the rat becomes part of the background).
*   **`--bg-method zone`** (recommended): The rat is always dropped in the top-right corner, so the other three quarters of the arena are empty at the start. They are taken from the start of the video (until the rat first enters them, at most 40 s). The top-right quarter (hand, food stick, rat drop) is taken from the first frames in which the rat is clearly in another quarter. The two parts are blended at the border. If the rat never leaves the top-right quarter, that quarter is taken from the first 2 s of the video and a warning is printed.

The background used is saved as `<video>_DEBUG_BACKGROUND.jpg` in the result folder. Check it for a rat or a hand. The statistics file records the background method (and for `zone`, which time windows were used).

### 4. Trajectory filtering
Crops the trajectory to a 300-second window, removes short tracking excursions and teleportation spikes, interpolates missing frames (rejecting gaps whose straight line would cross the inner wall), and applies a median + Savitzky-Golay low-pass filter.  
`ambitus filter results/LE_G17_N3_D0_T1/LE_G17_N3_D0_T1_static_trajectory.csv --config results/LE_G17_N3_D0_T1/LE_G17_N3_D0_T1_config.json --plot`

*   `--config`: Use the **aligned** config of the video. Without it, the unaligned `arena_config_poly.json` is used for the wall checks.
*   `--plot`: Opens the comparison dashboard in a window. The dashboard file is always saved (`_plot.jpg`).
*   `--plotly`: Saves the dashboard as interactive HTML (`_plotly.html`) instead of JPG.
*   `--no-excursion-filter`: Keeps short tracking excursions (a jump of more than 25 px in one frame that returns within 15 frames). By default they are removed and interpolated.
*   Window start: if the CSV starts later than 1 s (`--skip-seconds` or `--backtrack`), the window starts at the first row. Otherwise the rat insertion is detected automatically and 2 s are added.

Filtering results (window, excursions, spikes, rejected gaps) are appended to `<video>_static_statistics.txt`. Do not re-run `filter`/`filter_dir` on finished results without re-tracking: the statistics are appended again and `check` will count the rejected gaps twice.

### 5. Heatmap generation
Generate a color density heatmap from the filtered CSV and overlay it on the original video's first frame:  
`ambitus heatmap results/LE_G17_N3_D0_T1/LE_G17_N3_D0_T1_static_trajectory_filtered.csv --show`

### 6. Activity classification and feature extraction
Classify every frame as High Active (2), Low Active (1) or Inactive (0):  
`ambitus activity results/LE_G17_N3_D0_T1/LE_G17_N3_D0_T1_static_trajectory_filtered.csv --debug`

*   **High Active** (locomotion) is decided by speed only: speed over a centered 5-frame window above `--speed` (default **35 px/s**).
*   **Low Active vs Inactive** (micro-movement vs rest) is decided by local pixel change: number of changed pixels in a box around the rat (`--box-size`, default 80) between the current frame and the frame 5 frames earlier, above `--sad` (default 20) = Low Active.
*   Video frames are matched to the CSV rows by timestamp, so rows missing after rejected gaps do not shift the video.

Extract the behavioral, spatial and kinematic features into a JSON file:  
`ambitus extract results/LE_G17_N3_D0_T1/LE_G17_N3_D0_T1_static_trajectory_filtered_classified.csv --debug`

The config `results/LE_G17_N3_D0_T1/LE_G17_N3_D0_T1_config.json` is detected automatically. Feature groups:
*   **I. General locomotor**: total distance, High/Low-Active distance and time, inactivity time, average speeds, number and mean duration of activity/inactivity phases. States are smoothed by a 1-second majority vote. All time, distance and phase features use these smoothed states.
*   **II. Phase specific**: short (≤ 5 s) and long (> 5 s) activity/inactivity phases: count, time, distance, speed.
*   **III. Spatial and geometric**: corner time and ratio, clockwise/counter-clockwise distance and asymmetry index, U-turn count, spatial entropy.
*   **IV. Kinematic**: meander (tortuosity) and acceleration variance.
*   **V. Habituation**: per-minute statistics for minutes 1–5 and the habituation index.

Notes:
*   **CW/CCW** is decided by the sideways (tangential) movement around the arena center over 5 frames (at least 2 px), counted during High Active only.
*   **U-turns** are counted from clear-movement headings: a U-turn is a change of heading of at least 120° between runs of clear movement (≥ 2 px/frame). In-place turns are counted once the rat moves off in the opposite direction. The turn times are written to `<name>_features_uturns.txt` as intervals (the turn happened inside the interval). The count is more reliable than the exact time.
*   **Meander** and **acceleration variance** are dominated by the whole-pixel coordinates (1-px flicker of a resting rat) and should not be interpreted as behavioral measures yet.
*   Every extraction checks the invariants (time conservation, phase counts, 300 s window) and prints a validation report.

Output files: `<name>_features.json`, `<name>_features_debug.csv` (per-frame speed, state, direction, corner flag), `<name>_features_uturns.txt`, and with `--debug` an HTML dashboard (and a sector image if `--video` is given).

### 7. Playback and visualization
Run a playback on the filtered results. It dynamically draws the rat's state, fading comet tail, and directional vectors on the original video:  
`ambitus playback results/LE_G17_N3_D0_T1 --video-dir videos`  
*(Use `--save` to record the playback to an MP4 file.)*

---

## Fully automated pipelines

### Run pipeline on a single video
Executes the entire workflow (Align ROI -> Track -> Filter -> Heatmap -> Classify -> Extract Features) on a single video.

`ambitus run videos/LE_G17_N3_D0_T1.mp4 --backtrack --bg-method zone`

**Advanced run options:**
*   `--skip-seconds 40`: Skip the first 40 seconds manually (e.g., to ignore the researcher's hand).
*   `--bg-method zone|global|quiet`: Background method of the static tracker (see Tracking).
*   `--backtrack`: Finds the light-on event in the last 120 s of the video and tracks the 300 seconds before it (ending 1 s before the light-on). Forces the `global` background, except with `--bg-method zone`, which is kept (the zone background then only uses frames before the light-on). If no light-on is found (or it is less than 330 s into the video), tracking starts at 0 s, the rat insertion is detected automatically and `quiet` (or `zone`) is used.
*   `--no-excursion-filter`: Keep short tracking excursions (see Filtering).
*   `--min-area`: Minimum blob area in pixels (default 100).
*   `--out-dir`: Write the results to this folder.
*   `--export`: Updates `summary.xlsx` with the newly extracted features.
*   `--show` / `--debug`: Display results and debug views.

*Two videos of the published dataset have a fixed start time hard-coded in `tracker.py` (inconsistent light source), which overrides `--skip-seconds` and `--backtrack`: `LI_G19_N2_D3_T1` (33 s) and `LI_G17_N33_D1_T1` (48 s).*

### Run pipeline on a directory
Runs the complete pipeline recursively on the videos found in a directory and mirrors the folder structure into a `results/` folder next to the video directory. At the end, all feature files are exported to Excel (master and folder sheets).

`ambitus run_dir videos --backtrack --bg-method zone`

*   All `.mp4` files are processed. `.mpg` files are processed **only if the directory tree contains no `.mp4` at all**. Keep one format per run.
*   Supports `--backtrack`, `--bg-method`, `--skip-seconds`, `--min-area`, `--no-excursion-filter`, the filter/heatmap options, `--show` and `--debug`. `--export` and `--out-dir` are not available (the export always runs).

### Output files per video
`results/<subfolders>/<video>/`:
*   `<video>_config.json` – aligned ROI
*   `<video>_DEBUG_BACKGROUND.jpg` – background used by the static tracker
*   `<video>_static_trajectory.csv`, `_trajectory_map.jpg`, `_statistics.txt` – raw tracking
*   `<video>_static_trajectory_filtered.csv` – filtered trajectory (300 s), plus the filter dashboard and heatmap
*   `<video>_static_trajectory_filtered_classified.csv` – filtered trajectory with activity state
*   `<video>_static_features.json`, `_features_debug.csv`, `_features_uturns.txt` – features

---

## Data aggregation and quality control

### Export to Excel
Aggregates all extracted feature JSON files into flat Excel spreadsheets.  
`ambitus export results`
*   Always creates a `summary.xlsx` in the root of the results directory, with one row per video. Strain, breeding generation, animal number and treatment are parsed from the file and folder names.
*   Use `--all-excels` to additionally create a `<folder>_features_summary.xlsx` for every subfolder and a `<video>_features.xlsx` for every video.

### Audit results (quality check)
Scans all processed results and lists videos that need a closer look:  
`ambitus check results`
*   **Too short**: less than 280 s of evaluated time.
*   **Tracking error**: more than 5 rejected wall-crossing gaps.
*   **Escaped/lost**: more than 15% missed frames.
*   **Early start**: the 300 s window starts before 20 s (e.g. the light-on was not found and the hand was detected as the rat). Re-run these with `--skip-seconds`.
*   Generates a human-readable `Problematic_Videos_Report.txt` in the results directory.

---

## Validation

`generate.py` is a synthetic test suite for the filtering, activity, feature extraction and tracking steps. It generates rat-like paths (gradual turns, noise, rests) and synthetic videos with known ground truth and compares the pipeline output with it.

Run it from the package folder:

```
cd src/ambitus
python generate.py
```

The full suite takes about 5 minutes.
*   `--only test6`: run only tests whose name contains the text (e.g. `test8` activity frame sync, `test9` zone background).
*   `--no-noise`: skip the noise variants.
*   `--seeds 5`: more noise seeds.
*   `--html`: write the debug dashboards.

Test files are written to `./synthetic_test_suite`. The exit code is 1 if any check fails.

### Manual annotation

`annot.py` measures tracking accuracy on real recordings. It picks a random `.mp4` video from a folder tree and a random frame from it, shows the frame, and waits for the user to click on the center of the rat. Each click is logged, and the next random video is loaded. The clicked positions can then be compared with the tracked positions. The annotations of the published dataset were made with this tool and are included in the `annotation` folder of the dataset.

The script has no command-line arguments. Set the video folder in the `DATASET_DIR` line at the top of `annot.py`, then run it from the package folder:

```
cd src/ambitus
python annot.py
```

Controls in the annotation window:
*   **Left click**: log the position and load the next random video.
*   **`a`** / **`d`**: move 0.5 s backward / forward, for example when the rat is hidden or blurred.
*   **`s`**: skip this frame and load the next random video.
*   **`q`**: quit.

The positions are appended to `annotation_log.csv` next to the script, with the columns `video_path`, `frame_number`, `x` and `y` (pixels of the original frame). An existing log is continued, so annotation can be done in several sessions.

*Available legacy tracking methods for `analyze-exp`: dense optical flow with Farneback (dof), simple+Kalman-Filter (kf), simple+Optical-flow (of), pure optical flow (pure_of), Motion History Image (mhi).*

---

## Versions

- **v1.0.1**: the version described in the SoftwareX article. It differs from v1.0 only in repository layout (source code in `src/ambitus/`), documentation and license file. The results are identical.
- **v1.0** (`release_v1.0`): the version that produced the published dataset.

## Citation

If you use AmbitusVision, please cite the software article and the dataset:

- Schäffer L, Kalmár G. AmbitusVision: an open-source pipeline for tracking rats and extracting movement features from overhead videos of a square-corridor arena. *SoftwareX* (in progress).
- Schäffer L, Liszli P, Horváth G, Kalmár G. Video and trajectory dataset for rodent tracking in a square-corridor neuropsychopharmacological experiment. Scientific Data (under review).

## Dataset
Schäffer L, Liszli P, Horváth G, Kalmár G. Video and trajectory dataset for rodent tracking in a square-corridor neuropsychopharmacological experiment [dataset]. SZTE Data Repository; 2026. https://doi.org/10.82570/SZTE_Datarepo_81fgh-zcr72

## License

AmbitusVision is released under the MIT License. See [LICENSE.txt](LICENSE.txt).

## Contact
György Kalmár, Department of Technical Informatics, University of Szeged: kalmargy@inf.u-szeged.hu
László Schäffer, Department of Technical Informatics, University of Szeged: schaffer@inf.u-szeged.hu