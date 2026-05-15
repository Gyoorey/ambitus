# Ambitus Analysis CLI

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

6. Add data files  
   Paste the shared image and video files into:  
   `ambitus/data/`

## Video Conversion

Convert `.mpg` files to compatible `.mp4` videos before processing (Optional but recommended for speed).

Example:  
`ffmpeg -i ambitus/data/LE_17_1_45.mpg -vf "yadif,scale=768:576,setsar=1" -pix_fmt yuv420p ambitus/data/45.mp4`

*(Note: The software also directly supports `.mpg` format via `imageio`, automatically applying the `yadif,scale=768:576,setsar=1` de-interlacing parameters.)*

---

## Usage Guide & Commands

### 1. Basic Testing
Run a basic test command to verify everything works by displaying the first frame:
`ambitus load-video data/45.mp4 --show`

### 2. ROI Alignment
Calculate the camera offset and rotate the Region of Interest (ROI) mask for a new video based on a reference square (`base_reference.jpg`). Use `--debug` to see the aligned ROI.
`ambitus align_roi data/45.mp4 --debug`

*(You can manually define the ROI polygons via a GUI if needed: `ambitus roi data/45.mp4`)*

### 3. Tracking (Individual)
Track the rat using fast methods: `static` (Masked Running Average) or `simple` (MOG2).
`ambitus analyze data/45.mp4 --show --skip-seconds 0.0 --debug --method static --bg-method quiet --config results/45/45_config.json`

*   **`--bg-method quiet`**: (Default) Finds a stable 5-second window at the start to generate the background.
*   **`--bg-method global`**: Takes a median of the entire video to generate the background (Useful if objects like a stick are left in the arena at the start).

### 4. Trajectory Filtering
Interpolate missing frames (with strict geometric wall-collision checks), remove tracking teleportation spikes, and apply a low-pass filter to the trajectory CSV.
`ambitus filter results/45/45_static_trajectory.csv --plot --plotly`
*   `--plot`: Generates a static OpenCV `.jpg` comparison dashboard.
*   `--plotly`: Generates an interactive HTML dashboard.

### 5. Heatmap Generation
Generate a color density heatmap from the filtered CSV and overlay it on the original video's first frame:
`ambitus heatmap results/45/45_static_trajectory_filtered.csv --show`

### 6. Activity Classification & Feature Extraction
Classify micro-movements (High Active, Low Active, Inactive) based on local pixel variance:
`ambitus activity results/45/45_static_trajectory_filtered.csv --debug`

Extract 39 distinct behavioral, spatial, and kinematic features into a JSON file:
`ambitus extract results/45/45_static_trajectory_filtered_classified.csv --debug`

### 7. Playback & Visualization
Run a playback on the filtered results. It dynamically draws the rat's state, fading comet tail, and directional vectors on the original video:
`ambitus playback results/45 --method static --speed 2.0 --tail 3.0`
*(Use `--save` to record the playback to an MP4 file).*

---

## Fully Automated Pipelines

### Run Pipeline on a Single Video
Executes the entire workflow (Align ROI -> Track -> Filter -> Heatmap -> Classify -> Extract Features) on a single video.

`ambitus run data/45.mp4 --show --debug`

**Advanced Run Options:**
*   `--skip-seconds 40`: Skip the first 40 seconds manually (e.g., to ignore the researcher's hand).
*   `--bg-method global`: Use the global median background generator.
*   `--backtrack`: Automatically finds the light-on event at the end of the video, forces the `global` background, and perfectly crops the tracking to the exact 300 seconds before the light turns on.
*   `--export`: Automatically updates the Master Excel and Group Excel sheets with the newly extracted features.

### Run Pipeline on a Directory
Runs the complete pipeline recursively on all `.mp4` and `.mpg` videos found in a directory. Automatically mirrors the folder structure into the `results/` folder.

`ambitus run_dir data/Experiment_1/`
*(Supports the same advanced flags as the single `run` command, including `--backtrack`).*

---

## Data Aggregation & Quality Control

### Export to Excel
Aggregates all extracted feature JSON files into flat Excel spreadsheets.
`ambitus export results/`
*   Always creates an `Ambitus_Master_Features.xlsx` in the root of the results directory.
*   Use `--all-excels` to additionally create individual summary sheets for every subfolder and single video.

### Audit Results (Quality Check)
Scans all processed results for critical errors (e.g., rat escaped/climbed walls, video too short, tracking teleportation errors, early-start hand artifacts).
`ambitus check results/`
*   Generates a human-readable `Problematic_Videos_Report.txt` in the results directory.

---

*Available legacy tracking methods for `analyze-exp`: dense optical flow with Farneback (dof), simple+Kalman-Filter (kf), simple+Optical-flow (of), pure optical flow (pure_of), Motion History Image (mhi).*