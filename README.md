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
   Paste the shared image files into:  
   ambitus/data/

## Video Conversion

Convert .mpg files to compatible .mp4 videos before processing.

Example:  
`ffmpeg -i ambitus/data/LE_17_1_45.mpg -vf "yadif,scale=768:576,setsar=1" -pix_fmt yuv420p ambitus/data/45.mp4`

## Testing

Run a basic test command to verify everything works:

`ambitus load-video ambitus/data/45.mp4 --show`

Tracking with fast methods: static (diff) or simple (MOG2):

`ambitus analyze data/45.mp4 --show --skip-seconds 0.0 --debug --method static --config arena_config_poly.json`

ROI alignment based on reference square coords (base_reference.jpg): use --debug to see aligned ROI

`ambitus align_roi data/45.mp4 --debug`

Interpolate missing frames, remove tracking outliers, and apply a low-pass filter to the trajectory CSV. Use only --plot for cv2, use --plot and --plotly for html.

`ambitus filter results/45_static_trajectory.csv --plot --plotly`

Generate a color heatmap from the trajectory CSV data and overlay it on the original video's first frame. The script automatically finds the corresponding video file:

`ambitus heatmap results/45_static_trajectory_filtered.csv --show`

Run pipeline (align, tracking, filtering, heatmap):

`ambitus run data/45.mp4 --show --debug`

Run pipeline on a directory (align, tracking, filtering, heatmap):

`ambitus run_dir data`

Available legacy methods: dense optical flow with Farneback (dof), simple+Kalman-Filter (kf), simple+Optical-flow (of), pure opcital flow  (pure_of), Motion History Image (mhi)

`ambitus analyze-exp data/45.mp4 --show --skip-seconds 0.0 --debug --method dof`
