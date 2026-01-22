# Ambitus Analysis CLI

## Setup

1. Clone the repository  
   git clone <repository-url>

2. Navigate into the project directory  
   cd <repository-name>

3. Create a virtual environment  
   python3 -m venv venv

4. Activate the virtual environment  
   source venv/bin/activate

5. Install the package in editable mode  
   pip install -e .

6. Add data files  
   Paste the shared image files into:  
   ambitus/data/

## Video Conversion

Convert .mpg files to compatible .mp4 videos before processing.

Example:  
ffmpeg -i ambitus/data/LE_17_1_45.mpg -vf "yadif,scale=768:576,setsar=1" -pix_fmt yuv420p ambitus/data/45.mp4

## Testing

Run a basic test command to verify everything works:  
ambitus load-video ambitus/data/45.mp4 --show

Process video and plot and save trajectory:  
ambitus analyze ambitus/data/45.mp4 --show --skip-seconds 0.0
