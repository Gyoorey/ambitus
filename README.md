# Crack Detection CLI
git clone
cd into the repository
[python3 -m venv venv] - creates a virtual environment
[source venv/bin/activate] - activates venv
pip install -e .
[paste the shared image files into ambitus/data]

[convert mpg files to correct mp4 videos]
e.g.:
ffmpeg -i ambitus/data/LE_17_1_45.mpg -vf "yadif,scale=768:576,setsar=1" -pix_fmt yuv420p ambitus/data/45.mp4

[test with commands like:]
ambitus load-video ambitus/data/45.mp4 --show
