conda init
conda create -n yt python=3.8.18
conda activate yt
conda install -c conda-forge ffmpeg
conda install --yes -c pytorch pytorch=1.7.1 torchvision # cudatoolkit=11.0
python3.8 -m pip install -r requirements.txt
