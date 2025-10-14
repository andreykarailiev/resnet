# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create & activate the Conda environment
~/miniconda3/bin/conda create -n arena-env python=3.11 -y
~/miniconda3/bin/conda activate arena-env
~/miniconda3/envs/cfdx/bin/pip install -r ~/resnet/requirements.txt 
~/miniconda3/bin/conda install -n cfdx ipykernel --update-deps --force-reinstall -y
