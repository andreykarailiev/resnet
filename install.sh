# Create & activate the Conda environment
~/miniconda3/bin/conda init cfdx
~/miniconda3/bin/conda activate cfdx
~/miniconda3/envs/cfdx/bin/pip install -r ~/resnet/requirements.txt 
~/miniconda3/bin/conda install -n cfdx ipykernel --update-deps --force-reinstall -y