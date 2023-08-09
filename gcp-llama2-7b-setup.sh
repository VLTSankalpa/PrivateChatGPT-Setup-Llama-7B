sudo su -
passwd
passwd tharindu_sankalpa
usermod -a -G sudo,adm tharindu_sankalpa

vi /etc/ssh/sshd_config


systemctl restart sshd

ssh tharindu_sankalpa@IP

sudo apt-get update
sudo apt-get upgrade

nvidia-smi

sudo apt-get install nvidia-driver-525

sudo reboot now

ping IP
ssh tharindu_sankalpa@IP


tharindu_sankalpa@llama2-model-endpoint2:~$ nvidia-smi
Tue Aug  8 16:45:14 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA L4           Off  | 00000000:00:03.0 Off |                    0 |
| N/A   61C    P8    19W /  72W |     70MiB / 23034MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA L4           Off  | 00000000:00:04.0 Off |                    0 |
| N/A   61C    P8    19W /  72W |      4MiB / 23034MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1531      G   /usr/lib/xorg/Xorg                 59MiB |
|    0   N/A  N/A      1590      G   /usr/bin/gnome-shell               10MiB |
|    1   N/A  N/A      1531      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
tharindu_sankalpa@llama2-model-endpoint2:~$
tharindu_sankalpa@llama2-model-endpoint2:~$


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda


sudo reboot now


wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
chmod +x Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
./Miniconda3-py39_23.5.2-0-Linux-x86_64.sh

source .bashrc

conda create --name tf_gpu_env python=3.9.13
conda activate tf_gpu_env
pip install tensorflow

nvcc --version

sudo apt install nvidia-cuda-toolkit

wget https://storage.googleapis.com/windows-server-imaage-bucket/cudnn-11.3-linux-x64-v8.2.1.32.tgz
tar -xzvf cudnn-11.3-linux-x64-v8.2.1.32.tgz
sudo cp -P cuda/include/cudnn*.h /usr/local/cuda/include/
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
Num GPUs Available:  2



############ Setup llama2

conda create -n llama2 python=3.9
conda activate llama2
git clone https://github.com/thisserand/llama2_local.git
cd llama2_local/
pip install -r requirements.txt

huggingface-cli login
######### HF token

python llama.py --model_name="meta-llama/Llama-2-7b-chat-hf"