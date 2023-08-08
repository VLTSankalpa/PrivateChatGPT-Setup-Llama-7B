# **🚀 Setting Up Llama-7B on Google Cloud VM with NVIDIA**

This guide will help you to set up Llama-7B on your Google Cloud VM equipped with NVIDIA GPUs.

### **Prerequisites:**

- Google Cloud VM: 24vCPU, 96 RAM, and 2*NVIDAN L4s

### **1. Initial System Setup**

```bash

sudo su -
passwd
passwd tharindu_sankalpa
usermod -a -G sudo,adm tharindu_sankalpa

```

### **2. SSH Configuration**

```bash

vi /etc/ssh/sshd_config
systemctl restart sshd

```

After making changes to the SSH configuration, connect to the server:

```bash

ssh tharindu_sankalpa@YOUR_VM_IP_ADDRESS

```

### **3. System Update**

```bash

sudo apt-get update
sudo apt-get upgrade

```

### **4. Install NVIDIA Driver**

Check current NVIDIA status:

```bash

nvidia-smi

```

Install the NVIDIA driver:

```bash

sudo apt-get install nvidia-driver-525
sudo reboot now

```

Once the system is rebooted, check your VM's availability:

```bash

ping YOUR_VM_IP_ADDRESS
ssh tharindu_sankalpa@YOUR_VM_IP_ADDRESS

```

### **5. Install CUDA Toolkit**

Fetch and configure CUDA:

```bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

```

Reboot the VM:

```bash

sudo reboot now

```

### **6. Install Miniconda & Setup TensorFlow Environment**

```bash

wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
chmod +x Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
./Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
source .bashrc
conda create --name tf_gpu_env python=3.9.13
conda activate tf_gpu_env
pip install tensorflow

```

Verify CUDA compiler version:

```bash

nvcc --version
sudo apt install nvidia-cuda-toolkit

```

### **7. Install cuDNN**

```bash

wget https://storage.googleapis.com/windows-server-imaage-bucket/cudnn-11.3-linux-x64-v8.2.1.32.tgz
tar -xzvf cudnn-11.3-linux-x64-v8.2.1.32.tgz
sudo cp -P cuda/include/cudnn*.h /usr/local/cuda/include/
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

```

Then, adjust the environment and libraries:

```bash

conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

```

Test TensorFlow and GPU:

```bash

python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```

### **8. Setting Up Llama2**

```bash

conda create -n llama2 python=3.9
conda activate llama2
git clone https://github.com/thisserand/llama2_local.git
cd llama2_local/
pip install -r requirements.txt

huggingface-cli login
# Follow on-screen prompts to enter your Hugging Face token

```

Run Llama:
