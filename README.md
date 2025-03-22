# HO-Cap Annotation Pipeline

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/downloads/release/python-31015/)
[![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit/)
[![GLP-v3 License](https://img.shields.io/badge/License-GPL--3.0-3DA639.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)



## Installation

This code is tested with `Python 3.10` and `CUDA 11.8` on `Ubuntu 20.04`. **Make sure CUDA 11.8 is installed on your system before running the code.**

#### 1. Clone the repository

```bash
git clone https://github.com/JWRoboticsVision/HO-Cap-Annotation.git
```

#### 2. Change current directory to the repository

```bash
cd HO-Cap-Annotation
```

#### 3. Create conda environment

```bash
conda create -n hocap-annotation python=3.10
```

#### 4. Install PyTorch

```bash
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
```

#### 5. Install the HO-Cap Annotation Package

```bash
bash ./scripts/install_hocap-annotation.sh
```

#### 6. Download MANO models

Download MANO models and code (mano_v1_2.zip) from the [MANO website](https://mano.is.tue.mpg.de/) and place the extracted .pkl files under `config/mano_models` directory. The directory should look like this:

```
./config/mano_models
├── MANO_LEFT.pkl
└── MANO_RIGHT.pkl
```

#### 7. Install Third-Party Tools (Optional)

##### 7.1 Install FoundationPose

- Initialize and build FoundationPose:
  ```bash
  bash ./scripts/install_foundationpose.sh
  ```
- Download checkpoints
  ```
  bash ./scripts/download_models.sh --foundationpose
  ```

##### 7.2 Install SAM2

- Initialize and build SAM2:
  ```bash
  bash ./scripts/install_sam2.sh
  ```
- Download checkpoints
  ```bash
  bash ./scripts/download_models.sh --sam2
  ```
