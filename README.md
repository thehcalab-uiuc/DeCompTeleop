# Project Code for *Towards Real-Time Generation of Delay-Compensated Video Feeds for Outdoor Mobile Robot Teleoperation*

**Note:** This repository contains the code for our [ICRA 2025](https://2025.ieee-icra.org/) paper. For more details, please refer to the [project website](https://sites.google.com/illinois.edu/comp-teleop) and [arXiv preprint](https://arxiv.org/abs/2409.09921). For issues, contact Neeloy Chakraborty at neeloyc2@illinois.edu.

<div align="center">
    <img src="figures/offline.gif" height="240" alt="Offline evaluation of our method compared with baselines on a middle stage growth video.">
    <img src="figures/online.gif" height="240" alt="Real-time online evaluation of our method on an early stage growth video.">
</div>

## Abstract

Teleoperation is an important technology to enable supervisors to control agricultural robots remotely.
However, environmental factors in dense crop rows and limitations in network infrastructure hinder the reliability of data streamed to teleoperators.
These issues result in delayed and variable frame rate video feeds that often deviate significantly from the robot's actual viewpoint.
We propose a modular learning-based vision pipeline to generate delay-compensated images in real-time for supervisors. 
Our extensive offline evaluations demonstrate that our method generates more accurate images compared to state-of-the-art approaches in our setting.
Additionally, ours is one of the few works to evaluate a delay-compensation method in outdoor field environments with complex terrain on data from a real robot in real-time.
Resulting videos and code are provided [here](https://sites.google.com/illinois.edu/comp-teleop).

## Getting Started

### File Structure

Below, we give a high-level overview of the directory structure of our project.
```
./
└───depth
|   └───collect_eval_metrics - Code to evaluate performance of depth models
|   └───dav2 - Code to train DAv2 models and save depth predictions
|   └───dino - Code to train DINOv2 models and save depth predictions
└───inpainting
|   └───eval_inpaint.py - Script to evaluate performance of inpainting models
|   └───save_resnet_inpaint.py - Script to save inpainted predictions from trained ResNet model
|   └───save_telea_inpaint.py - Script to save inpainted predictions using Telea method
|   └───train_resnet_inpaint.py - Script to train ResNet inpainting model
└───ros/catkin_ws/src
|   └───combined_node - ROS node code to perform real-time delay compensation
|   └───data_collection - Code to generate dataset of frames from recorded SVO files
|   └───rtab_mapping - Code with launch files to play back SVO files and visualize topics in RViz
|   └───zed-ros-wrapper - ROS nodes for Stereolabs ZED SDK
└───weights
    └───depth - Weights for depth models
    └───inpainting - Weights for inpainting models
```

### Installation for Dataset Creation and Real-Time Evaluation with Pulsar
- Using Ubuntu 20.04 system with CUDA 12.4 GPU (we used NVIDIA GeForce RTX 2080 8GB)
- Create Conda environment with Python 3.9
    - ```
      conda create -n "terra-torch3d" python=3.9
      conda activate terra-torch3d
      ```
- Install required packages for Conda environment
    - Install packages using our environment yaml file [here](terra-torch3d.yml), **EXCEPT** for PyTorch3D
    - Install [our fork](https://github.com/TheNeeloy/pytorch3d) of PyTorch3D from source by following instructions [here](https://github.com/TheNeeloy/pytorch3d/blob/main/INSTALL.md)
- Install CUDA 11.7 user-wide (this installed CUDA may be different from the one referenced in the conda environment, and it is used by the ZED SDK)
    - Follow the instructions [here](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04) to download and install
    - Add necessary lines to `~/.bashrc` to ensure correct paths are sourced (modify depending on where it is installed)
        - ```
          export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
          export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.7/lib64
          ```
- Install ZED SDK 3.7
    - Download SDK from [here](https://www.stereolabs.com/developers/release/3.7)
    - Follow instructions [here](https://www.stereolabs.com/docs/installation/linux) to install the SDK
- Install ROS Noetic and dependencies
    - Follow the instructions at [https://wiki.ros.org/noetic/Installation/Ubuntu](https://wiki.ros.org/noetic/Installation/Ubuntu) to install ROS
    - (Optional) Install RTAB-MAP by following instructions [here](https://github.com/introlab/rtabmap_ros#rtabmap_ros)
- Clone repository and setup ROS workspace
    - Ensure GIT LFS is installed
        - To download weights alongside code, run `git lfs install`
        - To ignore weights and only pull code, run `git lfs install --skip-smudge`
    - ```
      git clone https://gitlab.engr.illinois.edu/hubris/decompteleop.git
      cd decompteleop/ros/catkin_ws
      catkin make
      ```

## Collecting Offline Dataset
At this time, we are not planning to release the dataset we collected for training and evaluation. 
However, we provide instructions for how to curate your own dataset containing RGB, depth, and pose frames.
Follow the instructions [here](ros/catkin_ws/src/data_collection/README.md) to collect your own dataset.

## Training and Evaluating Depth Models

Pretrained weights for the depth models are located at `weights/depth`.
Open the following docs to learn how to train your own depth models:

- DAv2: [here](depth/dav2/README.md)
- DINOv2: [here](depth/dino/README.md)

Open the doc [here](depth/collect_eval_metrics/README.md) to learn how to evaluate the accuracy of depth predictions.

## Training and Evaluating Inpainting Models

Pretrained weights for the inpainting models are located at `weights/inpainting`.
Open the doc [here](inpainting/README.md) to learn how to train and evaluate your own inpainting models.

## Real-time ROS Evaluation

Open the doc [here](ros/catkin_ws/src/combined_node/README.md) to learn how to run our real-time delay compensation ROS node.

## References
We thank the authors of the following projects for making their data/code/models available (in no particular order):
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [ZED ROS Wrapper](https://github.com/stereolabs/zed-ros-wrapper)
- [RTAB-MAP ROS Package](https://github.com/introlab/rtabmap_ros)
- [Terrasentia Dataset](https://github.com/jrcuaranv/terrasentia-dataset)
- [DAv2](https://github.com/DepthAnything/Depth-Anything-V2)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [W-RIZZ](https://github.com/andreschreiber/W-RIZZ)
- [SRVP](https://github.com/edouardelasalles/srvp)
- [DMVFN](https://github.com/hzwer/CVPR2023-DMVFN)
- [SynSin](https://github.com/facebookresearch/synsin)

## Citation

```
@INPROCEEDINGS{chakraborty2025towards,
  author={Chakraborty, Neeloy and Fang, Yixiao and Schreiber, Andre and Ji, Tianchen and Huang, Zhe and Mihigo, Aganze and Wall, Cassidy and Almana, Abdulrahman and Driggs-Campbell, Katherine},
  booktitle={2025 International Conference on Robotics and Automation (ICRA)}, 
  title={Towards Real-Time Generation of Delay-Compensated Video Feeds for Outdoor Mobile Robot Teleoperation}, 
}
```
