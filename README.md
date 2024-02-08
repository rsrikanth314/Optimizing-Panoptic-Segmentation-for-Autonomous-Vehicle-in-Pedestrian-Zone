# Optimizing Panoptic Segmentation for Autonomous Vehicles in Pedestrian Zones

This repository contains the implementation of optimizing panoptic segmentation for autonomous vehicles in pedestrian zones. The project employs the bottom-up approach using the panoptic-deeplab method and focuses on further optimizing its performance in specific areas, such as pedestrian zones.

## Folder Structure

- **FINROC (ROS):**
  - Contains C++ scripts for the panoptic segmentation module that loads the optimized ONNX model and performs inference.
  - Subscribes to incoming image data and publishes the panoptic segmentation output.
  - Includes a data extraction module for playback and extracting images from data recording `.bin` files, similar to ROS bag files.

- **infernce_scripts:**
  - Contains panoptic deeplab inference scripts.

- **ONNX:**
  - Contains scripts related to exporting the model to ONNX, optimization, and ONNX runtime inference scripts.

- **panoptic_seg_data_preparations:**
  - Scripts for preparing the data in COCO-panoptic segmentation format.

- **train:**
  - Contains training scripts and custom hooks.

## Thesis Overview

The thesis work focuses on enhancing the performance of panoptic segmentation in pedestrian zones, utilizing the bottom-up approach with the panoptic-deeplab method. Key aspects of the work include:

- Exploration and implementation of the panoptic-deeplab method.
- Feasibility study for deploying the model on edge devices.
- Optimization of runtime inference speed by exporting the model to ONNX.
- Model size reduction and improved inference speed optimizations.
- Runtime inference using the ONNX runtime.

## Usage

Provide instructions on how to use your project, including steps for:
1. Setting up the environment.
2. Running inference scripts.
3. Using the FINROC module for ROS integration.
4. Any additional configurations or dependencies.

## Data Preparation

Explain the process of data preparation using the scripts in the `panoptic_seg_data_preparations` folder. Include details on the COCO-panoptic segmentation format and how to adapt the scripts for custom datasets.

## Model Training

If applicable, provide details on how to use the scripts in the `train` folder for training the panoptic segmentation model. Include any custom hooks or configurations that might be necessary.

## ROS Integration (FINROC)

Explain the integration of the panoptic segmentation module into the Robot Operating System (ROS) using the scripts in the `FINROC` folder. Include information on subscribing to image data, publishing segmentation output, and utilizing the data extraction module.

## Acknowledgments

Mention any third-party libraries, tools, or resources that were used in the project.

## License

This project is licensed under the [LICENSE NAME] - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

For any questions or concerns, please feel free to contact [Your Name] at [your email].
