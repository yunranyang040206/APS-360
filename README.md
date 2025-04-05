# APS-360

# Object Detection and Classification Project

## Overview
This repository contains code and notebooks for an integrated object detection and classification model using advanced Region Proposal Networks (RPN), CBAM attention modules, ROI pooling, and a ResNet-based classifier.

## Repository Structure

- `Adding_labelsandboxes_for_ground_truth.ipynb`: Notebook for preparing and labeling ground truth data.
- `Data_processing_step_1.py`: Initial preprocessing step for raw data.
- `Data_processing_step_2.py`: Secondary data cleaning and formatting.
- `RPN+CBAM+ROI.ipynb`: Implementation of Region Proposal Network (RPN) with Convolutional Block Attention Module (CBAM) and Region of Interest (ROI) pooling.
- `RPN+ROI+Classification_integration.ipynb`: Integrated pipeline connecting RPN, ROI pooling, and the classification model.
- `RPN_CBAM.py`: Python module defining the RPN architecture with CBAM.
- `classification_model+train.ipynb`: Training notebook for the classification model (ResNet18).
- `image_enhancement.ipynb`: Notebook for image preprocessing, enhancement, and augmentation.
- `rpn_roi_integrated.py`: Python script combining RPN and ROI pooling for inference.
- `yolo.py`: Reference or auxiliary implementation using YOLO-based methods.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib


## Usage
1. **Data Processing:** Run `Data_processing_step_1.py` and `Data_processing_step_2.py` to prepare data.
2. **Model Training:** Use `classification_model+train.ipynb` to train your classifier.
3. **Object Detection and ROI Pooling:** Execute `RPN+CBAM+ROI.ipynb` for feature extraction and proposal generation.
4. **Full Integration:** Run `RPN+ROI+Classification_integration.ipynb` to perform complete detection and classification.

## Highlights
- **Enhanced RPN:** Uses CBAM attention mechanisms to improve proposal accuracy.
- **ROI Pooling:** Produces fixed-sized feature maps, enabling seamless integration with classifiers.
- **Transfer Learning:** Leverages a pre-trained ResNet18 model for object classification.

## Acknowledgments
Thanks to all contributors involved in the implementation and testing of the pipeline.

