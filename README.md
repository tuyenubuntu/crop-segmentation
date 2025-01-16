# crop-segmentation
# crop-segmentation

# Image Processor

## Image Processing Python Class

This repository provides a Python class for processing images using Multi-Otsu thresholding, detecting the largest connected component, and automatically cropping the image based on the detected bounding box.

## Contents

- [crop-segmentation](#crop-segmentation)
- [crop-segmentation](#crop-segmentation-1)
- [Image Processor](#image-processor)
  - [Image Processing Python Class](#image-processing-python-class)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Initialize ImageProcessor](#initialize-imageprocessor)

## Overview

This library provides an `ImageProcessor` class that helps with image analysis tasks such as:

- Applying multi-threshold segmentation using Otsu’s method.
- Detecting the largest connected component in the image (used for surface analysis).
- Calculating the bounding box around the largest surface.
- Cropping the image based on the bounding box.
- Visualizing the image and the thresholds through a histogram.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-processor.git
   ```
2. Navigate to the library folder:
   ```bash
   cd image-processor
   ```
3. Ensure Install the required dependencies:
   ```bash
   pip install opencv-python numpy matplotlib scikit-image
   ```

## Usage

### Initialize ImageProcessor
   ```bash
    from image_processor import ImageProcessor
    # Initialize the processor with an image path and number of thresholds
    processor = ImageProcessor('path/to/image.bmp*')  #Any extension
   ```
    