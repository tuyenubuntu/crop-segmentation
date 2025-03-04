# Image Processor

This repository contains a Python script for image processing using OpenCV, NumPy, and Scikit-Image. The script is designed to load a grayscale image, apply multi-level Otsu's thresholding, and extract the largest connected component for further analysis. The extracted component can then be used to calculate bounding boxes and crop regions of interest.

## Table of Contents

- [Image Processor](#image-processor)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Example](#example)
    - [Command-Line Usage](#command-line-usage)
  - [Class Details](#class-details)
    - [`ImageProcessor`](#imageprocessor)
      - [Attributes:](#attributes)
      - [Methods:](#methods)
    - [Example:](#example-1)
  - [Contribution](#contribution)
  - [License](#license)

## Features

- Load a grayscale image from a file path.
- Apply multi-level Otsu's thresholding to segment the image.
- Extract the largest connected component in a binary image.
- Automatically calculate and modify bounding boxes for cropping.
- Visualize image histograms with threshold markers.
- Display processed images.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-processor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd image-processor
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- Scikit-Image

You can install these dependencies using:
```bash
pip install opencv-python-headless numpy matplotlib scikit-image
```

## Usage

### Example

1. Import the `ImageProcessor` class:
   ```python
   from image_processor import ImageProcessor
   ```

2. Create an instance of `ImageProcessor`:
   ```python
   processor = ImageProcessor('path/to/image.bmp')
   ```

3. Display the histogram with thresholds:
   ```python
   processor.display_histogram_with_thresholds()
   ```

4. Crop the largest connected component:
   ```python
   cropped_surface = processor.crop_image()
   processor.display_image(cropped_surface, 'Cropped Surface')
   ```

### Command-Line Usage

You can also modify the script to accept command-line arguments for image path and number of thresholds.

## Class Details

### `ImageProcessor`

#### Attributes:
- `image_path`: Path to the input image.
- `num_thresholds`: Number of thresholds to calculate.
- `img`: Loaded grayscale image.
- `thresholds`: Thresholds calculated using Otsu's method.
- `binary_image`: Binary image after thresholding.
- `surface_mask`: Mask of the largest connected component.
- `bounding_box`: Bounding box of the largest connected component.

#### Methods:
- `load_image()`: Loads a grayscale image.
- `find_multiotsu_thresholds()`: Finds thresholds using Otsu's method.
- `apply_threshold(index=1)`: Applies binary thresholding.
- `find_largest_connected_component(binary_image)`: Finds the largest connected component.
- `auto_get_bounding_box()`: Automatically calculates the bounding box.
- `get_bounding_box(surface_mask)`: Calculates the bounding box of a specified connected component.
- `modify_bounding_box(bounding_box, delta_round=10)`: Modifies the bounding box.
- `crop_image(bounding_box=None)`: Crops the image using a bounding box.
- `display_histogram_with_thresholds()`: Displays the histogram with thresholds.
- `display_image(img, title='Image')`: Displays an image.

### Example:
   ```python
    python test.py
   ```
## Contribution

Feel free to contribute to this project by submitting issues or pull requests. Make sure to follow the coding standards and include proper documentation for new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
