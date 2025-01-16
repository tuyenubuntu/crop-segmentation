import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops

class ImageProcessor:
    def __init__(self, image_path, num_thresholds=4):
        """Initialize the ImageProcessor with an image path and number of thresholds."""
        self.image_path = image_path
        self.num_thresholds = num_thresholds
        self.img = self.load_image()
        self.thresholds = self.find_multiotsu_thresholds()
        self.binary_image = self.apply_threshold()
        self.surface_mask = self.find_largest_connected_component(self.binary_image)
        self.bounding_box = self.auto_get_bounding_box()

    def load_image(self):
        """Load an image in grayscale mode."""
        return cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

    def find_multiotsu_thresholds(self):
        """Find multiple thresholds using Otsu's method."""
        thresholds = threshold_multiotsu(self.img, classes=self.num_thresholds + 1)
        return thresholds

    def apply_threshold(self, index=1):
        """Apply binary thresholding to the image."""
        _, binary_image = cv2.threshold(self.img, self.thresholds[index], 255, cv2.THRESH_BINARY)
        return binary_image

    def find_largest_connected_component(self, binary_image):
        """Find the largest connected component in a binary image."""
        num_labels, labels_im = cv2.connectedComponents(binary_image.astype(np.uint8))
        label_counts = np.bincount(labels_im.ravel())[1:]  # Exclude background
        largest_label = label_counts.argmax() + 1
        surface_mask = labels_im == largest_label
        return surface_mask

    def auto_get_bounding_box(self):
        """Get the bounding box of the largest connected component."""
        if self.surface_mask is None:
            raise ValueError("Surface mask has not been calculated.")
        
        props = regionprops(self.surface_mask.astype(int))
        return props[0].bbox if props else None
    
    def get_bounding_box(self, surface_mask):
        """"Calculates the bounding box of a specified connected component."""
        props = regionprops(surface_mask.astype(int))
        return props[0].bbox if props else None

    def modify_bounding_box(self, bounding_box, delta_round=10):
        """Modify the bounding box to adjust the cropping region."""
        delta_x = bounding_box[1]
        return (
            0, 
            max(0, bounding_box[0] - delta_round), 
            bounding_box[3] + delta_x + delta_round, 
            bounding_box[2] + 2 * delta_round
        )

    def crop_image(self, bounding_box=None):
        """Crop the image using the bounding box."""
        if bounding_box is None:
            bounding_box = self.bounding_box
        if bounding_box:
            return self.img[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
        return None

    def display_histogram_with_thresholds(self):
        """Display histogram of the image with thresholds marked."""
        plt.figure()
        plt.hist(self.img.ravel(), bins=256, range=(0, 256), density=True)
        plt.title('Histogram of Grayscale Image with Segmentation')
        plt.xlabel('Gray Level (0 to 255)')
        plt.ylabel('Normalized Pixel Count')
        
        for threshold in self.thresholds:
            plt.axvline(threshold, color='r', linewidth=2)
        
        plt.legend(['Histogram'] + [f'Threshold {i+1}' for i in range(len(self.thresholds))])
        plt.show()

    def display_image(self, img, title='Image'):
        """Helper function to display an image."""
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.show()

# Example usage:
# processor = ImageProcessor('path/to/image.bmp')
# cropped_surface = processor.crop_image()  # Now you can crop without running the entire pipeline again
# processor.display_image(cropped_surface, 'Cropped Surface')
