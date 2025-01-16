import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
from skimage.filters import threshold_multiotsu  
from skimage.measure import label, regionprops  
  
# Load the image  
image_path = 'img/06_30-12_IMG_LOG__CAM_1_20240725_232246_587.bmp'  
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
  
# Find four thresholds using Otsu's method  
num_thresholds = 4  
thresholds = threshold_multiotsu(img, classes=num_thresholds + 1)  
  
# Display the histogram  
plt.figure()  
plt.hist(img.ravel(), bins=256, range=(0, 256), density=True)  
plt.title('Histogram of Grayscale Image with Segmentation')  
plt.xlabel('Gray Level (0 to 255)')  
plt.ylabel('Normalized Pixel Count')  
  
# Plot the thresholds  
for threshold in thresholds:  
    plt.axvline(threshold, color='r', linewidth=2)  
  
plt.legend(['Histogram'] + [f'Threshold {i+1}' for i in range(num_thresholds)])  
plt.show()  
  
# Apply adaptive thresholding  
threshold = thresholds[1] / 255.0  
_, binary_image = cv2.threshold(img, thresholds[3], 255, cv2.THRESH_BINARY)  
  
plt.figure()  
plt.imshow(binary_image, cmap='gray')  
plt.title('Binary Image')  
plt.show()  
  
# Find the largest connected component which should be the surface  
num_labels, labels_im = cv2.connectedComponents(binary_image.astype(np.uint8))  
label_counts = np.bincount(labels_im.ravel())[1:]  # Exclude background  
largest_label = label_counts.argmax() + 1  
  
surface_mask = labels_im == largest_label  
  
# Display the detected surface  
plt.figure()  
plt.imshow(img, cmap='gray')  
plt.title('Original Component Surface')  
plt.show()  
  
plt.figure()  
plt.imshow(surface_mask, cmap='gray')  
plt.title('Detected Component Surface')  
plt.show()  
  
# Get the bounding box of the surface mask  
props = regionprops(surface_mask.astype(int))  
bounding_box = props[0].bbox  
  
# Modify bounding box to crop from the left edge  
delta_x = bounding_box[1]  
delta_round = 10  
bounding_box = (0, bounding_box[0] - delta_round, bounding_box[3] + delta_x + delta_round, bounding_box[2] + 2 * delta_round)  
  
# Crop the detected surface using the bounding box  
cropped_surface = img[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]  
  
plt.figure()  
plt.imshow(cropped_surface, cmap='gray')  
plt.title('Cropped Component Surface')  
plt.show()