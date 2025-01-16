from multiotsu_segmentation import ImageProcessor

# Path to the image to be processed
image_path = 'images/bolt.jfif'

# Initialize the ImageProcessor class
processor = ImageProcessor(image_path)

org_img = processor.load_image()

# Process the image and crop the desired region
thrs = processor.find_multiotsu_thresholds()
apply_thrs = processor.apply_threshold(1)
surface = processor.find_largest_connected_component(apply_thrs)

bbx = processor.get_bounding_box(surface)
mdf_bbx = processor.modify_bounding_box(bbx)
print(mdf_bbx)
cropped_image = processor.crop_image(mdf_bbx)
print(cropped_image)

# Display the origin image
if org_img is not None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(org_img, cmap='gray')
    plt.title('Origin Image')
    plt.show()
else:
    print("The origin image could not be found.")



# Display the cropped image
if cropped_image is not None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(cropped_image, cmap='gray')
    plt.title('Cropped Image')
    plt.show()
else:
    print("The largest region for cropping could not be found.")
