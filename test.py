from multiotsu_segmentation import ImageProcessor

# Đường dẫn đến ảnh cần xử lý
image_path = 'img/R.jfif'

# Khởi tạo lớp ImageProcessor
processor = ImageProcessor(image_path)

org_img = processor.load_image()

# Xử lý ảnh và cắt vùng mong muốn
thrs = processor.find_multiotsu_thresholds()
apply_thrs = processor.apply_threshold(1)
surface = processor.find_largest_connected_component(apply_thrs)

bbx = processor.get_bounding_box(surface)
mdf_bbx = processor.modify_bounding_box(bbx)
print (mdf_bbx)
cropped_image = processor.crop_image(mdf_bbx)
print (cropped_image)
# Hiển thị ảnh đã cắt
if cropped_image is not None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(cropped_image, cmap='gray')
    plt.title('Cropped Image')
    plt.show()
else:
    print("Không tìm thấy vùng lớn nhất để cắt.")


# if org_img is not None:
#     import matplotlib.pyplot as plt

#     plt.figure()
#     plt.imshow(org_img, cmap='gray')
#     plt.title('Origin Image')
#     plt.show()
# else:
#     print("Không tìm thấy vùng lớn nhất để cắt.")