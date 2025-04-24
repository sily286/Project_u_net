import os
import cv2  # 用于加载图像
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_metrics(img1, img2):
    # 将图像转换为浮点类型
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 计算 PSNR
    psnr_value = psnr(img1, img2, data_range=img1.max() - img1.min())

    # 计算 SSIM
    ssim_value, _ = ssim(img1, img2, full=True, data_range=img1.max() - img1.min())

    return psnr_value, ssim_value


def compare_images(original_images_folder, predicted_images_folder):
    original_images = sorted(os.listdir(original_images_folder))
    predicted_images = sorted(os.listdir(predicted_images_folder))

    psnr_values = []
    ssim_values = []

    for i in range(10):  # 假设有10对图像
        original_img_path = os.path.join(original_images_folder, original_images[i])
        predicted_img_path = os.path.join(predicted_images_folder, predicted_images[i])

        # 读取图像
        original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
        predicted_img = cv2.imread(predicted_img_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像大小一致
        if original_img.shape != predicted_img.shape:
            print(f"Error: Image shapes do not match for {original_images[i]} and {predicted_images[i]}")
            continue

        # 计算PSNR和SSIM
        psnr_value, ssim_value = calculate_metrics(original_img, predicted_img)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

        print(f"Image {i + 1} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    # 计算平均PSNR和SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"\nAverage PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")


# 设置图像文件夹路径
original_images_folder = 'G:/download/KAIR-master/original_images'
predicted_images_folder = 'G:/download/KAIR-master/predicted_images'

# 比较图像
compare_images(original_images_folder, predicted_images_folder)
