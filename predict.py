import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 自定义损失函数
def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return mse + (1 - ssim)

def load_and_predict(input_folder, model_path, output_folder):
    # Step 1: 加载模型
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})

    # Step 2: 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 3: 遍历输入文件夹中的所有 .tif 文件
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.tif'):
            img_path = os.path.join(input_folder, filename)

            # Step 4: 加载并预处理输入图像
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)  # 添加通道维度
            img = np.expand_dims(img, axis=0)   # 添加 batch 维度

            # Step 5: 进行预测
            prediction = model.predict(img)

            # Step 6: 处理和保存预测结果
            output_img = np.squeeze(prediction, axis=0)
            output_img = (output_img * 255).astype(np.uint8)
            output_path = os.path.join(output_folder, filename.replace('.tif', '_denoised.tif'))
            cv2.imwrite(output_path, output_img)

            print(f"处理完成，结果保存在 {output_path}")

if __name__ == '__main__':
    input_folder = 'G:/FIBSEM数据集/3-2_FIB_10nm_tiff/'  # 输入图像文件夹
    model_path = 'unet_model_best.h5'  # 模型文件路径
    output_folder = 'G:/FIBSEM数据集/denosing_Unet_3-2/'  # 输出文件夹

    # 调用预测函数
    load_and_predict(input_folder, model_path, output_folder)
