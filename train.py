import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os


# 注意力机制
def attention_block(x, g):
    theta_x = Conv2D(1, (2, 2), padding='same')(x)
    phi_g = Conv2D(1, (1, 1), padding='same')(g)

    # 确保 theta_x 和 phi_g 的大小一致
    phi_g = tf.image.resize(phi_g, size=tf.shape(theta_x)[1:3])

    add_xg = Add()([theta_x, phi_g])
    relu_xg = tf.keras.layers.ReLU()(add_xg)
    psi = Conv2D(1, (1, 1), activation='sigmoid')(relu_xg)
    return multiply([x, psi])


# 带注意力机制的U-Net模型
def unet_model_with_attention(input_shape):
    inputs = Input(shape=input_shape)

    # 编码路径与注意力块
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # 底部
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # 解码路径与注意力块
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    c4 = tf.image.resize(c4, size=tf.shape(u6)[1:3])
    c4 = attention_block(c4, u6)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    c3 = tf.image.resize(c3, size=tf.shape(u7)[1:3])
    c3 = attention_block(c3, u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    c2 = tf.image.resize(c2, size=tf.shape(u8)[1:3])
    c2 = attention_block(c2, u8)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    c1 = tf.image.resize(c1, size=tf.shape(u9)[1:3])
    c1 = attention_block(c1, u9)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # 输出层
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    outputs = tf.image.resize(outputs, size=input_shape[:2])

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# 自定义损失函数
def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return mse + (1 - ssim)


# 加载图像数据
def load_images_from_directory(directory, img_height, img_width, color_mode='grayscale'):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=(img_height, img_width), color_mode=color_mode)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)


# 设置图像数据路径和相关参数
noisy_dir = 'C:/Users/hugtm/Desktop/train_save2/train_save2'
clean_dir = 'C:/Users/hugtm/Desktop/save2/save2'
test_dir = 'C:/Users/hugtm/Desktop/X_test/X_test'
img_height = 1300
img_width = 2000
input_shape = (img_height, img_width, 1)
color_mode = 'grayscale'

# 加载数据
X_train_noisy = load_images_from_directory(noisy_dir, img_height, img_width, color_mode)
X_train_clean = load_images_from_directory(clean_dir, img_height, img_width, color_mode)

# 构建模型
model = unet_model_with_attention(input_shape)
model.compile(optimizer='adam', loss=custom_loss)

# 模型检查点与学习率调度
checkpoint = ModelCheckpoint('unet_model_best.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# 训练模型
epochs = 10
history = model.fit(X_train_noisy, X_train_clean, epochs=epochs, batch_size=1, validation_split=0.1,
                    callbacks=[checkpoint, early_stopping, reduce_lr])

# 测试模型
X_test_noisy = load_images_from_directory(test_dir, img_height, img_width, color_mode)
denoised_images = model.predict(X_test_noisy)

# 图像保存路径
save_dir = '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

n = 1
plt.figure(figsize=(20, 10))

# 保存每个图像
for i in range(n):
    # 噪声图像
    plt.subplot(3, n, i + 1)
    plt.imshow(X_test_noisy[i].reshape(img_height, img_width), cmap='gray')
    plt.title("Noisy")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f'noisy_image_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=300)

    # 去噪图像
    plt.subplot(3, n, i + 1 + n)
    plt.imshow(denoised_images[i].reshape(img_height, img_width), cmap='gray')
    plt.title("Denoised")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f'denoised_image_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=300)

    # 原始图像
    plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(X_train_clean[i].reshape(img_height, img_width), cmap='gray')
    plt.title("Original")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f'original_image_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=300)

plt.show()
