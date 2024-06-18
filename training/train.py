import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

image_dir = "skins/"
image_size = (64, 64)
shrunk_size = (16, 16)

def add_noise_to_colored_pixels(img_array, noise_strength=20):
    is_colored = np.logical_and(img_array[:, :, :3] > 0, img_array[:, :, :3] < 255).all(axis=-1)

    noise = np.random.randint(-noise_strength, noise_strength+1, size=img_array.shape)
    noisy_img_array = img_array + noise

    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

    noisy_img_array[~is_colored] = img_array[~is_colored]

    return noisy_img_array

def load_minecraft_skins():
    skin_images = []
    input_images = []
    for filename in os.listdir(image_dir):
        if not filename.endswith(".png"):
            continue
        try:
            img = Image.open(os.path.join(image_dir, filename)).convert('RGBA')
        except Exception as e:
            continue

        

        img = img.resize(image_size)
        input_img = img.resize(shrunk_size).resize(image_size)

        img_array = add_noise_to_colored_pixels(np.array(img)) / 255.0
        input_img_array = np.array(img) / 255.0

        skin_images.append(img_array)
        input_images.append(input_img_array)

    skin_images = np.array(skin_images)
    input_images = np.array(input_images)
    return skin_images, input_images

skin_images, input_images = load_minecraft_skins()

split_idx = int(0.8 * len(input_images))
x_train, x_val = input_images[:split_idx], input_images[split_idx:]
y_train, y_val = skin_images[:split_idx], skin_images[split_idx:]

print(f"x_train shape: {x_train.shape}, type: {type(x_train)}")
print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")
print(f"x_val shape: {x_val.shape}, type: {type(x_val)}")
print(f"y_val shape: {y_val.shape}, type: {type(y_val)}")

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3), activation='sigmoid', padding='same'))
    return model

model = build_model(image_size + (4,))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('minecraft-skin-optimizer.h5')
