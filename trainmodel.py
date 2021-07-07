import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand

from getxy import get_xytest
from getxy import get_trainval
# from Subpixel import Subpixel

def create_model_form():
    upscale_factor = 4

    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = Input(shape=(44, 44, 3))
    x = Conv2D(64, 5, **conv_args)(inputs)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(32, 3, **conv_args)(x)
    x = Conv2D(3 * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam', loss='mse')

    model.summary()
    return model

def load_model():
    model = create_model_form()
    model.load_weights("model_rgb.h5")
    return model

def train():
    model = create_model_form()
    train_gen, val_gen = get_trainval()

    model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1, callbacks=[
        ModelCheckpoint('model_rgb.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    ])
    model.load_weights("model_rgb.h5")
    return model

def pred_test(test_idx, model = None):
    if model is None:
        model = load_model()
    x_test_list, y_test_list = get_xytest()

    x1_test = np.load(x_test_list[test_idx])
    x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True)
    y1_test = np.load(y_test_list[test_idx])
    y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))

    print(x1_test.shape, y1_test.shape)

    x1_test = (x1_test * 255).astype(np.uint8)
    x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
    y1_test = (y1_test * 255).astype(np.uint8)
    y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)

    x1_test = cv2.cvtColor(x1_test, cv2.COLOR_BGR2RGB)
    x1_test_resized = cv2.cvtColor(x1_test_resized, cv2.COLOR_BGR2RGB)
    y1_test = cv2.cvtColor(y1_test, cv2.COLOR_BGR2RGB)
    y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 4, 1)
    plt.title('input')
    plt.imshow(x1_test)
    plt.subplot(1, 4, 2)
    plt.title('resized')
    plt.imshow(x1_test_resized)
    plt.subplot(1, 4, 3)
    plt.title('output')
    plt.imshow(y_pred)
    plt.subplot(1, 4, 4)
    plt.title('groundtruth')
    plt.imshow(y1_test)
    plt.show()



