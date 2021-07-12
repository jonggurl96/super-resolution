import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand
import PIL
from PIL import Image

from getxy import get_xytest
from getxy import get_trainval
from getxy import get_trainval_yuv

def create_model_form_rgb(upscale_factor = 4):
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

    # model.summary()
    return model

def create_model_form_yuv(upscale_factor = 4):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = Input(shape=(44, 44, 1))
    x = Conv2D(64, 5, **conv_args)(inputs)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(32, 3, **conv_args)(x)
    x = Conv2D(upscale_factor ** 2, 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam', loss='mse')

    # model.summary()
    return model

def load_model(channels):
    if channels == "RGB":
        model = create_model_form_rgb()
        model.load_weights("model_rgb.h5")
    elif channels == "YUV":
        model = create_model_form_yuv()
        model.load_weights("model_yuv.h5")
        
    return model


def train_rgb():
    model = create_model_form_rgb()
    checkpoint = 'model_rgb.h5'
    
    train_gen, val_gen = get_trainval()

    model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1, callbacks=[
        ModelCheckpoint(checkpoint, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    ])
    model.load_weights(checkpoint)
    return model

def train_yuv():
    model = create_model_form_yuv()
    checkpoint = 'model_yuv.h5'
    train_gen, val_gen = get_trainval_yuv()

    print("input / target data yuv complete...")

    model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1, callbacks=[
        ModelCheckpoint(checkpoint, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    ])
    model.load_weights(checkpoint)
    return model

def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    img = Image.fromarray(img)
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = tf.keras.preprocessing.image.img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    out_img = np.array(out_img)
    out_img = (out_img / 255.0).astype(np.float32)
    return out_img

def Xy_input_output_split(test_idx, channel, model):
    x_test_list, y_test_list = get_xytest()

    x1_test = np.load(x_test_list[test_idx])
    x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True)
    y1_test = np.load(y_test_list[test_idx])
    if channel == "RGB":
        y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))
    elif channel == "YUV":
        y_pred = upscale_image(model, (x1_test*255.0).astype(np.uint8))
        y_pred = np.array(y_pred)


    x1_test = (x1_test * 255).astype(np.uint8)
    x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
    y1_test = (y1_test * 255).astype(np.uint8)
    y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)

    x1_test = cv2.cvtColor(x1_test, cv2.COLOR_BGR2RGB)
    x1_test_resized = cv2.cvtColor(x1_test_resized, cv2.COLOR_BGR2RGB)
    y1_test = cv2.cvtColor(y1_test, cv2.COLOR_BGR2RGB)
    y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)

    return x1_test, x1_test_resized, y_pred, y1_test

def pred_test(test_idx, channel, model):
    x1_test, x1_test_resized, y_pred, y1_test = Xy_input_output_split(
        test_idx, channel, model
    )

    fig = plt.figure()
    fig.suptitle(channel + " results")
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

def resized_output_psnr_ssim(test_idx, channel, model):
    x1_test, x1_test_resized, y_pred, y1_test = Xy_input_output_split(
        test_idx, channel, model
    )
    y_pred = (y_pred * 255).astype(np.uint8)

    resized_psnr = tf.image.psnr(y1_test, x1_test_resized, max_val=255)
    pred_psnr = tf.image.psnr(y1_test, y_pred, max_val=255)

    resized_ssim = tf.image.ssim(y1_test, x1_test_resized, max_val=255)
    pred_ssim = tf.image.ssim(y1_test, y_pred, max_val=255)

    return resized_psnr, pred_psnr, resized_ssim, pred_ssim

def plot_resized_pred_truth_psnr_ssim(test_idx, channel, model):
    x1_test, x1_test_resized, y_pred, y1_test = Xy_input_output_split(
        test_idx, channel, model
    )
    y_pred = (y_pred * 255).astype(np.uint8)

    resized_psnr = tf.image.psnr(y1_test, x1_test_resized, max_val=255)
    pred_psnr = tf.image.psnr(y1_test, y_pred, max_val=255)

    resized_ssim = tf.image.ssim(y1_test, x1_test_resized, max_val=255)
    pred_ssim = tf.image.ssim(y1_test, y_pred, max_val=255)

    fig = plt.figure()
    fig.suptitle(channel + " results")
    plt.subplot(2, 3, 1)
    plt.title('input')
    plt.imshow(x1_test)
    plt.subplot(2, 3, 2)
    plt.title('resized')
    plt.imshow(x1_test_resized)
    plt.subplot(2, 3, 3)
    plt.title('output')
    plt.imshow(y_pred)
    plt.subplot(2, 3, 4)
    plt.title('groundtruth')
    plt.imshow(y1_test)
    plt.subplot(2, 3, 5)
    plt.title('PSNR with Groundtruth')
    plt.bar(
        ['Resized', 'Predicted'],
        [resized_psnr, pred_psnr])
    plt.subplot(2, 3, 6)
    plt.title('SSIM with Groundtruth')
    plt.bar(
        ['Resized', 'Predicted'],
        [resized_ssim, pred_ssim]
    )
    plt.show()