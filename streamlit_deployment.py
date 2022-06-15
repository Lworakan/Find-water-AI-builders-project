import streamlit as st
import pandas as pd
import numpy as np
import os
import os.path

from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL

from keras.models import load_model

from PIL import Image

def load_image(image_file):
	img = Image.open(image_file)
	return img

st.title('Water LineFinder Detection!')

image_file = st.file_uploader("อัพโหลดรูปภาพ", type=["png","jpg","jpeg"])


if image_file is not None:
    file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
    st.image(load_image(image_file),width=250)
    images_dir = f"Images/{image_file.name}"

    def load_img_with_mask(image_path, images_dir: str = 'Images', masks_dir: str = 'Masks',images_extension: str = 'jpg', masks_extension: str = 'jpg') -> dict:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        mask_filename = tf.strings.regex_replace(image_path, images_dir, masks_dir)
        mask_filename = tf.strings.regex_replace(mask_filename, images_extension, masks_extension)
        mask = tf.io.read_file(mask_filename)
        mask = tf.image.decode_image(mask, channels=1, expand_animations = False)
        return (image, mask)

    def resize_images(images, masks, max_image_size=1500):
        shape = tf.shape(images)
        scale = (tf.reduce_max(shape) // max_image_size) + 1
        target_height, target_width = shape[-3] // scale, shape[-2] // scale
        images = tf.cast(images, tf.float32)
        masks = tf.cast(masks, tf.float32)
        if scale != 1:
            images = tf.image.resize(images, (target_height, target_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            masks = tf.image.resize(masks, (target_height, target_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return (images, masks)

    def scale_values(images, masks, mask_split_threshold = 128):
        images = tf.math.divide(images, 255)
        masks = tf.where(masks > mask_split_threshold, 1, 0)
        return (images, masks)

    def pad_images(images, masks, pad_mul=16, offset=0):
        shape = tf.shape(images)
        height, width = shape[-3], shape[-2]
        target_height = height + tf.math.floormod(tf.math.negative(height), pad_mul)
        target_width = width + tf.math.floormod(tf.math.negative(width), pad_mul)
        images = tf.image.pad_to_bounding_box(images, offset, offset, target_height, target_width)
        masks = tf.cast(tf.image.pad_to_bounding_box(masks, offset, offset, target_height, target_width), tf.uint8)
        return (images, masks)

    batch_size = 32
    test_set_size = 300
    validation_set_size = 250

    dataset = tf.data.Dataset.list_files(images_dir, seed=42)

    test_dataset = dataset.take(test_set_size)
    dataset = dataset.skip(test_set_size)
    test_dataset = test_dataset.map(load_img_with_mask)
    test_dataset = test_dataset.map(scale_values)
    test_dataset = test_dataset.shuffle(20)
    test_dataset = test_dataset.map(lambda img, mask: resize_images(img, mask, max_image_size=2500))
    test_dataset = test_dataset.map(pad_images)
    test_dataset = test_dataset.batch(1).prefetch(5)


    validation_dataset = dataset.take(validation_set_size)
    train_dataset = dataset.skip(validation_set_size)
    validation_dataset = validation_dataset.map(load_img_with_mask)
    validation_dataset = validation_dataset.map(scale_values)
    validation_dataset = validation_dataset.shuffle(20)
    validation_dataset = validation_dataset.map(resize_images)
    validation_dataset = validation_dataset.map(pad_images)
    validation_dataset = validation_dataset.batch(1).prefetch(5)

    train_dataset = train_dataset.map(load_img_with_mask)
    train_dataset = train_dataset.map(scale_values)
    train_dataset = train_dataset.shuffle(20)
    train_dataset = train_dataset.map(resize_images)
    train_dataset = train_dataset.map(pad_images)
    train_dataset = train_dataset.batch(1).prefetch(5)

    model=load_model('attempt01.h5')
    model.summary()

    for ele in test_dataset.take(1):
        image, y_true = ele
        prediction = model.predict(image)[0]
        prediction = tf.where(prediction > 0.25, 255, 0)

    plt.imshow(prediction)

    figure = plt.gcf()
    figure.set_size_inches(12, 8)
    plt.savefig("predicted.jpg", bbox_inches='tight', pad_inches=0)
    
if st.button('Click here to Start Prediction'):
    st.image(load_image("predicted.jpg"),width=250)