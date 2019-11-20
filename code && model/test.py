#!/usr/bin/env python

"""Description:
The test.py is to evaluate your model on the test images.
***Please make sure this file work properly in your final submission***

©2019 Created by Yiming Peng and Bing Xue
"""
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array, ImageDataGenerator

# You need to install "imutils" lib by the following command:
#               pip install imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse

import numpy as np
import random
import tensorflow as tf

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)


def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default = "data/test",
                      help = "path to test_data_dir")
    args = vars(args.parse_args())
    return args

DATA_PATH = "data/test/"


BATCH_SIZE= 9 # how many images process at one time, the less the better?  from 16 to 9

def load_images(path):
    datagen = ImageDataGenerator(rotation_range=40, #data pre process
        width_shift_range=0.2,
        height_shift_range=0.2,   # unify the format of images
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2)


    test = datagen.flow_from_directory(path, target_size= (300,300),
                                                      classes=['cherry', 'strawberry', 'tomato'], batch_size = BATCH_SIZE ,
                                                      class_mode='categorical')

    return test


# def convert_img_to_array(images, labels):
#     # Convert to numpy and do constant normalize
#     X_test = np.array(images, dtype = "float") / 255.0
#     y_test = np.array(labels)
#
#     # Binarize the labels
#     lb = LabelBinarizer()
#     y_test = lb.fit_transform(y_test)
#
#     return X_test, y_test


def preprocess_data(X):
    """
    Pre-process the test data.
    :param X: the original data
    :return: the preprocess data
    """
    # NOTE: # If you have conducted any pre-processing on the image,
    # please implement this function to apply onto test images.
    return X


def evaluate(test):
    """
    Evaluation on test images
    ******Please do not change this function******
    :param X_test: test images
    :param y_test: test labels
    :return: the accuracy
    """
    # batch size is 16 for evaluation
    batch_size = 16

    # Load Model
    model = load_model('model/CNNF.h5')
    return model.evaluate_generator(test)


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["test_data_dir"]

    # Image size, please define according to your settings when training your model.
    image_size = (300, 300)

    # Load images
    test = load_images(DATA_PATH)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    #X_test, y_test = convert_img_to_array(images, labels)

    # Preprocess data.
    # ***If you have any preprocess, please re-implement the function "preprocess_data"; otherwise, you can skip this***
    #X_test = preprocess_data(X_test)

    # Evaluation, please make sure that your training model uses "accuracy" as metrics, i.e., metrics=['accuracy']
    loss, accuracy = evaluate(test)
    print("loss={}, accuracy={}".format(loss, accuracy))
