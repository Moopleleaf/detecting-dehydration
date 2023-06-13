# Created by: Simon Lönnqvist & Oscar Eriksson

import tensorflow as tf
from tensorflow.keras import layers
import os
import cv2
import matplotlib.pyplot as plt

# Two static variables, where INDEXLIST is the index of which file is reserved as the testing patch
# and FILEINDEX is a list of all of the files that you would want to patch/augment
FILEINDEX = []
INDEXLIST = []

# Hardcoded paths for the folders containing the cropped images (change as needed)
# path --> path where the cropped images are
# path_rect_train --> path where the augmented training images shall be put
# path_rect_test --> path where the augmented test images shall be put
path = r"../testDataSet/4WayCrop/box2-4waycrop"
path_rect_train = r"../testDataSet/expandedTests/0622Test"
path_rect_test = r"../testDataSet/expandedTests/0622Test"
files = os.listdir(path)

# Horrible function for going through each of the files and patching and augmenting them
# by randomly flipping and rotating them.
for each in FILEINDEX:
    entered = False
    for entry in INDEXLIST:
        if each == entry:
            entered = True
            fileName = os.path.splitext(files[each])
            img = cv2.cvtColor(cv2.imread(os.path.join(path, files[each])), cv2.COLOR_BGR2RGB)
            image = tf.expand_dims(img, 0)
            patches = tf.image.extract_patches(images=image, sizes=[1, 32, 32, 1], strides=[1, 24, 24, 1],
                                               rates=[1, 1, 1, 1],
                                               padding='VALID')
            dataAugmentation = tf.keras.Sequential(
                [layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.25), ])

            print("Starting patching test file ", each)

            for images in patches:
                count = 0
                for i in range(12):
                    for j in range(16):
                        count += 1
                        cv2.imwrite(
                            os.path.join(path_rect_test, '{name:}_patch_{num:}.png'.format(name=fileName[0], num=count)),
                            tf.reshape(images[i, j], shape=(32, 32, 3)).numpy().astype('uint8'))
                        for current in range(15):
                            im = tf.reshape(images[i, j], shape=(32, 32, 3)).numpy().astype('uint8')
                            augmentedImage = dataAugmentation(im)
                            cv2.imwrite(os.path.join(path_rect_test,
                                                     '{name:}_patch_{num:}_AUG{val:}.png'.format(name=fileName[0],
                                                                                                 num=count,
                                                                                                 val=current)),
                                        tf.reshape(augmentedImage, shape=(32, 32, 3)).numpy().astype('uint8'))
    if entered == False:
        fileName = os.path.splitext(files[each])
        img = cv2.cvtColor(cv2.imread(os.path.join(path, files[each])), cv2.COLOR_BGR2RGB)
        image = tf.expand_dims(img, 0)
        patches = tf.image.extract_patches(images=image, sizes=[1, 32, 32, 1], strides=[1, 24, 24, 1],
                                               rates=[1, 1, 1, 1],
                                               padding='VALID')
        dataAugmentation = tf.keras.Sequential(
                [layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.25), ])

        print("Starting patching file", each)

        for images in patches:
            count = 0
            for i in range(12):
                for j in range(16):
                    count += 1
                    cv2.imwrite(
                            os.path.join(path_rect_train, '{name:}_patch_{num:}.png'.format(name=fileName[0], num=count)),
                            tf.reshape(images[i, j], shape=(32, 32, 3)).numpy().astype('uint8'))
                    for current in range(15):
                        im = tf.reshape(images[i, j], shape=(32, 32, 3)).numpy().astype('uint8')
                        augmentedImage = dataAugmentation(im)
                        cv2.imwrite(os.path.join(path_rect_train,
                                                     '{name:}_patch_{num:}_AUG{val:}.png'.format(name=fileName[0],
                                                                                                 num=count, val=current)),
                                        tf.reshape(augmentedImage, shape=(32, 32, 3)).numpy().astype('uint8'))

