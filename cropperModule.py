# Created by: Simon Lönnqvist & Oscar Eriksson

import os
import cv2
import matplotlib.pyplot as plt

# Path to the folders for where the uncropped image is and where the 4 way cropped
# image shall be stored.
# path --> path to the uncropped image
# path_rect --> path where the 4 way cropped images shall be stored
path = r"../testDataSet/croppedIMG IR/box2-crop-IR"
path_rect = r"../testDataSet/4WayCropIR/box2-4wayIR"

print("Cropping photos...")

# Dimenstions for the crop (change as needed/wanted)
widthCrop = 423
heightCrop = 310

# Function for going through and cropping all of images into 4 smaller images
for file in os.listdir(path):
    img = (cv2.cvtColor(cv2.imread(os.path.join(path, file)), cv2.COLOR_BGR2RGB))
    startHeightCrop = 0
    count = 0
    for i in range(2):
        startWidthCrop = 0
        startHeightCrop = heightCrop * i
        for j in range(2):
            img_crop = img[startHeightCrop:heightCrop*(i+1), startWidthCrop:widthCrop*(j+1)]
            startWidthCrop = widthCrop*(j+1)
            count += 1
            plt.imsave(os.path.join(path_rect,'{name:}_subCrop_{num:}.png'.format(name=file, num=count)), img_crop)

print("Done !")