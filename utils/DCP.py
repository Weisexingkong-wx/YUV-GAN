#!/usr/bin/env python
# encoding: utf-8

'''
Using Dark Channel Prior for thin cloud removal
'''

from PIL import Image
import numpy as np
import os
import cv2
import glob

# Dark Channel Prior
def haze_removal(image, windowSize=24, w0=0.6, t0=0.1):

    darkImage = image.min(axis=2)
    maxDarkChannel = darkImage.max()
    darkImage = darkImage.astype(np.double)

    t = 1 - w0 * (darkImage / maxDarkChannel)
    T = t * 255
    T.dtype = 'uint8'

    t[t < t0] = t0

    J = image
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * maxDarkChannel) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * maxDarkChannel) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * maxDarkChannel) / t
    result = Image.fromarray(J)
    result = np.array(result)
    return result

# load cloudy images in directory_name for haze_removal
def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        print(filename)
        img = cv2.imread(directory_name + "/" + filename)
        img = haze_removal(img, windowSize=24, w0=0.6, t0=0.1)

        #####save images#########
        cv2.imwrite("E:\Codes\ThinCloudRemove\dataset\\test\DCP_result" + "/" + filename, img)

    return filename

read_directory("E:\Codes\ThinCloudRemove\dataset\\test\cloud")

