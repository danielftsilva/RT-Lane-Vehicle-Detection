import cv2 as cv
import pyopencl as cl
import numpy as np
import imageFormsAntigo as iF
import math

# Image and file parameters
video = True                    # video or image
pathname = "Aux Files\\"          # image pathname
filename = "video1.MTS"         # image filename

width = 840                     # image resize width
height = 440                    # image resize height
diag_len = np.uint32(round(np.sqrt(width * width + height * height)))

# Sobel+threshold parameters
sobel_t1 = 10                       # good: 10,50
sobel_t2 = 50

# Hough parameters
num_thetas = 180                    # theta resolution. Must be integer
filter_hLow = 100                   # good: 100, 130
filter_hHigh = 130                  # area to apply the Hough transform. It's useful to filter out the hood

# Vehicle detection
cars_xml = 1                        # which cars_xml file to use: cars_xml (1), cars2_xml (2), cars3_xml (3)
scaleFactor = 1.4                   # classifier parameters
minSize = (24, 24)

