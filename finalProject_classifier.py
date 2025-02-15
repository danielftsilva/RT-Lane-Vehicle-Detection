import os
import cv2 as cv
import pyopencl as cl
import pyopencl.cltypes as cl_array
import numpy as np
import math as math

import aula9_sobel as sobel
import finalProject_params as param
import imageFormsAntigo as iF
import finalProject_hough as hough


# Detects car and draws them (color based on Hough Line (pt1,pt2).
def detectDrawCars(img, cars_xml, pt1, pt2, scaleFactor, minSize):
    # Get pre-trained classifier
    try:
        base_path = r'your_path_here'
        if cars_xml == 1:
            classifier = cv.CascadeClassifier(os.path.join(base_path, "cars.xml"))
        elif cars_xml == 2:
            classifier = cv.CascadeClassifier(os.path.join(base_path, "cars2.xml"))
        elif cars_xml == 3:
            classifier = cv.CascadeClassifier(os.path.join(base_path, "cars3.xml"))
        if classifier.empty():
            raise IOError("Failed to load classifier xml file.")
    except Exception as e:
        print(f"Classifier: {e}")
        return img

    # Convert image to gray
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply classifier
    cars = classifier.detectMultiScale(imgGray,
                                       scaleFactor=scaleFactor,
                                       minSize=minSize,
                                       maxSize=(imgGray.shape[1] // 2, imgGray.shape[0] // 2))
    # Cycle through cars list
    for (x, y, w, h) in cars:
        rect_center = (int((x + x + w) / 2), int((y + y + h) / 2))
        cross_product = ((pt2[0] - pt1[0]) * (rect_center[1] - pt1[1]) - (pt2[1] - pt1[1]) * (rect_center[0] - pt1[0]))

        # If right of the Hough Line: draw red rectangle. Else draw green
        if cross_product > 0:
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img