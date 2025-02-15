import cv2 as cv
import pyopencl as cl
import pyopencl.cltypes as cl_array
import numpy as np
import math as math

import aula9_sobel as sobel
import finalProject_params as param
import imageFormsAntigo as iF
import finalProject_hough as hough
import finalProject_classifier as cars

# Setup GPU programs and other variables
platform_sobel, device_sobel, ctx_sobel, commQ_sobel, prog_sobel = sobel.sobelSetup()
platform_hough, device_hough, ctx_hough, commQ_hough, prog_hough = hough.houghGPUSetup()

# Open video (or image)
if not param.video:
    file = cv.imread(param.pathname + param.filename)
    iF.showSideBySideImages(file, file, title="Image", BGR1=True, BGR2=True)
else:
    # Read file
    file = cv.VideoCapture(param.pathname + param.filename)

    while file.isOpened():
        ret, frame = file.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Resize for easier processing
        frame = cv.resize(frame, (param.width, param.height))

        # Lane pre-processing: Sobel Operator
        frame_sobel = sobel.sobelGPU(frame, param.sobel_t1, param.sobel_t2, platform_sobel, device_sobel, ctx_sobel, commQ_sobel, prog_sobel)

        # Hough transform
        accumulator = np.zeros((param.diag_len, param.num_thetas), dtype=np.uint32)
        frame, pt1, pt2 = hough.houghGPU(frame_sobel, frame, accumulator, param.filter_hLow, param.filter_hHigh, platform_hough, device_hough, ctx_hough, commQ_hough, prog_hough)

        # Vehicle detection
        frame = cars.detectDrawCars(frame, param.cars_xml, pt1, pt2, param.scaleFactor, param.minSize)

        # Visualize
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break
    file.release()
    cv.destroyAllWindows()




