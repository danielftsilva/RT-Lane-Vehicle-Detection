import cv2 as cv
import pyopencl as cl
import pyopencl.cltypes as cl_array
import numpy as np
import imageFormsAntigo as iF
import math

def houghGPUSetup():
    try:
        plaforms = cl.get_platforms()  # configure platform
        global plaform
        plaform = plaforms[0]

        devices = plaform.get_devices()  # configure device
        global device
        device = devices[0]

        global ctx  # set context
        ctx = cl.Context(devices)  # or dev_type=cl.device_type.ALL)
        global commQ
        commQ = cl.CommandQueue(ctx, device)  # create command queue

        file = open("hough.cl", "r")  # load file with program/kernel

        global prog  # get the kernel
        prog = cl.Program(ctx, file.read())
        prog.build()  # build the program/kernel

    except Exception as e:
        print(e)
    return plaform, device, ctx, commQ, prog

def houghSetup(angMin1, angMin2):
    # Returns accumulator
    # Recommended angles: -90, 90, 1
    thetas = np.deg2rad(np.arange(minAngle, maxAngle, angleSpacing))
    num_thetas = len(thetas)

    diag_len = np.uint32(np.ceil(np.sqrt(width * width + height * height)))  # max_dist
    rhos = np.linspace(-diag_len / 2, diag_len / 2, diag_len)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    accumulator = np.zeros((diag_len, num_thetas), dtype=np.uint32)
    return thetas, rhos, accumulator, sin_t, cos_t, num_thetas, diag_len

def houghGPU(img, imgOriginal, accumulator, filter_hLow, filter_hHigh, plaform, device, ctx, commQ, prog):
    # Convert to BGRA
    imageBGRA = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

    # Define Image object: create image and buffer objects
    imgFormat = cl.ImageFormat(
        cl.channel_order.BGRA,
        cl.channel_type.UNSIGNED_INT8)

    # Buffer In - image
    bufferIn = cl.Image(
      ctx,
        flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
        format=imgFormat,
        shape=(imageBGRA.shape[1], imageBGRA.shape[0]),  # image width, height
        pitches=(imageBGRA.strides[0], imageBGRA.strides[1]),
        hostbuf=imageBGRA.data)

    # Other parameters
    w_in = imageBGRA.shape[1]
    h_in = imageBGRA.shape[0]

    # Setup Work items and groups
    dimension = 4  # R,G,B,A
    xBlockSize = 16
    yBlockSize = 16

    xBlocksNumber = round(imageBGRA.shape[1] / xBlockSize)
    yBlocksNumber = round(imageBGRA.shape[0] / yBlockSize)

    workItemSize = (xBlockSize, yBlockSize, dimension)
    workGroupSize = (xBlocksNumber * xBlockSize, yBlocksNumber * yBlockSize, dimension)  # confirmar

    # Send parameters to device
    kernelName = prog.hough_GPU
    accBuff = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=accumulator)

    kernelName.set_arg(0, bufferIn)
    kernelName.set_arg(1, np.int32(w_in))
    kernelName.set_arg(2, np.int32(h_in))
    kernelName.set_arg(3, np.int32(filter_hLow))
    kernelName.set_arg(4, np.int32(filter_hHigh))
    kernelName.set_arg(5, accBuff)

    # Start program
    kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName, global_work_size=workGroupSize,
                                         local_work_size=workItemSize)  # execute kernel
    kernelEvent.wait()  # wait for kernel to finish

    cl.enqueue_copy(commQ, accumulator, accBuff)  # memBuff2->sum

    # Release device memory
    bufferIn.release()
    accBuff.release()

    # np.savetxt('test.csv', accumulator, delimiter=';', fmt='%d')

    # Draw line through max: for theta between 0 and 85
    max_index = np.unravel_index(accumulator.argmax(), accumulator.shape)
    rho = max_index[0]
    theta = np.deg2rad(max_index[1])
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    cv.line(imgOriginal, pt1, pt2, (255, 0, 0), 1, cv.LINE_AA)

    # 95: --> de 95 ate ao fim
    # Draw line through max: for theta between 95 and 180 accumulator[95:, ], [:, 95:]
    #accumulator = np.delete(accumulator, np.s_[:95], axis=1)
    #np.savetxt('test.csv', accumulator, delimiter=';', fmt='%d')
    #accumulator = np.delete(accumulator, np.s_[:95], axis=1)
    # max_index = np.unravel_index(accumulator[:, :95].argmax(), accumulator.shape)
    #accumulator = accumulator[:, :95]

    # max_index = np.unravel_index(accumulator[:, :95].argmax(), accumulator.shape)
    # rho = max_index[0]
    # theta = np.deg2rad(max_index[1])
    # # print("--------")
    # # print(rho)
    # # print(max_index[1])
    # a = math.cos(theta)
    # b = math.sin(theta)
    # x0 = a * rho
    # y0 = b * rho
    # pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    # pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    # cv.line(imgOriginal, pt1, pt2, (0, 255, 0), 1, cv.LINE_AA)

    return imgOriginal, pt1, pt2

