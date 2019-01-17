# coding:utf-8
import cv2
import time
from ctypes import *
import numpy as np


class Image(Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("srcWidth", c_int),
        ("srcHeight", c_int),
        ("dstWidth", c_int),
        ("dstHeight", c_int),
        ("cu_dst", POINTER(c_float)),
        ("cu_src", POINTER(c_ubyte)),
        ("cu_dst_resize", POINTER(c_ubyte))
    ]


class BBOX(Structure):
    _fields_ = [
        ("nums", c_int),
        ("cols", c_int),
        ("result", POINTER(c_float))
    ]


_so_path = './images.so'
images_lib = CDLL(_so_path)

init_cuda_memory = images_lib.init_cuda_memory
init_cuda_memory.argtypes = (c_int,c_int,c_int,c_int)
init_cuda_memory.restype = POINTER(Image)

resize_cu = images_lib.resize_cu
resize_cu.argtypes = [POINTER(c_ubyte),POINTER(Image)]
resize_cu.restype = POINTER(Image)

do_nms_sort = images_lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(c_float),POINTER(BBOX),c_int,c_float]
do_nms_sort.restype = None


def test_nms():
    # nms by C and python invoke.
    data = np.array([[603.8855, 215.5790,  20.1108,  54.5654,   0.9998,   1.0000],
        [951.5989, 544.6505,  26.7356,  75.6343,   1.0000,   1.0000],
        [830.8037, 695.8022,  27.9502,  72.3551,   1.0000,   1.0000],
        [830.8037, 695.8022, 27.9502, 72.3551, 1.0000, 1.0000],
        [604.3386, 215.3251,  19.3738,  54.6253,   1.0000,   1.0000]], dtype=np.float32)
    time1 = time.time()
    bbox = BBOX()
    bbox.nums = data.shape[0]
    bbox.cols = data.shape[1]
    a = np.zeros(data.shape, dtype=np.float32)
    bbox.result = a.ctypes.data_as(POINTER(c_float))
    do_nms_sort(data.ctypes.data_as(POINTER(c_float)), bbox, 1, 0.5)
    bbox = np.ctypeslib.as_array(bbox.result, (bbox.nums, 6))
    time2 = time.time()
    print (time2 - time1)*1000
    print data
    print '============================='
    print bbox


def test_images_cuda():
    im = cv2.imread('test.png')
    srcHeight,srcWidth,channel = im.shape
    dstHeight = 300
    dstWidth = 300
    struct_im = images_lib.init_cuda_memory(srcWidth, srcHeight, dstWidth, dstHeight)
    time1 = time.time()
    img = resize_cu(im.ctypes.data_as(POINTER(c_ubyte)), struct_im)
    image = np.ctypeslib.as_array(img.contents.data, (1, 3, dstHeight, dstWidth))
    time2 = time.time()
    print (time2 - time1)*1000
    print image.shape


if __name__ == '__main__':
    # test_nms()
    test_images_cuda()
























