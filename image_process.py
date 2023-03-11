import cv2 as cv
import numpy
import pandas as pd
import os

path = os.getcwd()
img = cv.imread(path + "\\images\\images\\n02111277\\n02111277_19.JPEG")
img = cv.resize(img, (128, 128))
# cv.imshow("img", img)
# cv.waitKey(0)
# cv.destroyAllWindows()


def HOG(imags,resize = (128,128)):

    winSize = (128, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    padding = (1, 1)

    finsh_hog = []

    for img in imags:
        img = cv.resize(img, resize)
        hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        hogdescriptor = hog.compute(img, padding)
        finsh_hog.append(hogdescriptor)
    return finsh_hog


if __name__ == '__main__':
    pass
    # img = cv.imread(path + "\\images\\images\\n02111277\\n02111277_19.JPEG")
    # # createTrainingInstances()
    # img = cv.resize(img, (128, 128))
    # # cv.imshow("img", img)
    # # cv.waitKey(0)
    # # cv.destroyAllWindows()
    #
    # winSize = (128, 128)
    # blockSize = (16, 16)
    # blockStride = (8, 8)
    # cellSize = (8, 8)
    # nbins = 9
    # padding = (1, 1)
    #
    # hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    # hogdescriptor = hog.compute(img, padding)
    # print(hogdescriptor.shape)



