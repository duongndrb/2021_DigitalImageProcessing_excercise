import numpy as np
import cv2
import math

def rgb_to_gray(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R+G+B)
    grayImage = img
    for i in range(3):
        grayImage[:, :, i] = Avg

    return grayImage


image = cv2.imread("lenna.png")
grayImage = rgb_to_gray(image)

cv2.imshow('RGB', image)
cv2.imshow('Gray', grayImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
