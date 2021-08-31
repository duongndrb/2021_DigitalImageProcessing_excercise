import cv2 
import numpy as np
from matplotlib import pyplot as plt

# # the function of inverse image
def inverseImage(gray):
    ## Convert the gray image to invert image gray
    # the input : gray scale image
    h, w = gray.shape
    mask = 255 * np.ones([h,w], dtype=np.uint8)
    inv_img = mask - gray

    return inv_img


if __name__ == "__main__":
    # Read the image
    img = cv2.imread("images/blox.jpg")

    #convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # inverse the image
    inv_img = inverseImage(gray)

    # the titles in matplotlib
    titles = ['Original Image','Gray Scale', 'Inverse Image']
    images = [img, gray, inv_img]

    for i in range (3):
        plt.subplot(1,3,i+1), plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show() 

    # #show the image
    # cv2.imshow("Original",gray)
    # cv2.imshow("Inverse Image", inv_img)
    # cv2.waitKey(0)