import cv2
from matplotlib import pyplot as plt
import numpy as np


def histogram(gray_image):
    # calculate the histogra of gray image:
    hist = np.zeros([256], dtype=np.uint32)
    #the shape information of image
    rows, columns = gray_image.shape
    for i in range(rows):
        for j in range(columns):
            hist[gray_image[i,j]] += 1
    return hist

def cumsumCalculate(hist):
    # create an array that represents the cumulative distributive function of the histogram
    cumsum = np.zeros(256,dtype= np.uint32)
    cumsum[0]= hist[0]
    for i in range (1,hist.size):
        cumsum[i] = cumsum[i-1] + hist[i]
    return cumsum

def mappingHistogram(cumsum,img_size):
    # create a mapping when each old colour value is mapped to a new one between 0 and 255
    mapping = np.zeros(256, dtype=np.uint32)
    gray_levels = 256
    height = img_size[0]
    weight = img_size[1]
    for i in range (gray_levels):
        mapping[i] = int((gray_levels*cumsum[i])/(height*weight))
    return mapping

def histogramEqualization(gray_image, mapping):
    # Change the mapping to our image
    h , w = gray_image.shape
    output = np.zeros([h,w], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            output[i,j] = mapping[gray_image[i,j]]
    return output

def showHistogram(hist1, title1 = "", hist2= None, title2=None):
    x_axis = np.arange(256)
    figure = plt.figure()
    if hist2 is None:
        plt.bar(x_axis, hist1)
        plt.title(title1)
    else:
        # Plot histogram of hist1
        figure.add_subplot(1, 2, 1)
        plt.bar(x_axis, hist1)
        plt.title(title1)

        # Plot histogram of hist2
        figure.add_subplot(1, 2, 2)
        plt.bar(x_axis, hist2)
        plt.title(title2)
    plt.show()

if __name__ == "__main__":
    #the colour image
    img = cv2.imread("images/Blender_Suzanne1.jpg")
    #convert to the grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = histogram(gray_image)
    cumsum = cumsumCalculate(hist)

    mapping = mappingHistogram(cumsum, gray_image.shape)

    equa_img = histogramEqualization(gray_image, mapping)

    equa_hist = histogram(equa_img)

    titles = ['Original','Equalization']
    images = [gray_image, equa_img]

    for i in range(2):
        plt.subplot(1,2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


    #show histogram
    showHistogram(hist, title1="Original image", hist2= equa_hist, title2="Histogram Equalization")

    
    
