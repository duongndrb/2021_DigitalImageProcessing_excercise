import cv2 as cv
import numpy as np

img = cv.imread('lenna.png')
Pgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
col, row = Pgray.shape

# mean filter
temp_mean_pic = np.ones((col+2, row+2), np.uint8)
for i in range(0, col+2):
    for j in range(0, row+2):
        if i == 0 or i == col+1 or j == 0 or j == row+1:
            temp_mean_pic[i][j] = 1
        else:
            temp_mean_pic[i][j] = Pgray[i-1][j-1]
mean_pic = np.ones((col, row), np.uint8)
for i in range(1, col+1):
    for j in range(1, row+1):
        mean_pic[i-1][j-1] = (
            temp_mean_pic[i-1][j-1]/9 +
            temp_mean_pic[i-1][j]/9 +
            temp_mean_pic[i-1][j+1]/9 +
            temp_mean_pic[i][j-1]/9 +
            temp_mean_pic[i][j]/9 +
            temp_mean_pic[i][j+1]/9 +
            temp_mean_pic[i+1][j-1]/9 +
            temp_mean_pic[i+1][j]/9 +
            temp_mean_pic[i+1][j+1]/9
        )

# median filter
temp_median_pic = np.ones((col+2, row+2), np.uint8)
for i in range(0, col+2):
    for j in range(0, row+2):
        if i == 0 or i == col+1 or j == 0 or j == row+1:
            temp_median_pic[i][j] = 1
        else:
            temp_median_pic[i][j] = Pgray[i-1][j-1]
median_pic = np.ones((col, row), np.uint8)
for i in range(1, col+1):
    for j in range(1, row+1):
        tmpp = []
        for k in range(i-1, i+2):
            for l in range(j-1, j+2):
                tmpp.append(temp_median_pic[k][l])

        matrix_ = np.sort(tmpp)
        median_pic[i-1][j-1] = matrix_[4]

# show Mean, Median, Gray
cv.imshow('Mean', mean_pic)
cv.imshow('Median', median_pic)
cv.imshow('Origin Gray', Pgray)

# check image
# print(Pgray)
# print(median_pic)
# print(mean_pic)

cv.waitKey(0)
cv.destroyAllWindows()