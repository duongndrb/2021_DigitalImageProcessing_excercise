import cv2
import numpy as np
from matplotlib import pyplot as plt

def sobelEgde(gray_image):

    #the raw image
    rows, columns = gray_image.shape
    #init the zeros matrix with expand size
    gray_matrix = np.zeros([rows+2, columns+2])
    # the raw image same to new expand image
    gray_matrix[1:1+rows, 1:1+columns] = gray_image

    #create the sobel operator
    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.intc)
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],dtype=np.intc)   

    # Init the output image
    sobel_filtered_image = np.zeros([rows, columns],dtype=np.uint8)

    #Check all elements
    for i in range(rows):
        for j in range(columns):
            
            gx = np.sum(np.multiply(gray_matrix[i:i+3,j:j+3],Gx))
            gy = np.sum(np.multiply(gray_matrix[i:i+3,j:j+3],Gy))
            sobel_filtered_image[i,j]= np.abs(gx)+ np.abs(gy)
    
    return sobel_filtered_image

def laplacianEdge(gray_image,type = "1"):
    #Create the Laplacian operator 
    laplacian_matrix_1 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=np.intc)
    laplacian_matrix_2 = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],dtype=np.intc)
    laplacian_matrix_3 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=np.intc)

    #the raw image
    rows, columns = gray_image.shape
    #init the zeros matrix with expand size
    gray_matrix = np.zeros([rows+2, columns+2])
    # the raw image same to new expand image
    gray_matrix[1:1+rows, 1:1+columns] = gray_image

    # Init the output image
    laplacian_image = np.zeros([rows, columns],dtype=np.uint8)

    #Check all elements
    for i in range(rows):
        for j in range(columns):
            if(type== 1):
                laplacian_image[i,j]= np.sum(np.multiply(gray_matrix[i:i+3,j:j+3],laplacian_matrix_1))
            elif(type == 2):
                laplacian_image[i,j]= np.sum(np.multiply(gray_matrix[i:i+3,j:j+3],laplacian_matrix_2))
            else:
                laplacian_image[i,j]= np.sum(np.multiply(gray_matrix[i:i+3,j:j+3],laplacian_matrix_3))
    
    return np.abs(laplacian_image)




if __name__ == "__main__":
    #read image 
    img = cv2.imread("images/smarties.png")

    #convert image to gray
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel = sobelEgde(gray_image)

    laplacian_1 = laplacianEdge(gray_image,type=1)
    laplacian_2 = laplacianEdge(gray_image,type=2)
    laplacian_3 = laplacianEdge(gray_image,type=3)


    # the titles in matplotlib
    titles = ['Original Image','Gray Scale', 'Sobel', 'Laplacian 1', 'Laplacian 2', 'Laplacian 3']
    images = [img, gray_image, sobel, laplacian_1 ,laplacian_2,laplacian_3]

    for i in range (6):
        plt.subplot(2,3,i+1), plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show() 
