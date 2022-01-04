import cv2
import numpy as np
#--------------------------------------
def pre_processing(data):
    print("-----PreProcessing-----")
    edges = list()
    for index in range(len(data)):
        data[index][0] = binarize(data[index][0])
        data[index][0] = check_background(data[index][0])
        data[index][0] = ero_dil(data[index][0])
        edges.append(edge_detection(data[index][0]))
    return data , edges
#--------------------------------------
def binarize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _,img = cv2.threshold(img, 150, 255, cv2.THRESH_OTSU)
    return img
#--------------------------------------
def check_background(img):
    backgrounnd = img[0][0]
    if backgrounnd == 0:
        img = cv2.bitwise_not(img)
    return img
#--------------------------------------
def ero_dil(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    return img
#--------------------------------------
def edge_detection(img):
    img = cv2.Canny(img,50,150,apertureSize = 3)
    return img
    