import cv2
import numpy as np
from skimage.measure import label

#--------------------------------------
def get_features(data,edges):
    print("-----Extracting Features-----")
    features = list()
    for index in range(len(data)):
        img_feature = list()
        img_feature.append(get_hist(data[index][0]))
        img_feature.append(HVSL(edges[index]))
        img_feature.append(num_pix_per_col(data[index][0]))
        img_feature.append(num_pix_per_row(data[index][0]))
        img_feature += [*oreinatation(data[index][0])]
        img_feature.append(count_cc(data[index][0]))
        features.append(img_feature)
    return features
#--------------------------------------
def get_hist(img):
    histg = cv2.calcHist([img],[0],None,[256],[0,256]) 
    return (round((histg[0][0]/(histg[-1][0]+histg[0][0]))*100))
#--------------------------------------
def HVSL(img):
    # Number of Horizontal and vertical lines
    lines = cv2.HoughLinesP(image=img,rho=1,theta=np.pi/2,threshold=int((img.shape[1])/3))
    L = 0
    if lines is not None:
        L = len(lines)
    return L
#--------------------------------------
def num_pix_per_col(img):
    freq = np.sum(img, axis=0)
    zero_idx = np.where(freq != 0)[0]
    if zero_idx.shape[0] == 1:
        return freq[zero_idx[0]]
    else:
        L = zero_idx[0]
        R = freq.shape[-1]
        avg = np.sum(freq[L : R+1]) / (R - L)
        return avg
#--------------------------------------   
def num_pix_per_row(img):
    freq = np.sum(img, axis=1)
    zero_idx = np.where(freq != 0)[0]
    if zero_idx.shape[0] == 1:
        return freq[zero_idx[0]]
    else:
        L = 0
        R = 1
        if zero_idx.shape[0]: L = zero_idx[0]
        if zero_idx.shape[0]: R = freq.shape[-1]
        avg = np.sum(freq[L : R+1]) / (R - L)
        return avg
#--------------------------------------
def oreinatation(myImg):    
    sobel_x = np.array([[ -1, 0, 1], 
                        [ -2, 0, 2], 
                        [ -1, 0, 1]])

    sobel_y = np.array([[ -1, -2, -1], 
                        [ 0, 0, 0], 
                        [ 1, 2, 1]])

    ang_category_count = 20
    count_ang = np.zeros(ang_category_count + 1)

    filtered_blurred_x = cv2.filter2D(myImg, cv2.CV_32F, sobel_x)
    filtered_blurred_y = cv2.filter2D(myImg, cv2.CV_32F, sobel_y)

    orien = cv2.phase(np.array(filtered_blurred_x, np.float32),
                    np.array(filtered_blurred_y, dtype=np.float32), 
                    angleInDegrees=True)
    orien = np.array(orien)
    
    step = 360/ang_category_count
    ang = step
    while(ang <= 360):
        last = ang - step
        count_ang[int(ang / step)] += ((orien >= last) & (orien <= ang)).sum()
        ang += step
    return count_ang 
#--------------------------------------
def count_cc(img):
    _, cur_count_cc = label(1-img, connectivity=1, return_num=True)
    return cur_count_cc
