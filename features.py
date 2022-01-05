import cv2
import numpy as np
#--------------------------------------
def get_features(data,edges):
    print("-----Extracting Features-----")
    features = list()
    for index in range(len(data)):
        img_feature = list()
        img_feature.append(get_hist(data[index][0]))
        img_feature.append(HVSL(edges[index]))
        c_c , compactness = connected_components(data[index][0])
        img_feature.append(c_c)
        img_feature.append(compactness)
        img_feature.append(num_pix_per_col(data[index][0]))
        img_feature.append(num_pix_per_row(data[index][0]))
        img_feature.append(before_after_morph(data[index][0]))
        img_feature.append(before_after_morph2(data[index][0]))
        img_feature.append(before_after_morph3(data[index][0]))
        ratio_pixels_above_basline , ratio_pixels_below_basline = baseline(data[index][0])
        img_feature.append(ratio_pixels_above_basline)
        img_feature.append(ratio_pixels_below_basline)
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
def connected_components(img):
    n,_=cv2.connectedComponents(img)
    return n , (n/(img.shape[0]*img.shape[1]))
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
def before_after_morph(img):
    num_bef = np.sum(img)
    img_copy = np.array(img)
    kernel = np.ones((5,5), np.uint8)
    img_copy = cv2.dilate(img_copy, kernel, iterations=1)
    img_copy = cv2.erode(img_copy, kernel, iterations=2)
    return np.sum(img_copy)/num_bef
#--------------------------------------
def before_after_morph2(img):
    num_bef = np.sum(img)
    img_copy = np.array(img)
    kernel_di = np.ones((7,7), np.uint8)
    kernel_er = np.ones((7,1), np.uint8)
    img_copy = cv2.dilate(img_copy, kernel_di, iterations=1)
    img_copy = cv2.erode(img_copy, kernel_er, iterations=2)
    return np.sum(img_copy)/num_bef
#--------------------------------------
def before_after_morph3(img):
    num_bef = np.sum(img)
    img_copy = np.array(img)
    kernel_di = np.ones((1,7), np.uint8)
    kernel_er = np.ones((5,4), np.uint8)
    img_copy = cv2.dilate(img_copy, kernel_di, iterations=2)
    img_copy = cv2.erode(img_copy, kernel_er, iterations=3)
    img_copy = cv2.dilate(img_copy, kernel_di, iterations=1)
    return np.sum(img_copy)/num_bef
#--------------------------------------
# def oreinatation(data,img,index):    
#     sobel_x = np.array([[ -1, 0, 1], 
#                        [ -2, 0, 2], 
#                        [ -1, 0, 1]])
#     sobel_y = np.array([[ -1, -2, -1], 
#                        [ 0, 0, 0], 
#                        [ 1, 2, 1]])
#     ang_category_count = 20
#     count_ang = np.zeros((len(data), ang_category_count + 1))
#     myImg = np.array(img)
#     filtered_blurred_x = cv2.filter2D(myImg, cv2.CV_32F, sobel_x)
#     filtered_blurred_y = cv2.filter2D(myImg, cv2.CV_32F, sobel_y)
#     orien = cv2.phase(np.array(filtered_blurred_x, np.float32),
#                       np.array(filtered_blurred_y, dtype=np.float32), 
#                       angleInDegrees=True)
#     orien = np.array(orien)
#     step = 360/ang_category_count
#     ang = step
#     while(ang <= 360):
#         last = ang - step
#         count_ang[index][int(ang / step)] += ((orien >= last) & (orien < ang)).sum()
#         ang += step   
#--------------------------------------
def baseline(img):
    rows = len(img)  
    histogram = list() 
    img = np.array(img)
    for index in range(int(rows/3),rows):
        row = img[index , :]
        hist = cv2.calcHist([row],[0],None,[256],[0,256]) 
        histogram.append(hist[0][0])
    basline_row = histogram. index(max(histogram))+int(rows/3)
    ratio_pixels_above_basline = get_hist(img[:basline_row,:])
    ratio_pixels_below_basline = get_hist(img[basline_row+1:,:])
    return ratio_pixels_above_basline, ratio_pixels_below_basline
    
         
