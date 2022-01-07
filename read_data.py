import cv2
import glob
#----------------------------------------------------
def read_all_data():
    print("-----Reading-----")
    data = list()
    labels = ["diwani", "naskh" ,"parsi", "rekaa", "thuluth", "maghribi", "kufi", "mohakek", "Squar-kufic"]
    filepaths = ['ACdata_base/1/*.jpg','ACdata_base/2/*.jpg','ACdata_base/3/*.jpg','ACdata_base/4/*.jpg','ACdata_base/5/*.jpg','ACdata_base/6/*.jpg','ACdata_base/7/*.jpg','ACdata_base/8/*.jpg','ACdata_base/9/*.jpg']
    for index in range(len(labels)):
        read_style(data,labels[index],filepaths[index])
    return data
#--------------------------------------------------------
def read_style(data,label,filepath):
    for filename in sorted(glob.glob(filepath)):
        img = cv2.imread(filename) ## cv2.imread reads images in RGB format
        data.append([img,label])
