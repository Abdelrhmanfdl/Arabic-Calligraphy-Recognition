import read_data
import preprocessing
import features
import train
import pickle
import numpy as np
from sklearn import metrics
import glob
import cv2
from time import time

dictionary = dict()
dictionary["diwani"] = 1
dictionary["naskh"] = 2
dictionary["parsi"] = 3
dictionary["rekaa"] = 4
dictionary["thuluth"] = 5
dictionary["maghribi"] = 6
dictionary["kufi"] = 7
dictionary["mohakek"] = 8
dictionary["Squar-kufic"] = 9

test_dirct = input("Enter test set directory: ")
out_dirct = input("Enter output directory: ")
filename = f'model.sav'
model = pickle.load(open(filename, 'rb'))
results = open(out_dirct+'results.txt', 'w')
times = open(out_dirct+'times.txt', 'w')
for filename in sorted(glob.glob(test_dirct+"*.png")):
    start = time()
    data = list()
    img = cv2.imread(filename) ## cv2.imread reads images in RGB format
    data.append([img,'dummy'])
    data , edges = preprocessing.pre_processing(data)
    data_features = features.get_features(data,edges)
    test_predictions = model.predict(data_features)
    results.write(str(dictionary[test_predictions[0]])+"\n")
    end = time()
    time_taken = end - start
    if time_taken == 0.0:
        time_taken = (1/1000)
    times.write(str(time_taken)+"\n")
    
    
