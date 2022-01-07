import preprocessing
import features
import pickle
import glob
import cv2
from time import time
import argparse


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

def main(test_dirct, out_dirct, filename):
    model = pickle.load(open(filename, 'rb'))
    results = open(out_dirct+'results.txt', 'w')
    times = open(out_dirct+'times.txt', 'w')
    index = 0
    for filename in sorted(glob.glob(test_dirct+"*.png")):
        start = time()
        try:
            data = list()
            img = cv2.imread(filename) ## cv2.imread reads images in RGB format
            data.append([img,'dummy'])
            data , edges = preprocessing.pre_processing(data)
            data_features = features.get_features(data,edges)
            test_predictions = model.predict(data_features)
            prediction = str(dictionary[test_predictions[0]])
            if index != 0:
                prediction = "\n"+str(prediction) 
            results.write(prediction)
            end = time()
            time_taken = end - start
            if time_taken == 0.0:
                time_taken = (1/1000)
            time_taken = str(round(time_taken, 2)) 
            if index != 0:
                time_taken = "\n"+str(time_taken) 
            times.write(time_taken)
        except:
            prediction = "-1"
            if index != 0:
                prediction = "\n"+prediction
            results.write(prediction)
            end = time()
            time_taken = str(round(time_taken, 2)) 
            if index != 0:
                time_taken = "\n"+str(time_taken) 
            times.write(time_taken)
        index += 1
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("inputfolder", help="Input File")
    parser.add_argument("outputfolder", help="Output File")

    args = parser.parse_args()
    
    filename = f'model.sav'
    main(args.inputfolder, args.outputfolder, filename)