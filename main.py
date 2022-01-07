import read_data
import preprocessing
import features
import train
import pickle
import cv2

data = read_data.read_all_data()
data , edges = preprocessing.pre_processing(data)
data_features = features.get_features(data,edges)
model = train.train_model(data,data_features)
filename = f'model.sav'
#pickle.dump(model, open(filename, 'wb'))

