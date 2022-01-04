import read_data
import preprocessing
import features
import train

data = read_data.read_all_data()
data , edges = preprocessing.pre_processing(data)
features = features.get_features(data,edges)
train.train_model(data,features)

