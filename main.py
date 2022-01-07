import read_data
import preprocessing
import features
import train

data = read_data.read_all_data()
data , edges = preprocessing.pre_processing(data)
data_features = features.get_features(data,edges)
model = train.train_model(data,data_features)
test_data = list()
print("-----Reading Test Data-----")
read_data.read_style(test_data,'dummy','test/*.png')
read_data.read_style(test_data,'dummy','test/*.jpg')
print("-----Preprocessing test data-----")
test_data , test_edges = preprocessing.pre_processing(test_data)
print("-----Extract test data features-----")
test_features = features.get_features(test_data,test_edges)
print("-----Predicting test data-----")
Prediction = model.predict(test_features)
print(Prediction)

