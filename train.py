from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
def train_model(data,features):
    print("-----Training-----")
    data = np.array(data , dtype = object)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(data,features)
    model = RandomForestClassifier()
    # Fit model
    model.fit(X_train, y_train)
    model_accuracy(model,X_train, y_train, X_val, y_val, X_test, y_test,data, features)
    return model
    
#--------------------------------------    
def split_data(data,features):
    X_train, X_test, y_train, y_test = train_test_split(features, data[:,1], 
        test_size=0.2, shuffle = True, random_state = 0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
        test_size=0.2, random_state= 0) 
    return X_train, y_train, X_val, y_val, X_test, y_test

def model_accuracy(model,X_train, y_train, X_val, y_val, X_test, y_test,data, features):
    # Model Accuracy, how often is the classifier correct?
    #validation
    val_predictions = model.predict(X_val)
    print("Validation Accuracy:",(metrics.accuracy_score(y_val, val_predictions))*100)
    train_val = list()
    for index in range(len(X_train)):
        train_val.append(y_train[index])
    for y in y_val:
        train_val.append(y)
    train_data = X_train + X_val
    model.fit(train_data, train_val)
    #Test
    test_predictions = model.predict(X_test)
    print("Test Accuracy:",(metrics.accuracy_score(y_test, test_predictions))*100)
    #diwani
    val_predictions = model.predict(features[:190])
    print("Accuracy of diwani:",(metrics.accuracy_score(data[:190,1], val_predictions))*100)
    #naskh
    val_predictions = model.predict(features[190:380])
    print("Accuracy of naskh:",(metrics.accuracy_score(data[190:380,1], val_predictions))*100)
    #parsi
    val_predictions = model.predict(features[380:560])
    print("Accuracy of parsi:",(metrics.accuracy_score(data[380:560,1], val_predictions))*100)
    #rekaa
    val_predictions = model.predict(features[560:745])
    print("Accuracy of rekaa:",(metrics.accuracy_score(data[560:745,1], val_predictions))*100)
    #thuluth
    val_predictions = model.predict(features[745:940])
    print("Accuracy of thuluth:",(metrics.accuracy_score(data[745:940,1], val_predictions))*100)
    #maghribi
    val_predictions = model.predict(features[940:1120])
    print("Accuracy of maghribi:",(metrics.accuracy_score(data[940:1120,1], val_predictions))*100)
    #kufi
    val_predictions = model.predict(features[1120:1305])
    print("Accuracy of kufi:",(metrics.accuracy_score(data[1120:1305,1], val_predictions))*100)
    #mohakek
    val_predictions = model.predict(features[1305:1495])
    print("Accuracy of mohakek:",(metrics.accuracy_score(data[1305:1495,1], val_predictions))*100)
    #Squar-kufic
    val_predictions = model.predict(features[1495:])
    print("Accuracy of Squar-kufic:",(metrics.accuracy_score(data[1495:,1], val_predictions))*100)

