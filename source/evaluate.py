import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score

def mean_squared_error(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean()

def evaluate(model,train_X,train_y,test_X,test_y,pred_y):
    print("ACCURACY SCORE:\nTrain :",accuracy_score(train_y,model.predict(train_X)),"\nTest :",accuracy_score(test_y,pred_y))
    conf_matrix = confusion_matrix(test_y, pred_y)
    print ('Confusion Matrix :')
    print(conf_matrix)
    print ('Classification Report : ')
    print (classification_report(test_y, pred_y))
    print('AUC-ROC:',roc_auc_score(test_y, pred_y)) #higher value, means better model
    print('LOGLOSS Value is',log_loss(test_y, pred_y)) #the smaller the better
    kf = KFold(n_splits=5, random_state=None)
    result = cross_val_score(model, test_X, test_y, cv = kf)
    print(result)
    print("CROSS VALIDATIONS:")
    scores=cross_val_score(model,train_X,train_y,cv=kf)
    print("Avg train accuracy: {}".format(scores.mean()))
    print("Avg test accuracy: {}".format(result.mean()))
    print("Mean squared error of train : ",mean_squared_error(train_y, model.predict(train_X)),", test : ",mean_squared_error(test_y, model.predict(test_X)))
