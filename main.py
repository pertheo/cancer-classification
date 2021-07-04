import pandas as pd
from evaluate import evaluate
from roc_curve import plot_roc_curve_rf, plot_roc_curve_lr, plot_roc_curve_gb
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def main():
    #reading files with training data using pandas
    df = pd.read_csv("arcene_train.data", header=None, nrows=1, sep=' ')
    columns = df.columns.tolist() # get the columns
    cols_to_use = columns[:len(columns)-1] # drop the last one
    df = pd.read_csv("arcene_train.data", header=None, sep=' ',usecols=cols_to_use)
    lf=pd.read_csv("arcene_train.labels",header=None, sep=' ')
    train_X=df.values #train features
    y_temp=lf.values
    train_y=y_temp.flatten() #train labels
    #reading files with validate data using pandas
    df1 = pd.read_csv("arcene_valid.data", header=None, nrows=1, sep=' ')
    columns = df1.columns.tolist()
    cols_to_use = columns[:len(columns)-1]
    df1 = pd.read_csv("arcene_valid.data", header=None, sep=' ',usecols=cols_to_use)
    lf1=pd.read_csv("arcene_valid.labels",header=None, sep=' ')
    test_X=df1.values #validate features, will be used to evaluate model
    y_temp1=lf1.values
    test_y=y_temp1.flatten() #validate labels, will be used to evaluate model
    
    #creating two models with hyperparameters chosen by random search and grid CV
    random_forest=RandomForestClassifier(bootstrap=False, max_depth=40, min_samples_split=8,
                      n_estimators=600) 
    gradient_boosting=GradientBoostingClassifier(learning_rate=0.1, max_depth=3,n_estimators=10)
    logistic_regression=LogisticRegression(solver='liblinear')
    
    #fitting training data
    logistic_regression.fit(train_X,train_y)
    random_forest.fit(train_X,train_y)
    gradient_boosting.fit(train_X,train_y)
    
    #predicting outcome of validate set
    predicted_y_rf=random_forest.predict(test_X)
    predicted_y_gb=gradient_boosting.predict(test_X)
    predicted_y_lr=logistic_regression.predict(test_X)
    print("\n*****RANDOM FOREST*****\n")
    evaluate(random_forest,train_X,train_y,test_X,test_y,predicted_y_rf)
    print("\n*****GRADIENT BOOSTING*****\n")
    evaluate(gradient_boosting,train_X,train_y,test_X,test_y,predicted_y_gb)
    print("\n*****LOGISTIC REGRESSION*****\n")
    evaluate(logistic_regression,train_X,train_y,test_X,test_y,predicted_y_lr)
    probs = random_forest.predict_proba(test_X)  
    probs = probs[:, 1]  
    
    #draw roc_curves
    fper, tper, thresholds = roc_curve(test_y, probs) 
    plot_roc_curve_rf(fper, tper)
    probs = gradient_boosting.predict_proba(test_X)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(test_y, probs) 
    plot_roc_curve_gb(fper, tper)
    probs = logistic_regression.predict_proba(test_X)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(test_y, probs) 
    plot_roc_curve_lr(fper, tper)   

main()
