Program that distinguishes cancer versus normal patterns from mass-spectrometric data based on the dataset: https://archive.ics.uci.edu/ml/datasets/Arcene 

Methods used: Random Forrest, Gradient Boosting, Logistic Regression. The solution is resolved as a classifier. Accuracy on the training set and on the test set are compared. Parameters were found by using both grid search and random search methods.

Metrics used: ROC curve, AUC, confusion matrix, log loss, accuracy score, classification report (precision, recall, F1-score, support), mean squared error.

Written in Python3+

Modules used: numpy, sklearn, matplotlib, pandas
