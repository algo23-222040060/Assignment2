# Assignment2

Predicting credit default by anaylsis of imbalanced data and forecasting the possiblity of credit default using random forest

## Data treatment
1. Use StratifiedShuffleSplit to divide data set into training_new and test_new
2. Use Imputer to replace all missing value by mean value
3. replace 96,98 in NumberTime30-59 and NumberTime60-89，90 by NaN
4. replace 0 in Age by NaN
5. building RF model from training_new 

## Model comparsion
1. build logistic regression model
2. build decision tree classifier model
3. compare the importance gain of the two model

## predicting by the best fit model
By choosing the best model and adjust the parameter, we test on both training_new and test data
best parameter: {'max_features': 2, 'min_samples_leaf': 50}

auc= 0.909869939233906 for training data
auc= 0.8645695844668151 for test data
