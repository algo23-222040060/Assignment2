import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer as Imputer

from sklearn.linear_model import LogisticRegression
from sklearn import tree

import warnings
warnings.filterwarnings('ignore')

def creatDictKV(keys, vals):
    lookup = {}
    if len(keys) == len(vals):
        for i in range(len(keys)):
            key = keys[i]
            val = vals[i]
            lookup[key] = val
    return lookup

#Calculating AUC function
def computeAUC(y_true,y_score):
    auc = roc_auc_score(y_true,y_score)
    print("auc=",auc)
    return auc

colnames = ['ID', 'label', 'RUUnsecuredL', 'age', 'NOTime30-59',
                'DebtRatio', 'Income', 'NOCredit', 'NOTimes90',
                'NORealEstate', 'NOTime60-89', 'NODependents']
col_nas = ['', 'NA', 'NA', 0, [98, 96], 'NA', 'NA', 'NA', [98, 96], 'NA', [98, 96], 'NA']
col_na_values = creatDictKV(colnames, col_nas)
dftrain = pd.read_csv("data\cs-training.csv", names=colnames, na_values=col_na_values, skiprows=[0])

y_train = np.asarray([int(x)for x in dftrain.pop("label")])
x_train = dftrain.values
dftest = pd.read_csv("data\cs-test.csv", names=colnames, na_values=col_na_values, skiprows=[0])
test_id = [int(x) for x in dftest.pop("ID")]
y_test = np.asarray(dftest.pop("label"))
x_test = dftest.values

# Use StratifiedShuffleSplit to divide data set into training_new and test_new
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.33333,random_state=0)
for train_index, test_index in sss.split(x_train, y_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train_new, x_test_new = x_train[train_index], x_train[test_index]
    y_train_new, y_test_new = y_train[train_index], y_train[test_index]

y_train = y_train_new
x_train = x_train_new
#3，Use Imputer to replace all missing value by mean value
imp = Imputer(missing_values=np.nan, strategy='mean')

x_test_new.shape

x_test.shape

# replace 96,98 in NumberTime30-59 and NumberTime60-89，90 by NaN
# replace 0 in Age by NaN
def main():
    colnames = ['ID', 'label', 'RUUnsecuredL', 'age', 'NOTime30-59',
                'DebtRatio', 'Income', 'NOCredit', 'NOTimes90',
                'NORealEstate', 'NOTime60-89', 'NODependents']
    col_nas = ['', 'NA', 'NA', 0, [98, 96], 'NA', 'NA', 'NA', [98, 96], 'NA', [98, 96], 'NA']
    col_na_values = creatDictKV(colnames, col_nas)
    dftrain = pd.read_csv("data\cs-training.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    # print(dftrain)
    y_train = np.asarray([int(x) for x in dftrain.pop("label")])
    x_train = dftrain.values

    dftest = pd.read_csv("data\cs-test.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    y_test = np.asarray(dftest.pop("label"))
    x_test = dftest.values
    # Use StratifiedShuffleSplit to divide data set into training_new and test_new
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33333, random_state=0)
    for train_index, test_index in sss.split(x_train, y_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train_new, x_test_new = x_train[train_index], x_train[test_index]
        y_train_new, y_test_new = y_train[train_index], y_train[test_index]

    y_train = y_train_new
    x_train = x_train_new

    imp = Imputer(missing_values=np.nan, strategy='mean')
    x_train = imp.fit_transform(x_train)
    x_test_new = imp.fit_transform(x_test_new)
    x_test = imp.fit_transform(x_test)
    # building RF model from training_new
    rf = RandomForestClassifier(n_estimators=100,
                                oob_score=True,
                                min_samples_split=2,
                                min_samples_leaf=50,
                                n_jobs=-1,
                                class_weight='balanced_subsample',
                                bootstrap=True)
    # model comparison
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    predicted_probs_train = lr.predict_proba(x_train)
    predicted_probs_train = [x[1] for x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)

    predicted_probs_test_new = lr.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)

    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predicted_probs_train = model.predict_proba(x_train)
    predicted_probs_train = [x[1] for x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)

    predicted_probs_test_new = lr.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)
    # compare importance
    rf.fit(x_train, y_train)
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), dftrain.columns), reverse=True))

    param_grid = {"max_features": [2, 3, 4], "min_samples_leaf": [50]}
    grid_search = GridSearchCV(rf, cv=10, scoring='roc_auc', param_grid=param_grid)
    # c.output best model
    # building model by bestfit parameter
    grid_search.fit(x_train, y_train)
    print("the best parameter:", grid_search.best_params_)
    print("the best score:", grid_search.best_score_)

    # predict train_new data
    predicted_probs_train = grid_search.predict_proba(x_train)
    predicted_probs_train = [x[1] for x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)
    # predict test_new data（validataion data）
    predicted_probs_test_new = grid_search.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)
    # predict test data
    predicted_probs_test = grid_search.predict_proba(x_test)
    predicted_probs_test = ["%.9f" % x[1] for x in predicted_probs_test]
    submission = pd.DataFrame({'Id': test_id, 'Probability': predicted_probs_test})
    submission.to_csv("rf_benchmark.csv", index=False)


if __name__ == "__main__":
    main()