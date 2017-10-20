
from sklearn.cross_validation import *

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn import metrics
import matplotlib.pyplot as plt
import visuals as vs


data = pd.read_csv('prsa.csv')
data = data.drop('No', axis=1)

data = data.dropna(axis=0, how='any')
pm25 = data["pm2.5"]

data['cbwd'].replace('cv', 0,inplace=True)
data['cbwd'].replace('NW', 1,inplace=True)
data['cbwd'].replace('NE', 2,inplace=True)
data['cbwd'].replace('SW', 3,inplace=True)
data['cbwd'].replace('SE', 4,inplace=True)

print "len: ", len(data)

features = data.drop('pm2.5', axis = 1)

# Success
print "PM 2.5 dataset has {} data points with {} variables each.".format(*data.shape)

def performance_metric(y_true, y_predict):
    score = metrics.r2_score(y_true, y_predict)
    return score


X_train, X_test, y_train, y_test = train_test_split(features, pm25, test_size=0.2, random_state=42)

X_full_train, _, y_full_train, _ = train_test_split(features, pm25, test_size=0, random_state=42)

# Success
print "Training and testing split was successful."

#vs.ModelLearning(features, pm25)

#vs.ModelComplexity(X_train, y_train)

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

regressor, y_pred = None, None

def fit_model(X, y):


    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(regressor, X_full_train, y_full_train, cv=5)
    print "KFold cross validation mean = ", scores


    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

score = reg.score(X_test, y_test)

print "Score is: ", score

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])

#Cross validation
