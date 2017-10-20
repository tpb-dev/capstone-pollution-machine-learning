
# TODO: Import 'train_test_split'
from sklearn.cross_validation import *

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn import metrics
import matplotlib.pyplot as plt
import visuals as vs

# Import supplementary visualizations code visuals.py
#import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

data = pd.read_csv('prsa.csv')
data = data.drop('No', axis=1)

data = data.dropna(axis=0, how='any')
pm25 = data["pm2.5"]
#print data["pm2.5"]
data['cbwd'].replace('cv', 0,inplace=True)
data['cbwd'].replace('NW', 1,inplace=True)
data['cbwd'].replace('NE', 2,inplace=True)
data['cbwd'].replace('SW', 3,inplace=True)
data['cbwd'].replace('SE', 4,inplace=True)

#print data.describe()

print "len: ", len(data)

features = data.drop('pm2.5', axis = 1)

# Success
print "PM 2.5 dataset has {} data points with {} variables each.".format(*data.shape)

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = metrics.r2_score(y_true, y_predict)

    # Return the score
    return score



# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, pm25, test_size=0.2, random_state=42)

# Success
print "Training and testing split was successful."

#vs.ModelLearning(features, pm25)

#vs.ModelComplexity(X_train, y_train)

from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.grid_search import GridSearchCV

regressor, y_pred = None, None

def fit_model(X, y):


    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DummyClassifier()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)



    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

score = reg.score(X_test, y_test)

print "Score is: ", score
