# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv('prsa.csv')
data = data.drop('No', axis=1)

data = data.dropna(axis=0, how='any')
pm25 = data["pm2.5"]
data['cbwd'].replace('cv', 0,inplace=True)
data['cbwd'].replace('NW', 1,inplace=True)
data['cbwd'].replace('NE', 2,inplace=True)
data['cbwd'].replace('SW', 3,inplace=True)
data['cbwd'].replace('SE', 4,inplace=True)

print data.describe()

print "len: ", len(data)

features = data.drop('pm2.5', axis = 1)

# Success
print "PM 2.5 dataset has {} data points with {} variables each.".format(*data.shape)


from sklearn.metrics import *


# TODO: Import 'train_test_split'
from sklearn.cross_validation import *

# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, pm25, test_size=0.2, random_state=42)

# Success
print "Training and testing split was successful."


from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression


def fit_model(X, y, X_test, y_test):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    regressor = LinearRegression()

    # Fit the grid search object to the data to compute the optimal model
    regressor.fit(X, y)

    y_pred = regressor.predict(X_test)

    score = metrics.r2_score(y_test, y_pred)

    # Return the optimal model after fitting the data
    return (regressor.intercept_, regressor.coef_, score)

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train, X_test, y_test)

print 'Estimated intercept coefficent:', reg[0]
print 'Number of coefficients:', len(reg[1])
print "Variance score: %.2f" % reg[2]



# Produce the value for 'max_depth'
#print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])



#http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
