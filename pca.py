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

# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, pm25, test_size=0.2, random_state=42)

# Success
print "Training and testing split was successful."

#vs.ModelLearning(features, pm25)

#vs.ModelComplexity(X_train, y_train)

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import visuals_pca as vspca

regressor, y_pred = None, None

def fit_model(X, y):


    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    logistic = linear_model.LogisticRegression()

    pca = decomposition.PCA(n_components=6).fit(X)

    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    #pca_results = vspca.pca_results(X, pca)

    n_components = 6

    estimator = GridSearchCV(pipe, dict())
    estimator.fit(X, y)

    plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))
    plt.show()


fit_model(X_train, y_train)
