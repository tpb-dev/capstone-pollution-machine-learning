
#Used code for feature importance from here: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

regressor, y_pred = None, None

def fit_model(X, y):

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    return clf

trained_model = fit_model(X_train, y_train)

predictions = trained_model.predict(X_test)

score = trained_model.score(X_test, y_test)

print "Prediction score is: ", score

importances = trained_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in trained_model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
