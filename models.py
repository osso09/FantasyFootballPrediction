# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.

import numpy as np
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import feature_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from get_data import test_players
import time
from metrics import mean_relative_error
from plots import histogram
import random

def hyperparameter_selection(regressors, x, y, folds):
    k_fold = cross_validation.StratifiedKFold(y, n_folds=folds)
    # get the index of the regressor with minimal average error over all folds
    MAE_averages = [0] * len(regressors)
    for index, reg in enumerate(regressors):
        print 'started next regressor', reg.kernel
        MAE_averages[index] = np.average(
            [mean_absolute_error(y[val], reg.fit(x[train], y[train]).predict(x[val])) for train, val in k_fold])
        print MAE_averages[index]
    return np.argmin(MAE_averages)


# Only one of the feature selection methods can be chosen
FEATURE_SELECTION = False
MANUAL_FEATURE_SELECTION = True
FEATURE_NORMALIZATION = False #True
HYPERPARAMETER_SELECTION = False
HISTOGRAM = True

# load data
# indices are
# 0: QB id
# 1: QB name
# 2: QB age
# 3: QB years pro
# 4-15: last game QB stats
# 16-27: last 10 games QB stats
# 28-31: last game defense stats
# 32-35: last 10 games defense stats
# 36: actual fantasy score = target
train = np.load('train.npy')
test = np.load('test.npy')

train_x = train[:, 2:36].astype(np.float) #Slicing arrays - take all rows, and columns 2-35 of the train array
train_y = train[:, 36].astype(np.float) #take last element of each row (actual score)
test_x = test[:, 2:36].astype(np.float)
test_y = test[:, 36].astype(np.float)
kernels = ['rbf', 'linear', 'sigmoid', 'poly']
degrees = [2, 3]
gamma_values = [0.05*k for k in range(1,4)]
C_values = [0.25*k for k in range(1, 5)]
epsilon_values = [0.05*k for k in range(1, 6)]

# Feature Normalization
if FEATURE_NORMALIZATION:
    print 'started feature normalization', time.time()
    x = np.concatenate((train_x, test_x), axis=0)
    x = preprocessing.scale(x)
    train_x = x[:len(train_x)]
    test_x = x[len(train_x):]


# Recursive Feature Elimination with cross-validation (RFECV)
#reduce features until no longer seeing improvement - returns a model
#seems to only pick 13 features 
if FEATURE_SELECTION:
    print 'started feature selection', time.time()
    selector = feature_selection.RFECV(estimator=SVR(kernel='linear'), step=3, cv=5)
    selector.fit(train_x, train_y)    
    print 'Selector ranking attempt 1: ', selector.ranking_
    selector.ranking_[0] = 8
    train_x = selector.transform(train_x)
    test_x = selector.transform(test_x)
    print 'Selector ranking: ', selector.ranking_
    print 'Number of Features selected: ', selector.n_features_
    print 'Support Vector: ', selector.support_

#TODO modify this to randomly selected
elif MANUAL_FEATURE_SELECTION: # leave out the two point attempts
    #manual_indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 14, 15, 16, 17, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    manual_indices = sorted(random.sample(range(0,34), random.randint(1,34)))
    print "indices used: ", manual_indices
    train_x = train_x[:, manual_indices]
    test_x = test_x[:, manual_indices]


# hyperparameter selection
if HYPERPARAMETER_SELECTION:
    regressors = []
    for C in C_values:
        for epsilon in epsilon_values:
            for kernel in kernels:
                if kernel == 'poly':
                    for gamma in gamma_values:
                        for degree in degrees:
                            regressors.append(SVR(C=C, epsilon=epsilon, kernel='poly', degree=degree, gamma=gamma))
                elif kernel in ['rbf', 'sigmoid']:
                    for gamma in gamma_values:
                        regressors.append(SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma))
                else:
                    regressors.append(SVR(C=C, epsilon=epsilon, kernel=kernel))

    print 'start hyperparameter selection', time.time()
    best_regressor = regressors[hyperparameter_selection(regressors, train_x, train_y, 5)]
    print best_regressor.C, best_regressor.epsilon, best_regressor.kernel, best_regressor.degree, best_regressor.gamma

else:
    best_regressor = SVR(C=0.25, epsilon=0.25, kernel='linear')

#Predict score (36) given facts 2-35
best_regressor.fit(train_x, train_y)
prediction = best_regressor.predict(test_x)

np.save('prediction.npy', prediction)

print 'RMSE, MAE, MRE (all):', mean_squared_error(test_y, prediction)**0.5, \
    mean_absolute_error(test_y, prediction), \
    mean_relative_error(test_y, prediction)

# determine error if only best 24 players are selected
indices = []
for index in range(len(test_x)):
    if test[index, 0] in test_players.keys():
        indices.append(index)
print 'RMSE, MAE, MRE (24 best):', mean_squared_error(test_y[indices], prediction[indices])**0.5, \
    mean_absolute_error(test_y[indices], prediction[indices]), \
    mean_relative_error(test_y[indices], prediction[indices])
print zip(test_y[indices], prediction[indices])

if HISTOGRAM:
    histogram(test_y, prediction)





