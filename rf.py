import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.learning import GradientBoostingQuantileRegressor
from skopt.learning import RandomForestRegressor,ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.utils._testing import assert_raises
from sklearn.gaussian_process.kernels import Matern
from modAL.models import BayesianOptimizer
from modAL.acquisition import optimizer_EI, max_EI,max_PI,max_UCB
from bayes_opt import BayesianOptimization
import pytest
from math import pi
from sklearn import preprocessing
from keras.utils import np_utils


# generating the data
X = np.linspace(0, 20, 1000).reshape(-1, 1)
y = np.sin(X)/2 - ((10 - X)**2)/50 + 2
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 5))
    plt.plot(X, y, c='k', linewidth=6)
    plt.title('The function to be optimized')
    plt.show()
# assembling initial training set
X_initial, y_initial = X[150].reshape(1, -1), y[150].reshape(1, -1)

# defining the kernel for the Gaussian process
kernel = Matern(length_scale=1.0)
base = RandomForestRegressor()
regressor = RandomForestRegressor(max_depth=3, random_state=0)#ExtraTreesRegressor()#GradientBoostingQuantileRegressor(random_state=0)
'''
base = RandomForestRegressor()
regressor = GradientBoostingQuantileRegressor(base_estimator=base)
with pytest.raises(ValueError):
    # 'type GradientBoostingRegressor',
    regressor.fit(X, y)
    '''
#print(regressor)

# initializing the optimizer
optimizer = BayesianOptimizer(
    estimator= regressor,
    X_training=X_initial, y_training=np.ravel(y_initial),
    query_strategy=max_UCB
)

# Bayesian optimization
for n_query in range(5):
    query_idx, query_inst = optimizer.query(X)
    optimizer.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1)[0])
#print(X)
y_pred, y_std = optimizer.predict(X, return_std=True)
y_pred, y_std = y_pred.ravel(), y_std.ravel()
X_max, y_max = optimizer.get_max()
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 5))
    plt.scatter(optimizer.X_training, optimizer.y_training, c='k', s=50, label='Queried')
    plt.scatter(X_max, y_max, s=100, c='r', label='Current optimum')
    plt.plot(X.ravel(), y, c='k', linewidth=2, label='Function')
    plt.plot(X.ravel(), y_pred, label='GP regressor')
    plt.fill_between(X.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.5)
    plt.title('First five queries of Bayesian optimization')
    plt.legend()
    plt.show()