# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:40:46 2019

@author: Zoltan
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from sklearn.model_selection import train_test_split

from AutoAIAlgorithm.ParticleSwarmOptimization import PSO

import matplotlib.pyplot as plt

from sklearn import datasets

from scipy import stats
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_squared_error


#--- MAIN ---------------------------------------------------------------------+
def main():
        
    boston = datasets.load_boston()
    
    # Outlier elimination
    z = np.abs(stats.zscore(boston.data))
    
    X = boston.data[(z < 3).all(axis=1)]
    y = boston.target[(z < 3).all(axis=1)]
    
    # Scaling in range of [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    X = scaler.fit_transform(X) 
    
    # Dealing with missing data
    imputer = Imputer(missing_values=np.nan, strategy="mean")
    X = imputer.fit_transform(X)
    
    # Splitting the data into training set, test set and validation set
    x_tr, x_test, y_tr, y_test = train_test_split(X, y, test_size=0.05)
    xtrain, xvalid, ytrain, yvalid = train_test_split(x_tr, y_tr, test_size=0.33)

    num_particles=10
    num_iterations=30
   
    pso = PSO(particle_count=num_particles, evaluation_metric=mean_squared_error, is_classification=False, is_maximization=False)
    
    best_metric, best_model = pso.fit(X_train=xtrain,
                                      X_test=xvalid,
                                      Y_train=ytrain,
                                      Y_test=yvalid,
                                      maxiter=num_iterations,
                                      verbose=True,
                                      max_distance=0.05)
    print("BEST")
    print(best_metric)
    print(best_model)
    
    predicted = best_model.predict(x_test)
    fig,ax = plt.subplots()
    ax.scatter(y_test, predicted)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    fig.show()


if __name__ == "__main__":
    main()