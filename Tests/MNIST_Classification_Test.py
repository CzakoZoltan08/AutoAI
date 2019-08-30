# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:58:21 2019

@author: Zoltan
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from sklearn import datasets

from sklearn.model_selection import train_test_split

from AutoAIAlgorithm.ParticleSwarmOptimization import PSO

from sklearn.metrics import accuracy_score


#--- MAIN ---------------------------------------------------------------------+
def main():
    # load the MNIST digits dataset
    mnist = datasets.load_digits()
    
    X = mnist.data
    y = mnist.target
   
    # Splitting the data into training set, test set and validation set
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    num_particles=5
    num_iterations=30
    
    pso = PSO(particle_count=num_particles, distance_between_initial_particles=0.7, evaluation_metric=accuracy_score)
    
    best_metric, best_model = pso.fit(X_train=x_train,
                                      X_test=x_test,
                                      Y_train=y_train,
                                      Y_test=y_test,
                                      maxiter=num_iterations,
                                      verbose=True,
                                      max_distance=0.05)
            
    print("BEST")
    print(best_metric)
    print(best_model)


if __name__ == "__main__":
    main()