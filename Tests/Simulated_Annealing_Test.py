# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:00:16 2019

@author: Zoltan
"""

import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression


dataset_name = "Diabetes.csv"
dataset_prima_indians_name = "diabetes_prima_indians.csv"
label_column_name = "Outcome"

# Parameters that are kept constant during the tuning process
const_param = {
    'silent': False,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'map',
    'seed': 42,
    'n_estimators': 20,
    'n_jobs': -1
}

# Parameter search space
param_dict = OrderedDict()
param_dict['max_depth'] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
param_dict['subsample'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
param_dict['colsample_bytree'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
param_dict['learning_rate'] = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.0]
param_dict['gamma'] = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
param_dict['scale_pos_weight'] = [10, 20, 30, 40, 50, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700]


# logistic regression
const_param_logistic_regression = {
    'verbose': 0,
    'dual':  False,
    'class_weight': 'balanced',
    'penalty': 'l1'
}

param_dict_logistic_regression = OrderedDict()
param_dict_logistic_regression['tol'] = [0.00001, 0.0001, 0.001, 0.01, 0.1]
param_dict_logistic_regression['C'] = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
param_dict_logistic_regression['solver'] = ['liblinear', 'saga']
param_dict_logistic_regression['intercept_scaling'] = [0.1, 0.5, 1.0, 1.5, 2.0]
param_dict_logistic_regression['max_iter'] = [10, 20, 30, 50, 70, 90, 100, 120, 150, 170]


# Sample train_model function #
###############################

# def train_model(curr_params, param, Xtrain, Xvalid, Ytrain,
#                 Yvalid):
#     """
#     Train the model with given set of hyperparameters
#     curr_params - Dict of hyperparameters and chosen values
#     param - Dict of hyperparameters that are kept constant
#     Xtrain - Train Data
#     Xvalid - Validation Data
#     Ytrain - Train labels
#     Yvalid - Validaion labels
#     """
#     params_copy = param.copy()
#     params_copy.update(curr_params)
#     model = XGBClassifier(**params_copy)
#     model.fit(Xtrain, Ytrain)
#     preds = model.predict(Xvalid)
#     metric_val = f1_score(Yvalid, preds) # Any metric can be used
#     return model, metric_val


def choose_params(param_dict, curr_params=None):
    """
    Function to choose parameters for next iteration
    Inputs:
    param_dict - Ordered dictionary of hyperparameter search space
    curr_params - Dict of current hyperparameters
    Output:
    Dictionary of parameters
    """
    if curr_params:
        next_params = curr_params.copy()
        param_to_update = np.random.choice(list(param_dict.keys()))
        param_vals = param_dict[param_to_update]
        curr_index = param_vals.index(curr_params[param_to_update])
        if curr_index == 0:
            next_params[param_to_update] = param_vals[1]
        elif curr_index == len(param_vals) - 1:
            next_params[param_to_update] = param_vals[curr_index - 1]
        else:
            next_params[param_to_update] = \
                param_vals[curr_index + np.random.choice([-1, 1])]
    else:
        next_params = dict()
        for k, v in param_dict.items():
            next_params[k] = np.random.choice(v)

    return next_params


def simulate_annealing(param_dict,
                       const_param,
                       X_train,
                       X_valid,
                       Y_train,
                       Y_valid,
                       fn_train,
                       maxiters=100,
                       alpha=0.85,
                       beta=1.3,
                       T_0=0.40,
                       update_iters=5):
    """
    Function to perform hyperparameter search using simulated annealing
    Inputs:
    param_dict - Ordered dictionary of Hyperparameter search space
    const_param - Static parameters of the model
    Xtrain - Train Data
    Xvalid - Validation Data
    Ytrain - Train labels
    Yvalid - Validaion labels
    fn_train - Function to train the model
        (Should return model and metric value as tuple, sample commented above)
    maxiters - Number of iterations to perform the parameter search
    alpha - factor to reduce temperature
    beta - constant in probability estimate
    T_0 - Initial temperature
    update_iters - # of iterations required to update temperature
    Output:
    Dataframe of the parameters explored and corresponding model performance
    """
    columns = [*param_dict.keys()] + ['Metric', 'Best Metric']
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_metric = -1.
    prev_metric = -1.
    prev_params = None
    best_params = dict()
    weights = list(map(lambda x: 10**x, list(range(len(param_dict)))))
    hash_values = set()
    T = T_0

    for i in range(maxiters):
        print('Starting Iteration {}'.format(i))
        while True:
            curr_params = choose_params(param_dict, prev_params)
            indices = [param_dict[k].index(v) for k, v in curr_params.items()]
            hash_val = sum([i * j for (i, j) in zip(weights, indices)])
            if hash_val in hash_values:
                print('Combination revisited')
            else:
                hash_values.add(hash_val)
                break

        model, metric = fn_train(curr_params, const_param, X_train,
                                 X_valid, Y_train, Y_valid)

        if metric > prev_metric:
            print('Local Improvement in metric from {:8.4f} to {:8.4f} '
                  .format(prev_metric, metric) + ' - parameters accepted')
            prev_params = curr_params.copy()
            prev_metric = metric

            if metric > best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} '
                      .format(best_metric, metric) +
                      ' - best parameters updated')
                best_metric = metric
                best_params = curr_params.copy()
                best_model = model
        else:
            rnd = np.random.uniform()
            diff = metric - prev_metric
            threshold = np.exp(beta * diff / T)
            if rnd < threshold:
                print('No Improvement but parameters accepted. Metric change' +
                      ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'
                      .format(diff, threshold, rnd))
                prev_metric = metric
                prev_params = curr_params
            else:
                print('No Improvement and parameters rejected. Metric change' +
                      ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'
                      .format(diff, threshold, rnd))

        results.loc[i, list(curr_params.keys())] = list(curr_params.values())
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric

        if i % update_iters == 0:
            T = alpha * T

    return results, best_model


# Function to train model
def train_model(curr_params, param, Xtrain, Xvalid, Ytrain, Yvalid, metric=f1_score):
    """
    Train the model with given set of hyperparameters
    curr_params - Dict of hyperparameters and chosen values
    param - Dict of hyperparameters that are kept constant
    Xtrain - Train Data
    Xvalid - Validation Data
    Ytrain - Train labels
    Yvalid - Validaion labels
    metric - Metric to compute model performance on
    """
    params_copy = param.copy()
    params_copy.update(curr_params)
    model = XGBClassifier(**params_copy)
    model.fit(Xtrain, Ytrain)
    preds = model.predict(Xvalid)
    metric_val = metric(Yvalid, preds)
    
    return model, metric_val


def train_logistic_regression(curr_params, param, Xtrain, Xvalid, Ytrain, Yvalid, metric=f1_score):
    params_copy = param.copy()
    params_copy.update(curr_params)
    model = LogisticRegression(**params_copy)
    model.fit(Xtrain, Ytrain)
    preds = model.predict(Xvalid)
    metric_val = metric(Yvalid, preds)
    
    return model, metric_val


def main():
    dataset = pd.read_csv(dataset_prima_indians_name)
    
    # calculate the mean of each column which will be used 
    # to eliminate Na values (missing values)
    mean_Glucose = dataset['Glucose'].mean(skipna=True)
    mean_BloodPressure = dataset['BloodPressure'].mean(skipna=True)
    mean_SkinTickness = dataset['SkinThickness'].mean(skipna=True)
    mean_Insulin = dataset['Insulin'].mean(skipna=True)
    mean_BMI = dataset['BMI'].mean(skipna=True)
    mean_DiabetesPedigreeFunction = dataset['DiabetesPedigreeFunction'].mean(skipna=True)
    '''mean_bp_1s = dataset['bp.1s'].mean(skipna=True)
    mean_bp_1d = dataset['bp.1d'].mean(skipna=True)
    mean_waist = dataset['waist'].mean(skipna=True)
    mean_hip = dataset['hip'].mean(skipna=True)'''
    
    # replace missing values this the mean
    dataset['Glucose'] = dataset['Glucose'].replace(0, mean_Glucose)
    dataset['BloodPressure'] = dataset['BloodPressure'].replace(0, mean_BloodPressure)
    dataset['SkinThickness'] = dataset['SkinThickness'].replace(0, mean_SkinTickness)
    dataset['Insulin'] = dataset['Insulin'].replace(0, mean_Insulin)
    dataset['BMI'] = dataset['BMI'].replace(0, mean_BMI)
    dataset['DiabetesPedigreeFunction'] = dataset['DiabetesPedigreeFunction'].replace(0, mean_DiabetesPedigreeFunction)
    '''dataset['bp.1s'].fillna(mean_bp_1s, inplace=True)
    dataset['bp.1d'].fillna(mean_bp_1d, inplace=True)
    dataset['waist'].fillna(mean_waist, inplace=True)
    dataset['hip'].fillna(mean_hip, inplace=True)'''
    
    x_tr, test = train_test_split(dataset, test_size=0.2, shuffle=True)
    train, valid = train_test_split(x_tr, test_size=0.25, shuffle=True)
    
    xtrain, ytrain = train.drop(label_column_name, axis=1), train[label_column_name]
    xvalid, yvalid = valid.drop(label_column_name, axis=1), valid[label_column_name]
    xtest, ytest = test.drop(label_column_name , axis=1), test[label_column_name]
    
    global_best = 0
    for i in range(20):
        res, best_model = simulate_annealing(param_dict_logistic_regression, const_param_logistic_regression, xtrain,
                                     xvalid, ytrain, yvalid,
                                     train_logistic_regression, maxiters=50,
                                     beta=0.5)
        best_res = res.tail(1)
        best_res = best_res.drop('Metric', axis=1)
        best_res = best_res.drop('Best Metric', axis=1)
        best_res_dict = dict()
        for column_name in best_res.columns.values:
            best_res_dict[column_name] = best_res.loc[column_name]
        
        model, metric = train_logistic_regression(best_res_dict, const_param_logistic_regression, xtrain,
                                 xvalid, ytrain, yvalid)
        
        if metric > global_best:
            global_best = metric
            global_model = model
            global_res = res

    print(global_res)
    
    
if __name__ == "__main__":
    main()

