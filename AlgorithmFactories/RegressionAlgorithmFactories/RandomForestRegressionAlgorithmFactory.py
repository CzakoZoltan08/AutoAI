# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:07:24 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.ensemble import RandomForestRegressor 

from ..Algorithm import Algorithm


const_param = {
    'verbose': 0,
    'warm_start': False
}

dicrete_hyper_parameter_list_of_criterion = ["mse", "mae"]
dicrete_hyper_parameter_list_of_max_depth = range(1,100)
dicrete_hyper_parameter_list_of_max_sample_split = range(1,100)
dicrete_hyper_parameter_list_of_estimators = range(1,320)
continuous_hyper_parameter_mapping_index_key_mapping = ["min_samples_leaf", "min_weight_fraction_leaf", "min_impurity_decrease"]
discrete_hyper_parameter_mapping = ["max_depth", "min_samples_split", "n_estimators","criterion"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["max_depth"] = dicrete_hyper_parameter_list_of_max_depth
discrete_parameter_dict["min_samples_split"] = dicrete_hyper_parameter_list_of_max_sample_split
discrete_parameter_dict["n_estimators"] = dicrete_hyper_parameter_list_of_estimators
discrete_parameter_dict["criterion"] = dicrete_hyper_parameter_list_of_criterion
parameter_constraint_dict = OrderedDict()

# logistic regression
param_dict_logistic_regression = OrderedDict()
param_dict_logistic_regression['max_depth'] = 1
param_dict_logistic_regression['min_samples_split'] = 1
param_dict_logistic_regression['n_estimators'] = 1
param_dict_logistic_regression['criterion'] = 'auto'
param_dict_logistic_regression['min_samples_leaf'] = 1.0
param_dict_logistic_regression['min_weight_fraction_leaf'] = 0.0
param_dict_logistic_regression['min_impurity_decrease'] = 0.0


bounds=[(0.0001, 0.5),(0.0, 0.5),(0.0, 10.99),(10.001,90.99),(1.001,90.99),(1.001,299.99),(0.001,1.99)]


def get_algorithm():
    return Algorithm(algorithm_type=RandomForestRegressor,
                     algorithm_name="RANDOM REGRESSION FOREST",
                     hyper_parameter_dict=param_dict_logistic_regression,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)