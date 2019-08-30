# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:05:14 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.ensemble import ExtraTreesRegressor

from ..Algorithm import Algorithm


const_param = {
    'max_depth': None
}

dicrete_hyper_parameter_list_of_criterion = ["mse", "mae"]
dicrete_hyper_parameter_list_of_n_estimators = range(1,320)
dicrete_hyper_parameter_list_of_min_samples_split  = range(1,240)
dicrete_hyper_parameter_list_of_min_samples_leaf = range(1,240)
dicrete_hyper_parameter_list_of_min_samples_leaf = range(1,240)

continuous_hyper_parameter_mapping_index_key_mapping = ["min_impurity_decrease", "min_weight_fraction_leaf"]
discrete_hyper_parameter_mapping = ["min_samples_split", "min_samples_leaf", "n_estimators", "criterion"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["min_samples_split"] = dicrete_hyper_parameter_list_of_min_samples_split
discrete_parameter_dict["min_samples_leaf"] = dicrete_hyper_parameter_list_of_min_samples_leaf
discrete_parameter_dict["criterion"] = dicrete_hyper_parameter_list_of_criterion
discrete_parameter_dict["n_estimators"] = dicrete_hyper_parameter_list_of_n_estimators
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['min_impurity_decrease'] = 0.0
param_dict['min_weight_fraction_leaf'] = 0.0
param_dict['min_samples_split'] = 1
param_dict['min_samples_leaf'] = 1
param_dict['criterion'] = 'mse'
param_dict['n_estimators'] = 100


bounds=[(1.001,50.99),(0.0,0.5),(1.001,199.99),(1.001,199.99),(1.001,299.99),(0.001,1.99)]


def get_algorithm():
    return Algorithm(algorithm_type=ExtraTreesRegressor,
                     algorithm_name="EXTRA TREES REGRESSION",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)