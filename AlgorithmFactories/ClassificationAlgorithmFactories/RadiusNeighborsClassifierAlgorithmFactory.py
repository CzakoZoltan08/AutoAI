# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:16:02 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.neighbors import RadiusNeighborsClassifier 

from ..Algorithm import Algorithm


const_param = {
    'weights': 'uniform',
    'leaf_size': 30
}

dicrete_hyper_parameter_list_of_algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
dicrete_hyper_parameter_list_of_weights = ["uniform", "distance"]
dicrete_hyper_parameter_list_of_powers = range(1,10)
continuous_hyper_parameter_mapping_index_key_mapping = ["radius"]
discrete_hyper_parameter_mapping = ["p", "weights","algorithm"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["p"] = dicrete_hyper_parameter_list_of_powers
discrete_parameter_dict["weights"] = dicrete_hyper_parameter_list_of_weights
discrete_parameter_dict["algorithm"] = dicrete_hyper_parameter_list_of_algorithms
parameter_constraint_dict = OrderedDict()

# logistic regression
param_dict_logistic_regression = OrderedDict()
param_dict_logistic_regression['p'] = 1
param_dict_logistic_regression['weights'] = 'uniform'
param_dict_logistic_regression['algorithm'] = 'auto'
param_dict_logistic_regression['radius'] = 1.0


bounds=[(10.001,300.99),(1.001,4.99),(0.001,1.99),(0.001,3.99)]


def get_algorithm():
    return Algorithm(algorithm_type=RadiusNeighborsClassifier,
                     algorithm_name="RADIUS NEIGHBORS CLASSIFIER",
                     hyper_parameter_dict=param_dict_logistic_regression,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)