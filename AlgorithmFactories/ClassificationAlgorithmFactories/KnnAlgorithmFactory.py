# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:01:13 2019

@author: czzo
"""

from collections import OrderedDict

from sklearn.neighbors import KNeighborsClassifier 

from ..Algorithm import Algorithm


const_param = {
    'metric': 'minkowski',
    'weights': 'uniform'
}

dicrete_hyper_parameter_list_of_algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
dicrete_hyper_parameter_list_of_powers = range(1,10)
dicrete_hyper_parameter_list_of_leaf_size = range(1,120)
dicrete_hyper_parameter_list_of_neighbors = range(1,120)
continuous_hyper_parameter_mapping_index_key_mapping = []
discrete_hyper_parameter_mapping = ["p", "n_neighbors", "leaf_size","algorithm"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["p"] = dicrete_hyper_parameter_list_of_powers
discrete_parameter_dict["n_neighbors"] = dicrete_hyper_parameter_list_of_neighbors
discrete_parameter_dict["leaf_size"] = dicrete_hyper_parameter_list_of_leaf_size
discrete_parameter_dict["algorithm"] = dicrete_hyper_parameter_list_of_algorithms
parameter_constraint_dict = OrderedDict()

# logistic regression
param_dict_logistic_regression = OrderedDict()
param_dict_logistic_regression['p'] = 1
param_dict_logistic_regression['n_neighbors'] = 1
param_dict_logistic_regression['leaf_size'] = 1
param_dict_logistic_regression['algorithm'] = 'auto'


bounds=[(1.001,4.99),(1.001,99.99),(1.001,99.99),(0.001,3.99)]


def get_algorithm():
    return Algorithm(algorithm_type=KNeighborsClassifier,
                     algorithm_name="KNN",
                     hyper_parameter_dict=param_dict_logistic_regression,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)