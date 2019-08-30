# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:06:12 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.ensemble import AdaBoostClassifier 

from ..Algorithm import Algorithm


const_param = {
}

dicrete_hyper_parameter_list_of_algorithm = ["SAMME", "SAMME.R"]
dicrete_hyper_parameter_list_of_n_estimators = range(1,2100)

continuous_hyper_parameter_mapping_index_key_mapping = ["learning_rate"]
discrete_hyper_parameter_mapping = ["n_estimators", "algorithm"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_estimators"] = dicrete_hyper_parameter_list_of_n_estimators
discrete_parameter_dict["algorithm"] = dicrete_hyper_parameter_list_of_algorithm
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['n_estimators'] = 50
param_dict['algorithm'] = "SAMME"
param_dict['learning_rate'] = 1.0

bounds=[(0.001,100.99),(0.001,1999.99),(0.001,1.99)]


def get_algorithm():
    return Algorithm(algorithm_type=AdaBoostClassifier,
                     algorithm_name="ADABOOST CLASSIFIER",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)