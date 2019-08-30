# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:54:44 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.linear_model import PassiveAggressiveClassifier 

from ..Algorithm import Algorithm


const_param = {
    
}

dicrete_hyper_parameter_list_of_intercept = [True, False]
dicrete_hyper_parameter_list_of_max_iter  = range(5, 2100)

continuous_hyper_parameter_mapping_index_key_mapping = ["C", "tol"]
discrete_hyper_parameter_mapping = ["max_iter", "fit_intercept"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["fit_intercept"] = dicrete_hyper_parameter_list_of_intercept
discrete_parameter_dict["max_iter"] = dicrete_hyper_parameter_list_of_max_iter
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['fit_intercept'] = True
param_dict['max_iter'] = 5
param_dict['C'] = 0.1
param_dict['tol'] = 0.1


bounds=[(0.001,100.99),(0.001,100.99),(1.001,1999.99),(0.001,1.99)]


def get_algorithm():
    return Algorithm(algorithm_type=PassiveAggressiveClassifier,
                     algorithm_name="PASSIVE AGRESSIVE CLASSIFIER",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)