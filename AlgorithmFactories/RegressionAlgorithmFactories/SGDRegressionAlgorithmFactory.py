# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:18:38 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.linear_model import SGDRegressor 

from ..Algorithm import Algorithm


const_param = {
}

dicrete_hyper_parameter_list_of_max_iter  = range(1,2100)
dicrete_hyper_parameter_list_of_loss = ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]
dicrete_hyper_parameter_list_of_penalty = ["none", "l2", "l1", "elasticnet"]
dicrete_hyper_parameter_list_of_learning_rate = ["constant", "optimal", "invscaling", "adaptive"]

continuous_hyper_parameter_mapping_index_key_mapping = ["alpha", 
                                                        "l1_ratio", 
                                                        "epsilon", 
                                                        "eta0", 
                                                        "power_t"]
discrete_hyper_parameter_mapping = ["max_iter", "loss", "penalty", "learning_rate"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["max_iter"] = dicrete_hyper_parameter_list_of_max_iter
discrete_parameter_dict["loss"] = dicrete_hyper_parameter_list_of_loss
discrete_parameter_dict["penalty"] = dicrete_hyper_parameter_list_of_penalty
discrete_parameter_dict["learning_rate"] = dicrete_hyper_parameter_list_of_learning_rate
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['max_iter'] = 5
param_dict['loss'] = 'squared_loss'
param_dict['learning_rate'] = 'constant'
param_dict['penalty'] = 'none'
param_dict['alpha'] = 0.0001
param_dict['l1_ratio'] = 0.15
param_dict['epsilon'] = 0.1
param_dict['eta0'] = 0.0
param_dict['power_t'] = 0.5

bounds=[(0.1,10.99),(0.0,1.00),(0.001,10.99),(0.0001,10.99),(0.001,10.99),(0.001,1999.99),(0.001,3.99),(0.001,3.99),(0.001,2.99)]


def get_algorithm():
    return Algorithm(algorithm_type=SGDRegressor,
                     algorithm_name="SGD REGRESSOR",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)