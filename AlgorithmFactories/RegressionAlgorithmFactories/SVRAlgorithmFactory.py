# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:39:05 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.svm import SVR 

from ..Algorithm import Algorithm


const_param = {
}

dicrete_hyper_parameter_list_of_shrinking = [True, False]
dicrete_hyper_parameter_list_of_degree = range(0, 210)
dicrete_hyper_parameter_list_of_kernel = ["linear", "poly", "rbf", "sigmoid"]

continuous_hyper_parameter_mapping_index_key_mapping = ["C", "gamma", "coef0", "epsilon"]
discrete_hyper_parameter_mapping = ["shrinking", "degree", "kernel"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["shrinking"] = dicrete_hyper_parameter_list_of_shrinking
discrete_parameter_dict["degree"] = dicrete_hyper_parameter_list_of_degree
discrete_parameter_dict["kernel"] = dicrete_hyper_parameter_list_of_kernel
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['shrinking'] = True
param_dict['degree'] = 3
param_dict['kernel'] = "rbf"
param_dict['C'] = 1.0
param_dict['gamma'] = 0.1
param_dict['coef0'] = 0.1
param_dict['epsilon'] = 0.1

bounds=[(0.0001,3.99),(0.0001,1.99),(0.0001,100.99),(0.0001,100.99),(0.001,1.99),(0.001,199.99),(0.001, 3.99)]


def get_algorithm():
    return Algorithm(algorithm_type=SVR,
                     algorithm_name="SVR REGRESSION",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)