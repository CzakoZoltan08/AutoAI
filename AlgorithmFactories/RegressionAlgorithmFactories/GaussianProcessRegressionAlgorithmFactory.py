# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:33:08 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.gaussian_process import GaussianProcessRegressor

from ..Algorithm import Algorithm


const_param = {
}

dicrete_hyper_parameter_list_of_kernel = ["linear", "rbf", "poly", "sigmoid"]

continuous_hyper_parameter_mapping_index_key_mapping = ["alpha"]
discrete_hyper_parameter_mapping = ["kernel"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["kernel"] = dicrete_hyper_parameter_list_of_kernel
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['kernel'] = "linear"
param_dict['alpha'] = 0.001

bounds=[(0.0001,10.99),(0.001,3.99)]


def get_algorithm():
    return Algorithm(algorithm_type=GaussianProcessRegressor,
                     algorithm_name="GAUSSIAN PROCESS REGRESSOR",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)