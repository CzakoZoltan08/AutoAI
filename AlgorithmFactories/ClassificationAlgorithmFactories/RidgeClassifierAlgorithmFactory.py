# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:38:49 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.linear_model import RidgeClassifier 

from ..Algorithm import Algorithm


const_param = {
    'copy_X': True,
    'class_weight': 'balanced'
}

dicrete_hyper_parameter_list_of_intercept = [True, False]
dicrete_hyper_parameter_list_of_normalize = [True, False]
dicrete_hyper_parameter_list_of_solver = ["auto", "svd", "cholesky", "sparse_cg", "lsqr", "sag"]

continuous_hyper_parameter_mapping_index_key_mapping = ["alpha", "tol"]
discrete_hyper_parameter_mapping = ["normalize", "fit_intercept", "solver"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["fit_intercept"] = dicrete_hyper_parameter_list_of_intercept
discrete_parameter_dict["normalize"] = dicrete_hyper_parameter_list_of_normalize
discrete_parameter_dict["solver"] = dicrete_hyper_parameter_list_of_solver
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['fit_intercept'] = True
param_dict['normalize'] = True
param_dict['alpha'] = 0.1
param_dict['tol'] = 0.1


bounds=[(0.001,100.99),(0.001,100.99),(0.001,1.99),(0.001,1.99),(0.001, 5.99)]


def get_algorithm():
    return Algorithm(algorithm_type=RidgeClassifier,
                     algorithm_name="RIDGE CLASSIFIER",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)