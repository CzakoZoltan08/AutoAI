# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.linear_model import ARDRegression

from ..Algorithm import Algorithm


const_param = {
    'copy_X': True
}

dicrete_hyper_parameter_list_of_intercept = [True, False]
dicrete_hyper_parameter_list_of_normalize = [True, False]
dicrete_hyper_parameter_list_of_n_iter  = range(10,3200)

continuous_hyper_parameter_mapping_index_key_mapping = ["alpha_1", "alpha_2", "lambda_1", "lambda_2", "threshold_lambda"]
discrete_hyper_parameter_mapping = ["normalize", "fit_intercept", "n_iter"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["fit_intercept"] = dicrete_hyper_parameter_list_of_intercept
discrete_parameter_dict["normalize"] = dicrete_hyper_parameter_list_of_normalize
discrete_parameter_dict["n_iter"] = dicrete_hyper_parameter_list_of_n_iter
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['fit_intercept'] = True
param_dict['normalize'] = True
param_dict['n_iter'] = 300
param_dict['alpha_1'] = 0.000001
param_dict['alpha_2'] = 0.000001
param_dict['lambda_1'] = 0.000001
param_dict['lambda_2'] = 0.000001
param_dict['threshold_lambda'] = 0.000001

bounds=[(0.000001,100.99),(0.000001,100.99),(0.000001,100.99),(0.000001,100.99),(0.000001,100.99),(0.001,1.99),(0.001,1.99),(10.99,2999.99)]


def get_algorithm():
    return Algorithm(algorithm_type=ARDRegression,
                     algorithm_name="ARD REGRESSOR",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)