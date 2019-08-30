# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 01:32:56 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.linear_model import LinearRegression 

from ..Algorithm import Algorithm


const_param = {
    'copy_X': True
}

dicrete_hyper_parameter_list_of_intercept = [True, False]
dicrete_hyper_parameter_list_of_normalize = [True, False]

continuous_hyper_parameter_mapping_index_key_mapping = []
discrete_hyper_parameter_mapping = ["normalize", "fit_intercept"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["fit_intercept"] = dicrete_hyper_parameter_list_of_intercept
discrete_parameter_dict["normalize"] = dicrete_hyper_parameter_list_of_normalize
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['fit_intercept'] = True
param_dict['normalize'] = True


bounds=[(0.001,1.99),(0.001,1.99)]


def get_algorithm():
    return Algorithm(algorithm_type=LinearRegression,
                     algorithm_name="LINEAR REGRESSOR",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)