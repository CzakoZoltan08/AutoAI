# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:10:07 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.linear_model import OrthogonalMatchingPursuit

from ..Algorithm import Algorithm


const_param = {
}

dicrete_hyper_parameter_list_of_intercept = [True, False]
dicrete_hyper_parameter_list_of_normalize = [True, False]

continuous_hyper_parameter_mapping_index_key_mapping = ["tol"]
discrete_hyper_parameter_mapping = ["normalize", "fit_intercept"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["fit_intercept"] = dicrete_hyper_parameter_list_of_intercept
discrete_parameter_dict["normalize"] = dicrete_hyper_parameter_list_of_normalize
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['fit_intercept'] = True
param_dict['normalize'] = True
param_dict['tol'] = 0.001

bounds=[(0.0001,100.99),(0.001,1.99),(0.001,1.99)]


def get_algorithm():
    return Algorithm(algorithm_type=OrthogonalMatchingPursuit,
                     algorithm_name="ORTHOGONAL MATCHING PURSUIT REGRESSOR",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)