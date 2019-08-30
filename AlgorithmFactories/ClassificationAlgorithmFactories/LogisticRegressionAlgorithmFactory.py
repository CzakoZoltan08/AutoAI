# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:28:07 2019

@author: czzo
"""

from collections import OrderedDict

from sklearn.linear_model import LogisticRegression

from ..Algorithm import Algorithm



const_param_logistic_regression = {
    'verbose': 0,
    'dual':  False,
    'class_weight': 'balanced',
    'penalty': 'l1'
}

dicrete_hyper_parameter_list_of_solvers = ["liblinear", "saga", "newton-cg", "lbfgs", "sag"]

continuous_hyper_parameter_mapping_index_key_mapping = ["C", "tol", "intercept_scaling"]
discrete_hyper_parameter_mapping = ["solver"]

discrete_parameter_dict_logistic_regression = OrderedDict()
discrete_parameter_dict_logistic_regression["solver"] = dicrete_hyper_parameter_list_of_solvers

parameter_constraint_dict = OrderedDict()
parameter_constraint_dict['sag'] = [('penalty','l2')]
parameter_constraint_dict['saga'] = [('penalty','l1')]
parameter_constraint_dict['newton-cg'] = [('penalty','l2')]
parameter_constraint_dict['lbfgs'] = [('penalty','l2')]
parameter_constraint_dict['liblinear'] = [('penalty','l1')]

# logistic regression
param_dict_logistic_regression = OrderedDict()
param_dict_logistic_regression['tol'] = 0.0
param_dict_logistic_regression['C'] = 0.0
param_dict_logistic_regression['solver'] = 'liblinear'
param_dict_logistic_regression['intercept_scaling'] = 0.0
param_dict_logistic_regression['max_iter'] = 1000
param_dict_logistic_regression['penalty'] = 'l1'

bounds=[(0.001,3),(0.001,2),(0.001,1),(0.0, 4.99)] 


def get_algorithm():
    return Algorithm(algorithm_type=LogisticRegression,
                       algorithm_name="LOGISTIC_REGRESSION",
                       hyper_parameter_dict=param_dict_logistic_regression,
                       discrete_hyper_parameter_dict=discrete_parameter_dict_logistic_regression,
                       discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                       continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                       parameter_constraint_dict=parameter_constraint_dict,
                       constant_hyper_parameter_dict=const_param_logistic_regression,
                       bounds=bounds)