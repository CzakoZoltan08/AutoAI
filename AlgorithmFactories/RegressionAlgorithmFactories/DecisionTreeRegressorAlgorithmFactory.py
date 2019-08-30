# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 20:00:57 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.tree import DecisionTreeRegressor 

from ..Algorithm import Algorithm


const_param = {
    'max_depth': None
}

dicrete_hyper_parameter_list_of_criterion = ["mse", "friedman_mse", "mae"]
dicrete_hyper_parameter_list_of_splitter = ["best", "random"]
dicrete_hyper_parameter_list_of_min_samples_split  = range(1,240)
dicrete_hyper_parameter_list_of_min_samples_leaf = range(1,240)
dicrete_hyper_parameter_list_of_min_samples_leaf = range(1,240)

continuous_hyper_parameter_mapping_index_key_mapping = ["min_impurity_decrease"]
discrete_hyper_parameter_mapping = ["min_samples_split", "min_samples_leaf", "splitter", "criterion"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["min_samples_split"] = dicrete_hyper_parameter_list_of_min_samples_split
discrete_parameter_dict["min_samples_leaf"] = dicrete_hyper_parameter_list_of_min_samples_leaf
discrete_parameter_dict["criterion"] = dicrete_hyper_parameter_list_of_criterion
discrete_parameter_dict["splitter"] = dicrete_hyper_parameter_list_of_splitter
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['min_impurity_decrease'] = 0.0
param_dict['min_samples_split'] = 1
param_dict['min_samples_leaf'] = 1
param_dict['criterion'] = 'mse'
param_dict['splitter'] = 'best'


bounds=[(1.001,50.99),(1.001,199.99),(1.001,199.99),(0.001,1.99),(0.001,2.99)]


def get_algorithm():
    return Algorithm(algorithm_type=DecisionTreeRegressor,
                     algorithm_name="DECISION TREE REGRESSOR",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)