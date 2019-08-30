# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 22:54:22 2019

@author: Zoltan
"""

from collections import OrderedDict

from xgboost import XGBClassifier

from ..Algorithm import Algorithm


const_param = {
}

dicrete_hyper_parameter_list_of_n_estimators = range(1,320)
dicrete_hyper_parameter_list_of_min_child_weight = range(1,320)
dicrete_hyper_parameter_list_of_max_delta_step = range(1,320)
dicrete_hyper_parameter_list_of_max_depth = range(1,320)

continuous_hyper_parameter_mapping_index_key_mapping = ["gamma", 
                                                        "subsample"
                                                        "colsample_bytree",
                                                        "colsample_bylevel",
                                                        "colsample_bynode",
                                                        "reg_alpha",
                                                        "reg_lambda",
                                                        "scale_pos_weigh"]
discrete_hyper_parameter_mapping = ["min_child_weight", 
                                    "max_delta_step", 
                                    "n_estimators", 
                                    "max_depth"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["max_depth"] = dicrete_hyper_parameter_list_of_max_depth
discrete_parameter_dict["max_delta_step"] = dicrete_hyper_parameter_list_of_max_delta_step
discrete_parameter_dict["min_child_weight"] = dicrete_hyper_parameter_list_of_min_child_weight
discrete_parameter_dict["n_estimators"] = dicrete_hyper_parameter_list_of_n_estimators
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['gamma'] = 0.0
param_dict['subsample'] = 0.0
param_dict['colsample_bytree'] = 0.0
param_dict['colsample_bylevel'] = 0.0
param_dict['colsample_bynode'] = 0.0
param_dict['reg_alpha'] = 0.0
param_dict['reg_lambda'] = 0.0
param_dict['scale_pos_weigh'] = 0.0

param_dict['n_estimators'] = 100
param_dict['min_child_weight'] = 10
param_dict['max_delta_step'] = 10
param_dict['max_depth'] = 100

bounds=[(0.001,10.99),(0.001,10.99),(0.0,1.0),(0.0,1.0),(0.001,10.99),(0.0,1.0),(0.001,10.99),(0.001,10.99),
        (1.001,299.99),(1.001,299.99),(1.001,299.99),(1.001,299.99)]


def get_algorithm():
    return Algorithm(algorithm_type=XGBClassifier,
                     algorithm_name="XGBOOST CLASSIFIER",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)