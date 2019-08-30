# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:17:02 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.ensemble import GradientBoostingClassifier 

from ..Algorithm import Algorithm


const_param = {
}

dicrete_hyper_parameter_list_of_n_estimators = range(1,2100)
dicrete_hyper_parameter_list_of_loss = ["deviance", "exponential"]
dicrete_hyper_parameter_list_of_criterion = ["friedman_mse", "mse", "mae"]

continuous_hyper_parameter_mapping_index_key_mapping = ["subsample", 
                                                        "learning_rate", 
                                                        "min_samples_split", 
                                                        "min_samples_leaf", 
                                                        "min_weight_fraction_leaf",
                                                        "min_impurity_decrease"]
discrete_hyper_parameter_mapping = ["n_estimators", "loss", "criterion"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_estimators"] = dicrete_hyper_parameter_list_of_n_estimators
discrete_parameter_dict["loss"] = dicrete_hyper_parameter_list_of_loss
discrete_parameter_dict["criterion"] = dicrete_hyper_parameter_list_of_criterion
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['n_estimators'] = 100
param_dict['loss'] = 'deviance'
param_dict['criterion'] = 'friedman_mse'
param_dict['subsample'] = 1.0
param_dict['learning_rate'] = 0.1
param_dict['min_samples_split'] = 0.1
param_dict['min_samples_leaf'] = 0.1
param_dict['min_weight_fraction_leaf'] = 0.0
param_dict['min_impurity_decrease'] = 0.0

bounds=[(0.1,0.99),(0.1,10.99),(0.001,1.00),(0.001,0.5),(0.001,0.5),(0.001,10.99),(1.99,1999.99),(0.001,0.99),(0.001,2.99)]


def get_algorithm():
    return Algorithm(algorithm_type=GradientBoostingClassifier,
                     algorithm_name="GRADIENT BOOSTING CLASSIFIER",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)