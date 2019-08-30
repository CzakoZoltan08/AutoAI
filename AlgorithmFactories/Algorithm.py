# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:30:02 2019

@author: czzo
"""


class Algorithm:
    def __init__(self,
                 algorithm_type,
                 algorithm_name,
                 hyper_parameter_dict,
                 discrete_hyper_parameter_dict,
                 discrete_hyper_parameter_mapping,
                 continuous_hyper_parameter_mapping,
                 parameter_constraint_dict,
                 constant_hyper_parameter_dict,
                 bounds):
        self.hyper_parameter_dict=hyper_parameter_dict
        self.algorithm_type=algorithm_type
        self.algorithm_name=algorithm_name
        self.discrete_hyper_parameter_dict=discrete_hyper_parameter_dict
        self.discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping
        self.continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping
        self.parameter_constraint_dict=parameter_constraint_dict
        self.constant_hyper_parameter_dict=constant_hyper_parameter_dict
        self.bounds=bounds
    
    def get_dimensions(self):
        return len(self.bounds)