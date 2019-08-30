# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:37:05 2019

@author: czzo
"""


import random

import numpy as np


class Particle:
    def __init__(self,
                 algorithm,
                 hyper_parameter_list,
                 evaluation_metric,
                 log_file,
                 T_0=0.4,
                 alpha_0=0.85,
                 T_min0=0.05,
                 update_iters0=5,
                 beta0=1.3,
                 shouldMaximize=True):
        self.algorithm=algorithm
        self.T=T_0
        self.T_min=T_min0
        self.update_iters=update_iters0
        self.alpha=alpha_0
        self.beta=beta0
        self.position_previous_i=[]
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.metric_best_i=-1       # best error individual
        self.metric_i=-1            # error individual
        self.metric_previous_i=-1
        self.model_best_i=any
        self.model_i=any
        self.log_file=log_file
        self.shouldMaximize=shouldMaximize
        self.evaluation_metric=evaluation_metric

        for i in range(0, self.algorithm.get_dimensions()):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(hyper_parameter_list[i])
            self.pos_best_i.append(hyper_parameter_list[i])


    def set_parameter_contstraints(self, parameter_name, current_value):
        if current_value in self.algorithm.parameter_constraint_dict:
            if parameter_name in self.algorithm.hyper_parameter_dict:
                for (p_name, v) in self.algorithm.parameter_constraint_dict[current_value]:
                    self.algorithm.hyper_parameter_dict[p_name] = v
                
                
    def update_current_parameters(self, verbose=False):
        i=0
        for i in range(len(self.algorithm.continuous_hyper_parameter_mapping)):
            parameter_name = self.algorithm.continuous_hyper_parameter_mapping[i]
            current_value = self.position_i[i]
            self.algorithm.hyper_parameter_dict[parameter_name] = current_value
            self.set_parameter_contstraints(parameter_name, current_value)
            if verbose:
                print("* Parameter={} Value={} *".format(parameter_name, self.algorithm.hyper_parameter_dict[parameter_name]))
                self.log_file.write("* Parameter={} Value={} \n*".format(parameter_name, self.algorithm.hyper_parameter_dict[parameter_name]))
          
        if len(self.algorithm.continuous_hyper_parameter_mapping) >= 1:
            i+=1
            
        for j in range(len(self.algorithm.discrete_hyper_parameter_mapping)):
            parameter_name = self.algorithm.discrete_hyper_parameter_mapping[j]
            current_position = self.position_i[i+j]
            whole, fraction = divmod(current_position, 1)
            current_index = int(whole)
            current_value = self.algorithm.discrete_hyper_parameter_dict[parameter_name][current_index]
            self.algorithm.hyper_parameter_dict[parameter_name] = current_value
            self.set_parameter_contstraints(parameter_name, current_value)
            if verbose:
                print("* Parameter={} Value={} *".format(parameter_name, self.algorithm.hyper_parameter_dict[parameter_name]))
                self.log_file.write("* Parameter={} Value={} \n*".format(parameter_name, self.algorithm.hyper_parameter_dict[parameter_name]))
        

    # evaluate current fitness
    def evaluate(self,
                 fn_train,
                 X_train,
                 X_valid,
                 Y_train,
                 Y_valid,
                 epoch,
                 verbose=False):
        
        self.update_current_parameters(verbose)
        
        self.model_i, self.metric_i = fn_train(self.algorithm.hyper_parameter_dict, 
                                               self.algorithm.constant_hyper_parameter_dict, 
                                               X_train, 
                                               X_valid, 
                                               Y_train, 
                                               Y_valid,
                                               self.algorithm.algorithm_type,
                                               metric=self.evaluation_metric)

        if self.acceptance_criteria_previous_metric():
            self.position_previous_i = self.position_i
            self.metric_previous_i = self.metric_i
            
            if self.acceptance_criteria_best_metric():
                self.metric_best_i = self.metric_i
                self.model_best_i = self.model_i
                self.pos_best_i = self.position_i
        else:
            rnd = np.random.uniform()
            diff = self.metric_i - self.metric_previous_i
            threshold = np.exp(self.beta * diff / self.T)
            if rnd < threshold:
                self.position_previous_i = self.position_i
                self.metric_previous_i = self.metric_i
                
        if epoch % self.update_iters == 0:
            self.T = self.alpha * self.T
            
    
    def acceptance_criteria_previous_metric(self):
        if self.shouldMaximize:
            return self.metric_i > self.metric_previous_i
        else:
            return self.metric_i < self.metric_previous_i
            
    def acceptance_criteria_best_metric(self):
        if self.shouldMaximize:
            return self.metric_i > self.metric_best_i
        else:
            return self.metric_i < self.metric_best_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=3       # social constant

        dimensions = self.algorithm.get_dimensions()
        for i in range(0,dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social


    # update the particle position based off new velocity updates
    def update_position(self):
        dimensions = self.algorithm.get_dimensions()
        for i in range(0, dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > self.algorithm.bounds[i][1]:
                self.position_i[i]=self.algorithm.bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < self.algorithm.bounds[i][0]:
                self.position_i[i]=self.algorithm.bounds[i][0]