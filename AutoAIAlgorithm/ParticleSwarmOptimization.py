# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:44:59 2019

@author: czzo
"""

from AutoAIAlgorithm.Particle import Particle
from Helpers.Utils import calculate_min_distance
from Helpers.Utils import generate_initial_best_positions
from Helpers.Utils import create_all_supported_algorithm_list
from Helpers.Utils import create_all_supported_regresion_algorithm_list
from Helpers.Utils import generate_initial_particle_positions
from Helpers.Utils import generate_initial_particle_positions_for_regression

from Helpers.Utils import get_classification_algorithm_mapping
from Helpers.Utils import get_regression_algorithm_mapping

from Helpers.Utils import train_algorithm
from Helpers.Utils import train_regression_algorithm

from sklearn.metrics import f1_score

import numpy as np
import copy

import datetime

import os.path


class PSO():
    def __init__(self,
                 particle_count,
                 distance_between_initial_particles=0.7,
                 is_classification=True,
                 evaluation_metric=f1_score,
                 is_maximization=True):
        
        # establish the swarm
        self.swarm=[]
        self.particle_count=particle_count
        
        self.evaluation_metric=evaluation_metric
        
        self.cost_function=train_algorithm
        self.algorithm_mapping = get_classification_algorithm_mapping()
        algorithms = create_all_supported_algorithm_list(particle_count)
        
        initial_positions=[]
        initial_positions = generate_initial_particle_positions(num_particles=particle_count,
                                                                distance_between_initial_particles=distance_between_initial_particles) 
        
        if is_classification == False:
            self.cost_function = train_regression_algorithm
            self.algorithm_mapping = None
            self.algorithm_mapping = get_regression_algorithm_mapping()
            algorithms = create_all_supported_regresion_algorithm_list(particle_count)
            initial_positions = generate_initial_particle_positions_for_regression(num_particles=particle_count,
                                                                    distance_between_initial_particles=distance_between_initial_particles)
            
        self.num_particles=len(algorithms)
        currentDT = datetime.datetime.now()
        
        log_file_name = "Logs\\Log_{}.txt".format(str(currentDT.strftime("%Y-%m-%d_%H-%M-%S")))
        log_file_path = os.path.join( os.getcwd(), '..', log_file_name)
        self.log_file = open(log_file_path, "w")
        self.isMaximization=is_maximization
        self.initial_best_positions = generate_initial_best_positions(algorithms)

        for i in range(0, self.num_particles):
            self.swarm.append(Particle(algorithm=algorithms[i], 
                                       hyper_parameter_list=initial_positions[i],
                                       evaluation_metric=evaluation_metric,
                                       log_file=self.log_file))
    
    
    def calculate_max_distance_between_particles(self):
        max_diff = abs(self.swarm[1].metric_best_i - self.swarm[0].metric_best_i)
        min_element = self.swarm[0].metric_best_i 
      
        arr_size = len(self.swarm)
        
        for i in range(1, arr_size): 
            if (abs(self.swarm[i].metric_best_i - min_element) > max_diff): 
                max_diff = abs(self.swarm[i].metric_best_i - min_element)
          
            if (self.swarm[i].metric_best_i < min_element): 
                min_element = self.swarm[i].metric_best_i 
        return max_diff 
    
    
    def _remove_worst(self, verbose=False):
        if self.isMaximization:
            (m,i) = min((v.metric_best_i,i) for i,v in enumerate(self.swarm))
        else:
            (m,i) = max((v.metric_best_i,i) for i,v in enumerate(self.swarm))
        
        if verbose:
            print("\n* Particle {} Removed --- Algorithm Type: {} With Metric {} *".format(i, self.swarm[i].algorithm.algorithm_name, self.swarm[i].metric_best_i))
            self.log_file.write("\n* Particle {} Removed --- Algorithm Type: {} With Metric {} *".format(i, self.swarm[i].algorithm.algorithm_name, self.swarm[i].metric_best_i))
        
        self.swarm.pop(i)
    

    def _add_to_best(self, verbose=False):
        best_particle = self._get_best_particle()
        
        algorithm_type = best_particle.algorithm.algorithm_type
        
        particles = self._get_particles_by_algorithm_type(algorithm_type)
        
        current_hyper_parameters = self._generate_list_of_hyper_parameters(particles)

        if verbose:
            print("\n* Particle Added -- Algorithm Type {} *".format(best_particle.algorithm.algorithm_name))
            self.log_file.write("\n* Particle Added -- Algorithm Type {} *".format(best_particle.algorithm.algorithm_name))

        return self._generate_new_particle(best_particle, current_hyper_parameters, algorithm_type)


    def _generate_new_particle(self, best_particle, current_hyper_parameters, algorithm_type):
        dimensions = best_particle.algorithm.get_dimensions()
        distance_between_initial_particles = 0.2
        minimum_distance = 0
        while minimum_distance < distance_between_initial_particles:
           hyper_parameter_list = []  
           for j in range(dimensions):
               min_bound = best_particle.algorithm.bounds[j][0]
               max_bound = best_particle.algorithm.bounds[j][1]
               parameter_value = np.random.uniform(min_bound, max_bound)
               hyper_parameter_list.append(parameter_value)
           minimum_distance = calculate_min_distance(current_hyper_parameters, hyper_parameter_list)
        
        new_algorithm = copy.deepcopy(best_particle.algorithm)
        
        return Particle(new_algorithm, hyper_parameter_list=hyper_parameter_list, log_file=self.log_file, evaluation_metric=self.evaluation_metric)


    def _get_best_particle(self):
        if self.isMaximization:
            (m, i) = max((v.metric_best_i,i) for i,v in enumerate(self.swarm))
        else:
            (m, i) = min((v.metric_best_i,i) for i,v in enumerate(self.swarm))
        return self.swarm[i]

    def _get_particles_by_algorithm_type(self, algorithm_type):
        return [particle for particle in self.swarm if particle.algorithm.algorithm_type == algorithm_type]


    def _generate_list_of_hyper_parameters(self, particles):
        current_hyper_parameters = []
        for i in range(0,len(particles)):
            current_algorithm_hyper_parameters = particles[i].position_i
            current_hyper_parameters.append(current_algorithm_hyper_parameters)
        return current_hyper_parameters


    def _acceptance_criteria(self, particle_index, metric_best_g):
        if self.isMaximization:
            return self.swarm[particle_index].metric_best_i > metric_best_g
        else:
            return self.swarm[particle_index].metric_best_i < metric_best_g

    def fit(self,
            X_train,
            X_test,
            Y_train,
            Y_test,
            maxiter=20,
            verbose=False,
            max_distance=0.05):
        
        metric_best_g=-1                   # best error for group
        
        if self.isMaximization == False:
            metric_best_g = 9999999999999999
        
        pos_best_g=self.initial_best_positions                   # best position for group
        model_best_g=any
        
        # begin optimization loop
        i=0
        while i < maxiter:
            if verbose:
                print("--- START EPOCH {} ---".format(i))
                self.log_file.write("--- START EPOCH {} ---\n".format(i))
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,self.num_particles):               
                self.swarm[j].evaluate(
                        self.cost_function,
                        X_train,
                        X_test,
                        Y_train,
                        Y_test,
                        epoch=i,
                        verbose=verbose)

                # determine if current particle is the best (globally)
                if self._acceptance_criteria(j,metric_best_g):
                    golbal_best_position_index=self.algorithm_mapping[self.swarm[j].algorithm.algorithm_name]
                    pos_best_g[golbal_best_position_index]=list(self.swarm[j].position_i)
                    metric_best_g=float(self.swarm[j].metric_best_i)
                    model_best_g=self.swarm[j].model_best_i
                    
                if verbose:
                    print("* Particle {} Algorithm Type {}: personal best metric={} *".format(j, self.swarm[j].algorithm.algorithm_name, self.swarm[j].metric_best_i))
                    self.log_file.write("* Particle {} Algorithm Type {}: personal best metric={} *".format(j, self.swarm[j].algorithm.algorithm_name, self.swarm[j].metric_best_i))

            max_distance_between_particle = self.calculate_max_distance_between_particles()
            
            if max_distance_between_particle < max_distance:
                return metric_best_g, model_best_g

            particle_to_remove = 1
            if self.particle_count > 2:
                particle_to_remove = int(self.particle_count/2)

            for k in range(particle_to_remove):
                self._remove_worst(verbose)
                new_particle = self._add_to_best(verbose)
            
                new_particle.evaluate(
                        self.cost_function,
                        X_train,
                        X_test,
                        Y_train,
                        Y_test,
                        epoch=i,
                        verbose=verbose)
                
                self.swarm.append(new_particle)
            
            if self.num_particles <= 0:
                self.log_file.close()
                return pos_best_g, metric_best_g, model_best_g

            # cycle through swarm and update velocities and position
            for j in range(0,self.num_particles):
                golbal_best_position_index=self.algorithm_mapping[self.swarm[j].algorithm.algorithm_name]
                self.swarm[j].update_velocity(pos_best_g[golbal_best_position_index])
                self.swarm[j].update_position()
            i+=1
            if verbose:
                print("--- END EPOCH {} ---".format(i))
                self.log_file.write("--- END EPOCH {} ---\n".format(i))

        # print final results
        print('FINAL PSO:')
        print(pos_best_g)
        print(metric_best_g)
        print(model_best_g)
        
        self.log_file.close()
        return metric_best_g, model_best_g