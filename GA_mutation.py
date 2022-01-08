# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 14:26:55 2022

@author: void
"""


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime
from PIL import Image 
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import pygad
import numpy
from sklearn import metrics
from scipy.spatial.distance import cdist


def euclidean_distance(X, Y):
    # for i in range(num_clusters):
        
    a = numpy.sqrt(numpy.sum(numpy.power(X - Y, 2), axis=1))
    # print(a)
    return a

def cluster_data(solution, solution_idx):
    global num_clusters, feature_vector_length, data
    cluster_centers = []
    all_clusters_dists = []
    clusters = []
    clusters_sum_dist = []

    for clust_idx in range(num_clusters):
        cluster_centers.append(solution[feature_vector_length*clust_idx:feature_vector_length*(clust_idx+1)])
        cluster_center_dists = euclidean_distance(data, cluster_centers[clust_idx])
        all_clusters_dists.append(numpy.array(cluster_center_dists))

    cluster_centers = numpy.array(cluster_centers)
    all_clusters_dists = numpy.array(all_clusters_dists)

    cluster_indices = numpy.argmin(all_clusters_dists, axis=0)
    # print(cluster_indices)
    for clust_idx in range(num_clusters):
        # print(numpy.where(cluster_indices == clust_idx))
        clusters.append(np.where(cluster_indices == clust_idx)[0])
        if len(clusters[clust_idx]) == 0:
            clusters_sum_dist.append(0)
        else:
            clusters_sum_dist.append(numpy.sum(all_clusters_dists[clust_idx, clusters[clust_idx]]))

    clusters_sum_dist = numpy.array(clusters_sum_dist)
    # print(cluster_indices,clusters_sum_dist)

    return cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist

def fitness_func(solution, solution_idx):
    _, _, _, _, clusters_sum_dist = cluster_data(solution, solution_idx)
    
    a = numpy.sqrt(numpy.sum(numpy.power(clusters_sum_dist, 2), axis=0))
    
    fitness = 1.0 / (numpy.sum(clusters_sum_dist) + 0.00000001)
    # fitness = 1.0 / (a + 0.00000001)
    # fitness = 1.0 / (np.log(a) + 0.00000001)
    # print(fitness)
    # fitness = numpy.sum(clusters_sum_dist)

    return fitness


    


if __name__ == "__main__":
    #load image
    img = Image.open("kmeans.png")
    img_arr = np.array(img)
    vectorized = img_arr.reshape((-1,3))
    vectorized = np.float32(vectorized)
    data = vectorized

    num_clusters = 4
    data = vectorized
    feature_vector_length = data.shape[1]
    num_genes = num_clusters * feature_vector_length
    fig = plt.figure()
    bes_fit = []
    best_solut = []
    
    data = vectorized
    #random mutation
    ga_instance_1 = pygad.GA(num_generations=30,
                           sol_per_pop=20,
                           init_range_low=0.0,
                           init_range_high=255.0,
                           num_parents_mating=5,
                           keep_parents=-1,
                           num_genes=num_genes,
                            # mutation_type="adaptive",
                            mutation_type="random",
                            # mutation_probability = [0.5, 0.3],
                           fitness_func=fitness_func,
                           parent_selection_type="tournament",
                           K_tournament=6,
                           # crossover_type="single_point",
                           crossover_probability=0.6,
                           mutation_percent_genes=20,
                           # random_mutation_min_val=0,
                           # random_mutation_max_val=1,
                           save_best_solutions=True,
                           suppress_warnings=True)
    
    #adaptive mutation
    ga_instance_2 = pygad.GA(num_generations=30,
                           sol_per_pop=20,
                           init_range_low=0.0,
                           init_range_high=255.0,
                           num_parents_mating=5,
                           keep_parents=-1,
                           num_genes=num_genes,
                            mutation_type="adaptive",
                           # mutation_type="random",
                            mutation_probability = [0.5, 0.3],
                           fitness_func=fitness_func,
                           parent_selection_type="tournament",
                           K_tournament=6,
                           # crossover_type="single_point",
                           crossover_probability=0.6,
                           mutation_percent_genes=20,
                           # random_mutation_min_val=0,
                           # random_mutation_max_val=1,
                           save_best_solutions=True,
                           suppress_warnings=True)
    #scarmble mutation
    ga_instance_3 = pygad.GA(num_generations=30,
                           sol_per_pop=20,
                           init_range_low=0.0,
                           init_range_high=255.0,
                           num_parents_mating=5,
                           keep_parents=-1,
                           num_genes=num_genes,
                            mutation_type="scramble",
                           # mutation_type="random",
                            # mutation_probability = [0.5, 0.3],
                           fitness_func=fitness_func,
                           parent_selection_type="tournament",
                           K_tournament=6,
                           # crossover_type="single_point",
                           crossover_probability=0.6,
                           mutation_percent_genes=20,
                           # random_mutation_min_val=0, 
                           # random_mutation_max_val=1,
                           save_best_solutions=True,
                           suppress_warnings=True)
    # print(ga_instance.initial_population)
    ga_instance_1.run()
    ga_instance_2.run()
    ga_instance_3.run()
    
    best_solution, best_solution_fitness, best_solution_idx = ga_instance_1.best_solution()
    cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx)
    a_1=ga_instance_1.best_solutions_fitness
    
    best_solution, best_solution_fitness, best_solution_idx = ga_instance_2.best_solution()
    cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx)
    a_2=ga_instance_2.best_solutions_fitness
    
    best_solution, best_solution_fitness, best_solution_idx = ga_instance_3.best_solution()
    cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx)
    a_3=ga_instance_3.best_solutions_fitness
    
    
    # best_solution, best_solution_fitness, best_solution_idx = ga_instance_1.best_solution()
    # cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx)
    # a_1=ga_instance_1.best_solutions_fitness
    
    # best_solution, best_solution_fitness, best_solution_idx = ga_instance_2.best_solution()
    # cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx)
    # a_2=ga_instance_2.best_solutions_fitness
    
    # best_solution, best_solution_fitness, best_solution_idx = ga_instance_3.best_solution()
    # cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx)
    # a_3=ga_instance_3.best_solutions_fitness
    #plot the result
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.plot(a_1, linewidth=3, color="r")
    
    plt.plot(a_2, linewidth=3, color="b")
    
    plt.plot(a_3, linewidth=3, color="g")
    
    plt.show()