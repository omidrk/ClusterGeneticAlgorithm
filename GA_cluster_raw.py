# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 14:17:25 2022

@author: void
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:07:02 2022

@author: jadidi
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

def plot_me(best_sul, 
                    title="PyGAD - Generation vs. Fitness", 
                    xlabel="Generation", 
                    ylabel="Fitness", 
                    linewidth=3, 
                    font_size=14, 
                    plot_type="plot",
                    color="#3870FF",
                    save_dir=None):
    

        """
        Creates, shows, and returns a figure that summarizes how the fitness value evolved by generation. Can only be called after completing at least 1 generation. If no generation is completed, an exception is raised.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            color: Color of the plot which defaults to "#3870FF".
            save_dir: Directory to save the figure.

        Returns the figure.
        """

        if ga_instance.generations_completed < 1:
            raise RuntimeError("The plot_fitness() (i.e. plot_result()) method can only be called after completing at least 1 generation but ({generations_completed}) is completed.".format(generations_completed=ga_instance.generations_completed))

#        if self.run_completed == False:
#            if not self.suppress_warnings: warnings.warn("Warning calling the plot_result() method: \nGA is not executed yet and there are no results to display. Please call the run() method before calling the plot_result() method.\n")
        
        fig = plt.figure()
        for i in best_sul:
            plt.plot(i, linewidth=linewidth, color=color)
            
        plt.title(title, fontsize=font_size)
        plt.xlabel(xlabel, fontsize=font_size)
        plt.ylabel(ylabel, fontsize=font_size)
        
        if not save_dir is None:
            fig.savefig(fname=save_dir, 
                                      bbox_inches='tight')
        plt.show()

        return fig
    
    
if __name__ == "__main__":
    #load image
    img = Image.open("kmeans.png")
    img_arr = np.array(img)
    vectorized = img_arr.reshape((-1,3))
    vectorized = np.float32(vectorized)
    data = vectorized
    
    num_clusters = 4
    feature_vector_length = data.shape[1]
    num_genes = num_clusters * feature_vector_length
    
    #define GA class
    
    ga_instance = pygad.GA(num_generations=30,
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
    
    print("Initial Population")
    # run GA
    ga_instance.run()
    #extract solution from GA class
    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    print("Best solution is {bs}".format(bs=best_solution))
    print("Fitness of the best solution is {bsf}".format(bsf=best_solution_fitness))
    print("Best solution found after {gen} generations".format(gen=ga_instance.best_solution_generation))
    cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx)
    
    #plot cluster result
    lbimd = cluster_indices.reshape((img_arr.shape[0],img_arr.shape[1],1))
    print(lbimd.shape)
    plt.imshow(lbimd)
    
    #plot fitness function
    ga_instance.save_solutions=True
    ga_instance.plot_fitness()

    
  