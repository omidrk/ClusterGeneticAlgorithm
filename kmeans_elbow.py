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

if __name__ == "__main__":
    #load image
    img = Image.open("kmeans.png")
    img_arr = np.array(img)
    vectorized = img_arr.reshape((-1,3))
    vectorized = np.float32(vectorized)
    
    #run kmeans with elbow method
    iner = []
    for i in range(10):
        kmeans = KMeans(n_clusters=int(i+1))
        kmeans.fit(vectorized)
        labels = kmeans.predict(vectorized)
        iner.append(kmeans.inertia_)
        # lbimd = labels.reshape((img_arr.shape[0],img_arr.shape[1],1))
    #%%
    plt.plot(range(10),iner)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.ylabel("inertia")
    plt.xlabel("number of clusters")
    plt.show()