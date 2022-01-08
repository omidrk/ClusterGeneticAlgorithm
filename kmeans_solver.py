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
    print(vectorized.shape)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(vectorized)
    labels = kmeans.predict(vectorized)
    lbimd = labels.reshape((img_arr.shape[0],img_arr.shape[1],1))
    # print(lbimd.shape)
    plt.imshow(lbimd)