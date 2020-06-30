#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:25:51 2020

@author: aarav
"""

import numpy as np
import matplotlib.pyplot as plt

"""---------------------------------------------------------------------------"""

def find_error(centrds,ans,rows_count):
    # Total Square Error
    t_s_e = 0
    for i in range(len(centroids)):
        t_s_e += np.sum(np.power((ans[i+1]-centrds[i,:]),2)) 
        
    return(t_s_e/rows_count)

"""---------------------------------------------------------------------------"""

f = input(f"\nPlease Enter the Data file name:\n")
fc = input(f"\nPlease Enter the Centroid file name:\n")

"""---------------------------------------------------------------------------"""

fch = open(fc,'r')
x = fch.readline().strip()
m_centroid = int(x)
c = [(fch.readline().strip('\n').split('\t')) for i in range(2)]
centroids = np.array(c, dtype=float)
fch.close()

"""---------------------------------------------------------------------------"""

fh = open(f,'r')

# Reading first line of the file
x = fh.readline().strip().split()
m = int(x[0])
n = int(x[1])
d = [(fh.readline().strip('\n').split('\t')) for i in range(m)]
fh.close()
data = np.array(d,dtype = float)


"""---------------------------------------------------------------------------"""   

# Printing two initial clusters
print(f"\nInitial Centroids are:\n{centroids[0]}\n{centroids[1]}\nhere each row represents datapoints in (x,y) format")

color = ['green','blue']
labels = ['Cluster1', 'Cluster2']

# Initial plot to represent unclustered data points.
plt.scatter(data[:,0],data[:,1],c='black',label='Unclustered',marker='.')
plt.scatter(centroids[:,0],centroids[:,1],c=color,marker='^')
plt.xlabel('x1 Axis')
plt.ylabel('x2 Axis')
plt.title('Initial Data Points')
plt.show()

"""---------------------------------------------------------------------------""" 

k = 2
count = 0
last_centroids = np.copy(centroids)

answer = {}
all_sq_errors = []

# Loop stops when same centroid value repeat 4 times.
while(count < 4):
    dist=np.empty(shape=(m,0))
    for i in range(k):
        diff = (data-centroids[i,:])
        diff_2 = np.power(diff,2)
        temp_dist = np.sqrt(np.sum(diff_2, axis = 1))
        dist = np.c_[dist,temp_dist]
        
    # Assigning cluster based on minimum distance from the centeroids
    cluster_assigned = np.argmin(dist,axis=1)+1

    # Dictionary to separate data according to the clusters
    data_in_clusters = {} 
    
    # Initializing the dictionary of cluster data
    for i in range(k):
        data_in_clusters[i+1] = np.empty(shape=(0,2))
    
    # Putting data points in their respective clusters
    for i in range(m):
        data_in_clusters[cluster_assigned[i]] = np.vstack((data_in_clusters[cluster_assigned[i]],data[i]))
    
    #Finding new centroids according to the clusters formed
    for i in range(k):
        centroids[i] = np.mean(data_in_clusters[i+1],axis = 0)
    # When values don't change
    if(np.array_equal(centroids,last_centroids) == True):
        count += 1
    last_centroids = np.copy(centroids)
    answer = data_in_clusters

        
"""---------------------------------------------------------------------------"""  

# Final Plotting
for i in range(k):
    plt.scatter(answer[i+1][:,0],answer[i+1][:,1],c = color[i],marker='.')
    
plt.scatter(centroids[:,0],centroids[:,1],c=color,marker='^')
plt.xlabel('x1 Axis')
plt.ylabel('x2 Axis')
plt.title('Clustered Data Points')
plt.show()

"""---------------------------------------------------------------------------""" 

# Results
print(f"\nFinal Centroids are:\n {centroids}\nhere each row represents datapoints in (x,y) format")
squared_error = find_error(centroids, answer,m)
print(f"\nError = {squared_error}")

"""---------------------------------------------------------------------------""" 

