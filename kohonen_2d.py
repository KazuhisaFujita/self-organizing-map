#---------------------------------------
#Since : 2017/06/09
#Update: 2020/12/25
# -*- coding: utf-8 -*-
#---------------------------------------
from PIL import Image
import numpy as np
import math as mt
import pylab as plt
import networkx as nx
import sys
from scipy import ndimage
from sklearn import cluster, datasets

class Kohonen:
    def __init__(self, num = 4, dim = 2, end = 2000, rate = 1.0, sigma = 10, sig_kernel = 0.5):
        self.N = num**2
        self.units = np.random.random((self.N, dim))
        self.END = end
        self.rate = rate
        self.sigma = sigma
        self.sig_kernel = sig_kernel

        self.g_units = nx.grid_2d_graph(num, num)
        labels = dict( ((i,j), i + (num - 1 - j) * num ) for i, j in self.g_units.nodes() )
        nx.relabel_nodes(self.g_units, labels, False)
        self.pos = np.array([[i, j] for i in range(1, num + 1) for j in range(1, num + 1)])/float(num)

    def alpha(self, A, ac, end):
        return (A * (1.0 - float(ac)/float(end)) )

    def neig_func(self, dist, t):
        return np.exp(- (dist**2) / 2 / (self.alpha(self.sigma, t, self.END)**2))

    def kernel(self, x):
        return(np.exp(- np.linalg.norm(np.expand_dims(x, axis = 1) - x,axis=2)**2/2/(self.sig_kernel**2)))

    def affinity(self):
        A = nx.adjacency_matrix(self.g_units)
        A = np.array(A.todense())
        A = np.where(A > 0, 1, 0)
        A = A * self.kernel(self.units)
        return A

    def normalize(self, data):
        self.mindata = data[np.argmin(np.linalg.norm(data, axis=1))]
        self.diff_max_min = np.linalg.norm( data[np.argmax(np.linalg.norm(data, axis=1))] - data[np.argmin(np.linalg.norm(data, axis=1))])
        data = (data - self.mindata) / self.diff_max_min
        return data

    def train(self, data):
        units = self.units
        num_sample = data.shape[0]
        for t in range(self.END):
            x = data[np.random.choice(num_sample)]
            dists_x = np.linalg.norm(units - x, axis = 1)
            min_unit_num = dists_x.argmin()
            dists_p = np.linalg.norm(self.pos - self.pos[min_unit_num], axis=1)
            units +=  self.alpha(self.rate, t, self.END) * np.multiply((x - units), self.neig_func(dists_p, t).reshape(self.N,1))

if __name__ == '__main__':

    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    kohonen = Kohonen(num = 10, dim = 2, end = 100000, rate = 0.1, sigma = 0.5)

    kohonen.train(noisy_circles[0])

    plt.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1])

    nx.draw_networkx_nodes(kohonen.g_units,kohonen.units,node_size=5,node_color=(0.5,1,1))
    nx.draw_networkx_edges(kohonen.g_units,kohonen.units,width=5,edge_color='b',alpha=0.5)

    plt.savefig("kohonen.png")
