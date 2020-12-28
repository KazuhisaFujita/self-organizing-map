#---------------------------------------
#Since : Jun/17/2012
#Update: 2020/12/25
# -*- coding: utf-8 -*-
#---------------------------------------
from PIL import Image
import numpy as np
import math as mt
import pylab as pl
import networkx as nx
import sys
from scipy import ndimage
from sklearn import cluster, datasets
import matplotlib.pyplot as plt

class GNG(object):
    def __init__(self, num = 25, end = 100000, lam = 100, ew = 0.1, en = 0.01, amax = 20.0, alpha = 0.5, beta = 0.9, sig_kernel = 0.5):
        # Set Parameters

        # max of units
        self.MAX_NUM = num
        # the number of delete processes
        self.END = end
        # insert
        self.lam = lam
        # Learning coefficient
        self.Ew = ew
        # Learning coefficient of neighbors
        self.En = en
        # threshold to remove a edge
        self.AMAX = amax
        # reduction rate of Error when the insertion of a new neuron.
        self.Alpha = alpha
        # reduction rate of Error
        self.Beta = beta

        # kernel
        self.sig_kernel = sig_kernel

    def initialize_units(self, data):
        #dimension
        self.I_DIM = data[0].size

        self.N = data.shape[0] # the number of data points

        self.units = np.zeros((self.MAX_NUM, self.I_DIM))
        self.sumerror = np.zeros(self.MAX_NUM)
        self.g_units = nx.Graph()

        # reference vectors of dead units are set to infinity.
        self.units += float("inf")

        # initialize the two units
        self.units[0], self.units[1] = data[np.random.permutation(data.shape[0])[[0, 1]]]

        self.g_units.add_node(0)
        self.g_units.add_node(1)
        self.g_units.add_edge(0, 1, weight=0)

    def dists(self, x, units):
        #calculate distance
        return np.linalg.norm(units - x, axis=1)

    def dw(self, x, unit):
        return x - unit

    def kernel(self, x):
        return(np.exp(- np.linalg.norm(np.expand_dims(x, axis = 1) - x,axis=2)**2/2/(self.sig_kernel**2)))

    def affinity(self):
        self.units = self.units[np.isfinite(self.units[:,0])]
        A = nx.adjacency_matrix(self.g_units, weight=1)
        A = np.array(A.todense())
        A = A * self.kernel(self.units)
        return A

    def normalize(self, data):
        self.mindata = data[np.argmin(np.linalg.norm(data, axis=1))]
        self.diff_max_min = np.linalg.norm( data[np.argmax(np.linalg.norm(data, axis=1))] - data[np.argmin(np.linalg.norm(data, axis=1))])
        data = (data - self.mindata) / self.diff_max_min
        return data

    def train(self, data):
        self.initialize_units(data)

        units = self.units
        g_units = self.g_units
        sumerror = self.sumerror

        count = 0
        oE = 0
        for t in range(self.END):

            # Generate a random input.
            num = np.random.randint(self.N)
            x = data[num]

            # Find the nearest and the second nearest neighbors, s_1 s_2.
            existing_units = np.array(g_units.nodes())
            dists = self.dists(x, units[existing_units])
            s_1, s_2 = existing_units[dists.argsort()[[0,1]]]

            # Add the distance between the input and the nearest neighbor s_1.
            sumerror[s_1] += dists[existing_units == s_1]

            # Move the nearest neighbor s_1 towards the input.
            units[s_1] += self.Ew * self.dw(x, units[s_1])

            if g_units.has_edge(s_1, s_2):
                # Set the age of the edge of s_1 and s_2 to 0.
                g_units[s_1][s_2]['weight'] = 0
            else:
                # Connect NN and second NN with each other.
                g_units.add_edge(s_1,s_2,weight = 0)

            for i in list(g_units.neighbors(s_1)):
                # Increase the age of all the edges emanating from the nearest neighbor s_1
                g_units[s_1][i]['weight'] += 1

                # Move the neighbors of s_1 towards the input.
                units[i] += self.En  * self.dw(x, units[i])

                if g_units[s_1][i]['weight'] > self.AMAX:
                    g_units.remove_edge(s_1,i)
                if g_units.degree(i) == 0:
                    g_units.remove_node(i)
                    units[i] += float("inf")
                    sumerror[i] = 0
                    #end set

            # Every lambda, insert a new neuron.
            count += 1
            if count == self.lam:
                count = 0
                nodes = list(g_units.nodes())

                # Find the neuron q with the maximum error.
                q = nodes[sumerror[nodes].argmax()]

                # Find the neighbor neuron f with maximum error.
                neighbors = list(g_units.neighbors(q))
                se = sumerror[neighbors]
                f = neighbors[np.argmax(se)]

                for i in range(self.MAX_NUM):
                    if units[i][0] == float("inf"):
                        # Insert a new neuron r.
                        units[i] = 0.5 * (units[q] + units[f])
                        g_units.add_node(i)

                        # Insert new edges between the neuron and q and f.
                        g_units.add_edge(i, q, weight=0)
                        g_units.add_edge(i, f, weight=0)

                        # Remove the edges between q and f.
                        g_units.remove_edge(q, f)

                        # Decrease the error of q and f.
                        sumerror[q] *= self.Alpha
                        sumerror[f] *= self.Alpha

                        # Set the error of the new neuron to that of q
                        sumerror[i] = sumerror[q]
                        break

            # Decrease all errors.
            sumerror *= self.Beta

        self.units = self.units[np.isfinite(units[:,0])]



if __name__ == '__main__':

    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    gng = GNG(num = 100, end = 100000, lam = 100, ew = 0.1, en = 0.005, amax = 50.0, alpha = 0.5,  beta = 0.995, sig_kernel = 1)

    gng.train(noisy_circles[0])
    plt.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1])


    nx.draw_networkx_nodes(gng.g_units,gng.units,node_size=50,node_color=(0.5,1,1))
    nx.draw_networkx_edges(gng.g_units,gng.units,width=2,edge_color='b',alpha=0.5)

    plt.savefig("gng.png")
