# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:42:23 2017

@author: MZanin
"""

import numpy as np
import networkx as nx
import scipy.io
import glob

import LargeRandomNetworks as lrn


def BasicFunction(N, L):
    G = nx.gnm_random_graph(N, L)
    while not nx.is_connected( G ):
        G = nx.gnm_random_graph(N,L)

    return G



numRandomNets = 100


for group in [1]:

    allFiles = glob.glob( './Networks/G%d_*.npy' % group )
    numFiles = len( allFiles )

    for index in range( numFiles ):

        print( 'Processing group %d, network %d' % ( group, index ) )
        fileName = './Networks/G%d_%05d.npy' % ( group, index )
        AMO = np.load( fileName )

        N = np.size( AMO, 0 )
        L = int( np.sum( AMO[:] ) / 2.0 )
        newAM = np.zeros( (numRandomNets, N, N), dtype = int )

        for k in range( numRandomNets ):

            tRes = None
            while tRes == None:
                tRes = lrn.BinarySearchMethod(N, L)

            newAM[k, :, :] = nx.to_numpy_matrix( tRes )

        fileName = './RndNetworks/G%d_%05d.npy' % ( group, index )
        np.save( fileName, newAM )
