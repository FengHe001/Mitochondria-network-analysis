# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:31:40 2017

@author: MZanin
"""


import numpy as np
import networkx as nx
import scipy.io
import glob

import Motifs as m3
import Motifs4B as m4



for group in [1]:

    allProp = []

    allFiles = glob.glob( './Networks/G%d_*.npy' % group )
    numFiles = len( allFiles )

    for index in range( numFiles ):

        print( 'Processing group %d, network %d' % ( group, index ) )
        fileName = './Networks/G%d_%05d.npy' % ( group, index )
        AMO = np.load( fileName )

        if np.size( AMO, 0 ) > 50 or np.size( AMO, 0 ) < 3:
            continue

        p1 = m3.GetMotifs( AMO )
        p1 = np.array( p1 )[ [7, 12] ]
        p1 = np.ravel( p1 )
        p2 = m4.GetMotifs4B_Fast( AMO )
        p2 = np.ravel( p2 )
        prop = np.hstack( (p1, p2) )

        rndProp = np.zeros((np.size(prop, 0), 100))
        fileName = './RndNetworks/G%d_%05d.npy' % ( group, index )
        AMR = np.load( fileName )

        for k in range(100):
            AM = AMR[ k, :, : ]
            p1 = m3.GetMotifs(AM)
            p1 = np.array(p1)[[7, 12]]
            p1 = np.ravel(p1)
            p2 = m4.GetMotifs4B_Fast(AM)
            p2 = np.ravel(p2)
            rprop = np.hstack((p1, p2))
            rndProp[:, k] = rprop

        tmpStd = np.std(rndProp, 1)
        normProp = (prop - np.mean(rndProp, 1)) / tmpStd

        allProp.append( normProp[:] )


    np.save( 'allMotifs_G%d' % group, allProp )

