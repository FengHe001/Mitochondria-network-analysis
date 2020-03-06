# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:18:20 2017

@author: MZanin
"""

import numpy as np
from itertools import permutations

from numba import jit



def __PermuteNodes( AM ):
    
    allAM = []
    
    a = [0, 1, 2, 3]

    perms = set()
    for perm in permutations(a):
        perms.add(perm)
    
    for k in perms:
        
        newAM = np.zeros( (4, 4) )
        for n1 in range(4):
            for n2 in range(4):
                newAM[ k[n1], k[n2] ] = AM[ n1, n2 ]
        allAM.append( newAM )

    return allAM
    
    
    
@jit(cache=True, nopython=True, nogil=True) 
def __Motif_1( AM ):
    
    if np.sum( AM ) != 6:
        return 0
    
    if  AM[0, 1] == 1 and AM[1, 0] == 1 and \
        AM[0, 2] == 1 and AM[2, 0] == 1 and \
        AM[0, 3] == 1 and AM[3, 0] == 1:
            
        return 1
        
    return 0
    
    
@jit(cache=True, nopython=True, nogil=True) 
def __Motif_2( AM ):
    
    if np.sum( AM ) != 6:
        return 0
    
    if  AM[0, 1] == 1 and AM[1, 0] == 1 and \
        AM[1, 2] == 1 and AM[2, 1] == 1 and \
        AM[2, 3] == 1 and AM[3, 2] == 1:
            
        return 1
        
    return 0
    
    
@jit(cache=True, nopython=True, nogil=True) 
def __Motif_3( AM ):
    
    if np.sum( AM ) != 8:
        return 0
    
    if  AM[0, 1] == 1 and AM[1, 0] == 1 and \
        AM[1, 2] == 1 and AM[2, 1] == 1 and \
        AM[2, 3] == 1 and AM[3, 2] == 1 and \
        AM[0, 2] == 1 and AM[2, 0] == 1:
            
        return 1
        
    return 0
    
    
@jit(cache=True, nopython=True, nogil=True) 
def __Motif_4( AM ):
    
    if np.sum( AM ) != 8:
        return 0
    
    if  AM[0, 1] == 1 and AM[1, 0] == 1 and \
        AM[1, 2] == 1 and AM[2, 1] == 1 and \
        AM[2, 3] == 1 and AM[3, 2] == 1 and \
        AM[3, 0] == 1 and AM[0, 3] == 1:
            
        return 1
        
    return 0
    
    
@jit(cache=True, nopython=True, nogil=True) 
def __Motif_5( AM ):
    
    if np.sum( AM ) != 10:
        return 0
    
    if  AM[0, 1] == 1 and AM[1, 0] == 1 and \
        AM[1, 2] == 1 and AM[2, 1] == 1 and \
        AM[2, 3] == 1 and AM[3, 2] == 1 and \
        AM[3, 0] == 1 and AM[0, 3] == 1 and \
        AM[0, 2] == 1 and AM[2, 0] == 1:
            
        return 1
        
    return 0
    
    
@jit(cache=True, nopython=True, nogil=True) 
def __Motif_6( AM ):
    
    if np.sum( AM ) != 12:
        return 0
    
    if  AM[0, 1] == 1 and AM[1, 0] == 1 and \
        AM[1, 2] == 1 and AM[2, 1] == 1 and \
        AM[2, 3] == 1 and AM[3, 2] == 1 and \
        AM[3, 0] == 1 and AM[0, 3] == 1 and \
        AM[1, 3] == 1 and AM[3, 1] == 1 and \
        AM[0, 2] == 1 and AM[2, 0] == 1:
            
        return 1
        
    return 0
    
    
    
@jit(cache=True, nopython=True, nogil=True) 
def GetMotifs4B_Fast( AM ):
    
    resMotifs = np.zeros( (6, 1) )
    nn = AM.shape[0]
    
    for n1 in range(nn):
        for n2 in range(nn):
            for n3 in range(nn):
                for n4 in range(nn):
                    
                    if n1 == n2 or n1 == n3 or n1 == n4 or n2 == n3 or n2 == n4 or n3 == n4:
                        continue
                    
                    subAM = AM[ np.array([n1, n2, n3, n4]), :]
                    subAM = subAM[ :, np.array([n1, n2, n3, n4]) ]
                    resMotifs[0] += __Motif_1( subAM )
                    resMotifs[1] += __Motif_2( subAM )
                    resMotifs[2] += __Motif_3( subAM )
                    resMotifs[3] += __Motif_4( subAM )
                    resMotifs[4] += __Motif_5( subAM )
                    resMotifs[5] += __Motif_6( subAM )

    return resMotifs



def GetMotifs4B( AM ):
    
    resMotifs = np.zeros( (6, 1) )
    nn = AM.shape[0]
    
    for n1 in range(nn):
        for n2 in range(nn):
            for n3 in range(nn):
                for n4 in range(nn):
                    
                    if n1 == n2 or n1 == n3 or n1 == n4 or n2 == n3 or n2 == n4 or n3 == n4:
                        continue
                    
                    subAM = AM[ np.ix_([n1, n2, n3, n4], [n1, n2, n3, n4]) ]
                    resMotifs[0] += __Motif_1( subAM )
                    resMotifs[1] += __Motif_2( subAM )
                    resMotifs[2] += __Motif_3( subAM )
                    resMotifs[3] += __Motif_4( subAM )
                    resMotifs[4] += __Motif_5( subAM )
                    resMotifs[5] += __Motif_6( subAM )

    return resMotifs

