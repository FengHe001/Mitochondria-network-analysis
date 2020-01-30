# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:32:05 2016

@author: MZanin
"""

import numpy as np
import scipy as sp



def NetInfCont( AM ):

    numNodes = np.size(AM, 0)
    Info = 0
    
    DistanceMatrix = np.zeros( ( numNodes, numNodes ) )
    for n1 in range(0, numNodes):
        for n2 in range(0, numNodes):
            
            if n1 == n2:
                DistanceMatrix[n1, n2] = numNodes
                continue
            
            DistanceMatrix[n1, n2] = __HD( AM[n1, :], AM[n2, :], AM[:, n1], AM[:, n2] )    
    
    
    for k in range(0, numNodes - 2):
        (tInfo, AM, DistanceMatrix) = __OneIteration(AM, DistanceMatrix)
        Info = Info + tInfo
                
    return Info





def CompareNetworks( AM1, AM2, num_runs = 100 ):
    
    DiffAM = np.abs( AM1 - AM2 )
    Info = NetInfCont( DiffAM )
    
    numNodes = np.size(AM1, 0)
    linkDensity = np.float( np.sum( DiffAM ) ) / ( numNodes * numNodes )
    
    temp = []
    for l in range(num_runs):
        AM_t = np.random.binomial( 1, linkDensity, (numNodes, numNodes) )
        tInfo = NetInfCont( AM_t )
        temp.append( tInfo )
        
    zScore = ( Info - np.mean(temp) ) / np.std( temp )
    pValue = 1 - sp.special.ndtr( - zScore )
    
    return ( pValue, Info )




def Normalize( AM, num_runs = 1000 ):
    
    numNodes = np.size(AM, 0)
    linkDensity = np.float( np.sum( AM ) ) / ( numNodes * numNodes )
    
    temp = []
    for l in range(num_runs):
        AM_t = np.random.binomial( 1, linkDensity, (numNodes, numNodes) )
        Info = NetInfCont( AM_t )
        temp.append( Info )
        
    return np.mean( temp )







def __OneIteration(AM, DistanceMatrix):

    numNodes = np.size(AM, 0)
    
    minDist = DistanceMatrix.min()
    maxDist = DistanceMatrix.max()
    
    bestNodes = []
    bestHD = 0
    if minDist < ( 2 * numNodes - maxDist ):
        bestNodes = np.argmin( DistanceMatrix )
    else:
        bestNodes = np.argmax( DistanceMatrix )
    bestNodes = np.unravel_index( bestNodes, DistanceMatrix.shape )
    
    bestNode = bestNodes[0]
    bestNode2 = bestNodes[1]
    bestHD = DistanceMatrix[ bestNode, bestNode2 ]
    
    if bestNode == bestNode2:
        bestNode = 1
        bestNode2 = 2
    
    p = bestHD / ( numNodes * 2 )
    Info = 0.0
    if p > 0 and p < 1:
        Info = - p * np.log2( p ) - (1.0 - p) * np.log2( 1.0 - p )
        Info = Info * numNodes * 2.0
  
    
    newDM1 = np.copy( DistanceMatrix )
    
    for n1 in range(0, numNodes):
        for n2 in range(0, numNodes):
            
            if n1 == n2:
                continue
            
            newDM1[n1, n2] = newDM1[n1, n2] + np.double( AM[n1, bestNode] == AM[n2, bestNode] ) - 1.0
            newDM1[n1, n2] = newDM1[n1, n2] + np.double( AM[bestNode, n1] == AM[bestNode, n2] ) - 1.0
            
    newDM1 = np.delete( newDM1, bestNode, 0 )
    newDM1 = np.delete( newDM1, bestNode, 1 )
  
    
    newDM2 = np.copy( DistanceMatrix )
    
    for n1 in range(0, numNodes):
        for n2 in range(0, numNodes):
            
            if n1 == n2:
                continue
            
            newDM2[n1, n2] = newDM2[n1, n2] + np.double( AM[n1, bestNode2] == AM[n2, bestNode2] ) - 1.0
            newDM2[n1, n2] = newDM2[n1, n2] + np.double( AM[bestNode2, n1] == AM[bestNode2, n2] ) - 1.0
            
    newDM2 = np.delete( newDM2, bestNode2, 0 )
    newDM2 = np.delete( newDM2, bestNode2, 1 )
   
    
    maxV1 = np.max( newDM1 )
    maxV2 = np.max( newDM2 )
    minV1 = np.min( newDM1 )
    minV2 = np.min( newDM2 )
    
    if maxV1 < ( 2 * ( numNodes - 1 ) - minV1 ):
        maxV1 = 2 * ( numNodes - 1 ) - minV1
    if maxV2 < ( 2 * ( numNodes - 1 ) - minV2 ):
        maxV2 = 2 * ( numNodes - 1 ) - minV2
    
    if maxV1 > maxV2:
        newAM = np.copy( AM )
        newAM = np.delete( newAM, bestNode, 0 )
        newAM = np.delete( newAM, bestNode, 1 )
        newDM = newDM1
    else:
        newAM = np.copy( AM )
        newAM = np.delete( newAM, bestNode2, 0 )
        newAM = np.delete( newAM, bestNode2, 1 )
        newDM = newDM2
    
    
    return (Info, newAM, newDM)
            
  




def __HD( vector1, vector2, vector3, vector4 ):

    sizeVector = np.size( vector1 )
    answ = np.sum( vector1 == vector2 ) + np.sum( vector3 == vector4 )
    
    answ = 2 * sizeVector - answ
    
    return answ



def __approxF( p ):

    if p == 0.0 or p == 1.0:
        return 0.0

    approxInfo = - p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    
    return approxInfo




def __Normalize( p, numNodes, directed ):

    if directed == 1:
        KMax = 0.1625 - 1.822 * np.power( numNodes + 10.08, -1.098 )
    else:
        KMax = 0.1622 - 0.6298 * np.power( numNodes + 5.905, -1.132 )
    
    alpha = KMax * 4.0;
    effP = alpha * p - alpha * p * p;
    
    Info = 0.0;
    
    for k in range( numNodes, 2, -1 ):
        temp = (2 * k) * __approxF( effP )
        Info = Info + temp
    
    return Info



