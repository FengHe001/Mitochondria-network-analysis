# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:49:14 2016

@author: MZanin
"""


import networkx as nx
import numpy as np
import sys

        
def PrepareNetwork( AM ):
    
    G = nx.from_numpy_matrix( AM, create_using = nx.DiGraph() )

    return G



def PrepareNetworkW( WM ):
    
    numNodes = len( WM )
    G = nx.DiGraph()
    G.add_nodes_from( range( numNodes ) )    

    for ch1 in range( numNodes ):
        for ch2 in range( numNodes ):

            if WM[ch1, ch2] > 0.0:
                G.add_edge( ch1, ch2, weight = WM[ch1, ch2] )

    return G



def PrepareNetworkUnd( AM ):
    
    G = nx.from_numpy_matrix( AM, create_using = nx.Graph() )

    return G




def PrepareNetworkUndW( WM ):
    
    numNodes = len( WM )
    G = nx.Graph()
    G.add_nodes_from( range( numNodes ) )    

    for ch1 in range( numNodes ):
        for ch2 in range( numNodes ):

            if WM[ch1, ch2] > 0.0:
                G.add_edge( ch1, ch2, weight = WM[ch1, ch2] )

    return G




def GetNestedness_AM( AM ):
    
    # Jonhson, S., Domínguez-García, V., & Muñoz, M. A. (2013).
    # Factors determining nestedness in complex networks.
    # PloS one, 8(9), e74025.

    numNodes = np.size( AM, 0 )
    
    degrees = np.sum( AM, 0 )
    meanD = np.mean( degrees )
    meanD2 = np.mean( np.square( degrees ) )
    
    AM2 = np.dot( AM, AM )
    
    nestedness = 0.0
    
    for n1 in range( numNodes ):
        for n2 in range( numNodes ):
            t = AM2[n1, n2]
            t2 = np.ravel( degrees )[n1] * np.ravel( degrees )[n2]
            if t2 != 0.0:
                t /= t2
                nestedness += t
    
    nestedness *= ( meanD * meanD ) / ( meanD2 * numNodes )
    
    return nestedness


    
def GetEfficiency( G ):

    Eff = 0.0
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 == n2:
                continue;
                
            try:
                dist = nx.shortest_path_length( G, n1, n2 )
                Eff += 1.0 / dist
            except:
                pass
    
    numNodes = len( G.nodes() )
    Eff /= (float) (numNodes) * ( numNodes - 1.0 )
    return Eff

    
    
def GetEfficiency_AM( AM ):
    G = PrepareNetwork( AM )
    return GetEfficiency( G )

    
    
    
def Get1vs2Degree_AM( AM ):

    degree = np.sum( AM, 0 )
    degree = np.sort( degree )    
    return np.float( degree[-1] ) / degree[-2]



def GetRandicIndex_AM( AM ):
    
    degree = np.ravel( np.sum( AM, 0 ) )
    numNodes = np.size( AM, 0 )
    rInd = 0.0
    
    for n1 in range( numNodes ):
        for n2 in range( numNodes ):
            
            if AM[n1, n2] >= 0:
                t = np.sqrt( degree[n1] * degree[n2].T )
                if t != 0.0: 
                    rInd += 1.0 / t
    
    return rInd




def GetRandomNetworks( AM, exactRnd ):
    
    import SynthRandomNetworks as srn
    
    numNodes = len(AM)
    numLinks = np.sum( AM )    
    p = numLinks / ( numNodes * ( numNodes - 1.0 ) )
    
    if exactRnd:
        newAM = srn.GetExactNetwork( numNodes, numLinks )
    else:
        newAM = srn.GetApproxNetwork( numNodes, p )
                
    return newAM

    
    
def GetModularity( G ):

    import Communities as nc

    Mod = 0.0    
    
#    try:
    H = G.to_undirected()
    part = nc.best_partition( H )
    tempM = nc.modularity( part, H )
    Mod = tempM
#    except:
#        Mod = 0.0

    return Mod
    
    
def GetModularity_AM( AM ):

    G = PrepareNetwork( AM )
    return GetModularity( G )
    



def GetAssortativity_AM( AM ):

    G = PrepareNetwork( AM )
    return nx.degree_pearson_correlation_coefficient( G )
    
    
    
    
def GetSW_AM( AM ):
    
    G = PrepareNetwork( AM )
    
    nsc = nx.number_strongly_connected_components( G )
    
    sw = 0.0
    if nsc == 1:
        aspl = nx.average_shortest_path_length( G )
        cc = nx.transitivity( G )
        sw = cc / aspl
        
    return sw
    
    
    
    
def GetEntropy( AM ):
        
    nn = np.size( AM, 0 )
    degree = np.sum( AM, 0 )
    
    E = 0.0
    for k in range( nn ):
        if np.sum( degree == k ) == 0:
            continue
        
        nDegree = float( np.sum( degree == k ) ) / nn
        E -= nDegree * np.log2( nDegree )
        
    return E





def GetSizeGiantComponent( AM ):
    
    G = PrepareNetwork( AM )
    allS = [len(gc) for gc in sorted(nx.strongly_connected_component_subgraphs(G), key = len, reverse = True)]
    return allS[0]


    
    
def GetAllProperties( AM, enableIC = True ):
    
    G = PrepareNetwork( AM )
    
    nn = np.size( AM, 0 )
    ld = nx.density( G )

    maxd = np.max( np.sum( AM, 0 ) )
    maxd_in = np.max( np.sum( AM, 1 ) )

    eff = GetEfficiency( G )
    mod = GetModularity( G )
    ass = nx.degree_pearson_correlation_coefficient( G )
    cc = nx.transitivity( G )
    nest = GetNestedness_AM( AM )
    
    ic = 0.0
    if enableIC:
        import NetInfCont as nic
        ic = nic.NetInfCont( AM )
    
    nsc = nx.number_strongly_connected_components( G )
    nwc = nx.number_weakly_connected_components( G )
    
    diam = nn + 1
    if nsc == 1:
        diam = nx.diameter( G )
     
    aspl = 0.0
    sw = 0.0
    if nsc == 1:
        aspl = nx.average_shortest_path_length( G )
        sw = cc / aspl

    
    allP = ( nn, ld, maxd, eff, mod, ass, cc, ic, diam, aspl, sw, nsc, nwc )
    allP = np.array( allP, float )
    return allP
    
    

def GetNormProperties( AM, numRand = 100, enableIC = True, exactRnd = False ):
    
    origP = GetAllProperties( AM, enableIC )  
    
    tempP = np.zeros( (15, numRand) )
    for k in range( numRand ):
        newAM = GetRandomNetworks( AM, exactRnd )
        tempP[:, k] = GetAllProperties( newAM, enableIC )
    
    normP = ( origP - np.mean(tempP, 1) ) / np.std(tempP, 1)
    return normP
