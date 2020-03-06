#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:14:20 2018

@author: mzanin
"""

import numpy as np
import networkx as nx
import random



def __getGCCNodeCount(G):
    
    Gccs=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
    return Gccs[0].number_of_nodes()



def BinarySearchMethod( N, L ):
    
    locN,locL=N, L
    G=nx.gnm_random_graph(locN,locL)
    delta=1
    
    
    #   Check if we are lucky...
    if __getGCCNodeCount(G)==locN:
        return G
    
    
    finished=False
    
    while not finished:
    
        while __getGCCNodeCount(G)<locN:
            G=nx.gnm_random_graph(locN+delta,locL)
            delta=delta*2
        
        if __getGCCNodeCount(G)>=locN:        
            delta=int(delta/2)
            left,right=int(locN+delta/2),int(locN+delta)        
            while left!=right:
                middle=left+(right-left)/2
                G=nx.gnm_random_graph(int(middle),locL)
                NC=__getGCCNodeCount(G)
                if NC>locN:
                    right=middle
                elif NC<locN:
                    left=middle
                else:
                    left=right
        
        Gccs=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
        G0=Gccs[0]
        
        nodes=list(G0.nodes())            
        while G.number_of_edges()<locL:
            a=random.choice(nodes)    
            b=random.choice(nodes)    
                
            G0.add_edge(a,b)
            
        curN,curL,curCon=G0.number_of_nodes(),G0.number_of_edges(),nx.is_connected(G0)
        
        if curN==locN and curL==locL and curCon==True:
            finished=True
            
    return G0





