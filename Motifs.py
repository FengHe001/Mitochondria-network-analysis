# -*- coding: utf-8 -*-


import numpy as np

from numba import jit



"""  ------------------------------------------------------------
//  Pre-identifies the nodes that are not connected.
//  Used to improve calculation performance on very sparse networks.
"""

def GetConnectedness( AM, numNodes ):
    
    IsConnected = np.zeros( (numNodes, 1), dtype = np.bool )
    
    for n1 in range(0, numNodes):
        for n2 in range(0, numNodes):
            if AM[n1, n2] > 0 or AM[n2, n1] > 0:
                IsConnected[n1] = True
                break
            
    return IsConnected



@jit(cache=True, nopython=True, nogil=True) 
def setBit(int_type, offset):
    
    mask = 1 << offset
    return(int_type | mask)




def _DecryptMotifs( MotifsArray ):
    
    Motifs = [0] * 13
    
    
    # Motif 1
    
    Offset = setBit(0, 1) +  setBit(0, 2)
    Motifs[ 0 ] = Motifs[ 0 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 3) +  setBit(0, 5)
    Motifs[ 0 ] = Motifs[ 0 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 6) +  setBit(0, 7)
    Motifs[ 0 ] = Motifs[ 0 ] + MotifsArray[ Offset ]


    # Motif 2
    
    Offset = setBit(0, 2) +  setBit(0, 3)
    Motifs[ 1 ] = Motifs[ 1 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 6)
    Motifs[ 1 ] = Motifs[ 1 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 5)
    Motifs[ 1 ] = Motifs[ 1 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 3) +  setBit(0, 7)
    Motifs[ 1 ] = Motifs[ 1 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 7)
    Motifs[ 1 ] = Motifs[ 1 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 5) +  setBit(0, 6)
    Motifs[ 1 ] = Motifs[ 1 ] + MotifsArray[ Offset ]


    # Motif 3
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 3)
    Motifs[ 2 ] = Motifs[ 2 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 6)
    Motifs[ 2 ] = Motifs[ 2 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 3) +  setBit(0, 5)
    Motifs[ 2 ] = Motifs[ 2 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 3) +  setBit(0, 5) +  setBit(0, 7)
    Motifs[ 2 ] = Motifs[ 2 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 2 ] = Motifs[ 2 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 2 ] = Motifs[ 2 ] + MotifsArray[ Offset ]


    # Motif 4
    
    Offset = setBit(0, 2) +  setBit(0, 5)
    Motifs[ 3 ] = Motifs[ 3 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 7)
    Motifs[ 3 ] = Motifs[ 3 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 3) +  setBit(0, 6)
    Motifs[ 3 ] = Motifs[ 3 ] + MotifsArray[ Offset ]


    # Motif 5
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 5)
    Motifs[ 4 ] = Motifs[ 4 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 7)
    Motifs[ 4 ] = Motifs[ 4 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 3) +  setBit(0, 5)
    Motifs[ 4 ] = Motifs[ 4 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 3) +  setBit(0, 5) +  setBit(0, 6)
    Motifs[ 4 ] = Motifs[ 4 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 4 ] = Motifs[ 4 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 3) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 4 ] = Motifs[ 4 ] + MotifsArray[ Offset ]


    # Motif 6
    
    Offset = setBit(0, 3) +  setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 5 ] = Motifs[ 5 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 5 ] = Motifs[ 5 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 3) +  setBit(0, 5)
    Motifs[ 5 ] = Motifs[ 5 ] + MotifsArray[ Offset ]


    # Motif 7
    
    Offset = setBit(0, 1) +  setBit(0, 3) +  setBit(0, 6)
    Motifs[ 6 ] = Motifs[ 6 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 3) +  setBit(0, 7)
    Motifs[ 6 ] = Motifs[ 6 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 3) +  setBit(0, 6)
    Motifs[ 6 ] = Motifs[ 6 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 5) +  setBit(0, 6)
    Motifs[ 6 ] = Motifs[ 6 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 5) +  setBit(0, 7)
    Motifs[ 6 ] = Motifs[ 6 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 5) +  setBit(0, 7)
    Motifs[ 6 ] = Motifs[ 6 ] + MotifsArray[ Offset ]


    # Motif 8
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 3) +  setBit(0, 6)
    Motifs[ 7 ] = Motifs[ 7 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 3) +  setBit(0, 5) +  setBit(0, 7)
    Motifs[ 7 ] = Motifs[ 7 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 7 ] = Motifs[ 7 ] + MotifsArray[ Offset ]


    # Motif 9
    
    Offset = setBit(0, 1) +  setBit(0, 5) +  setBit(0, 6)
    Motifs[ 8 ] = Motifs[ 8 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 3) +  setBit(0, 7)
    Motifs[ 8 ] = Motifs[ 8 ] + MotifsArray[ Offset ]


    # Motif 10
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 5) +  setBit(0, 6)
    Motifs[ 9 ] = Motifs[ 9 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 3) +  setBit(0, 5) +  setBit(0, 6)
    Motifs[ 9 ] = Motifs[ 9 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 9 ] = Motifs[ 9 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 3) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 9 ] = Motifs[ 9 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 3) +  setBit(0, 5) +  setBit(0, 7)
    Motifs[ 9 ] = Motifs[ 9 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 3) +  setBit(0, 7)
    Motifs[ 9 ] = Motifs[ 9 ] + MotifsArray[ Offset ]


    # Motif 11
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 5) +  setBit(0, 7)
    Motifs[ 10 ] = Motifs[ 10 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 3) +  setBit(0, 5) +  setBit(0, 6)
    Motifs[ 10 ] = Motifs[ 10 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 3) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 10 ] = Motifs[ 10 ] + MotifsArray[ Offset ]


    # Motif 12
    
    Offset = setBit(0, 1) +  setBit(0, 2)
    Offset += setBit(0, 3) +  setBit(0, 5) +  setBit(0, 6)
    Motifs[ 11 ] = Motifs[ 11 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 2)
    Offset += setBit(0, 3) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 11 ] = Motifs[ 11 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 3)
    Offset += setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 11 ] = Motifs[ 11 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 2)
    Offset += setBit(0, 3) +  setBit(0, 5) +  setBit(0, 7)
    Motifs[ 11 ] = Motifs[ 11 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 2)
    Offset += setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 11 ] = Motifs[ 11 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 3)
    Offset += setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 11 ] = Motifs[ 11 ] + MotifsArray[ Offset ]


    # Motif 13
    
    Offset = setBit(0, 1) +  setBit(0, 2)
    Offset += setBit(0, 3) +  setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 12 ] = Motifs[ 12 ] + MotifsArray[ Offset ]

    return Motifs



def _DecryptMotifsB( MotifsArray ):
    
    Motifs = [0] * 2
    
    
    # Motif 8
    
    Offset = setBit(0, 1) +  setBit(0, 2) +  setBit(0, 3) +  setBit(0, 6)
    Motifs[ 0 ] = Motifs[ 0 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 1) +  setBit(0, 3) +  setBit(0, 5) +  setBit(0, 7)
    Motifs[ 0 ] = Motifs[ 0 ] + MotifsArray[ Offset ]
    
    Offset = setBit(0, 2) +  setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 0 ] = Motifs[ 0 ] + MotifsArray[ Offset ]




    # Motif 13
    
    Offset = setBit(0, 1) +  setBit(0, 2)
    Offset += setBit(0, 3) +  setBit(0, 5) +  setBit(0, 6) +  setBit(0, 7)
    Motifs[ 1 ] = Motifs[ 1 ] + MotifsArray[ Offset ]

    return Motifs





def GetMotifs( AM ):
    
    numNodes = np.size(AM, 0)
    
    MotifsArray = np.zeros( (512, 1) )

    IsConnected = GetConnectedness( AM, numNodes )
    
    for node1 in range(0, numNodes):
        
        if IsConnected[node1] == False:
            continue
        
        for node2 in range(node1 + 1, numNodes):
            if IsConnected[node2] == False:
                continue
            
            XReg = 0
            
            if AM[node1, node2] > 0:
                XReg = setBit(XReg, 1)
            if AM[node2, node1] > 0:
                XReg = setBit(XReg, 3)

            for node3 in range(node2 + 1, numNodes):
                
                if IsConnected[node3] == False:
                    continue
                
                Reg = XReg
                if AM[node1, node3] > 0:
                    Reg = setBit(Reg, 2)
                if AM[node2, node3] > 0:
                    Reg = setBit(Reg, 5)
                if AM[node3, node1] > 0:
                    Reg = setBit(Reg, 6)
                if AM[node3, node2] > 0:
                    Reg = setBit(Reg, 7)
                
                MotifsArray[ Reg ] = MotifsArray[ Reg ] + 1

    Motifs = _DecryptMotifs( MotifsArray )
    return Motifs




def GetMotifsB( AM ):
    
    numNodes = np.size(AM, 0)
    
    MotifsArray = np.zeros( (512, 1) )

    IsConnected = GetConnectedness( AM, numNodes )
    
    for node1 in range(0, numNodes):
        
        if IsConnected[node1] == False:
            continue
        
        for node2 in range(node1 + 1, numNodes):
            if IsConnected[node2] == False:
                continue
            
            XReg = 0
            
            if AM[node1, node2] > 0:
                XReg = setBit(XReg, 1)
            if AM[node2, node1] > 0:
                XReg = setBit(XReg, 3)

            for node3 in range(node2 + 1, numNodes):
                
                if IsConnected[node3] == False:
                    continue
                
                Reg = XReg
                if AM[node1, node3] > 0:
                    Reg = setBit(Reg, 2)
                if AM[node2, node3] > 0:
                    Reg = setBit(Reg, 5)
                if AM[node3, node1] > 0:
                    Reg = setBit(Reg, 6)
                if AM[node3, node2] > 0:
                    Reg = setBit(Reg, 7)
                
                MotifsArray[ Reg ] = MotifsArray[ Reg ] + 1

    Motifs = _DecryptMotifsB( MotifsArray )
    return Motifs






    
@jit(cache=True, nopython=True, nogil=True) 
def __Motif_1( AM ):
    
    if np.sum( AM ) != 4:
        return 0
    
    if  AM[0, 1] == 1 and AM[1, 0] == 1 and \
        AM[0, 2] == 1 and AM[2, 0] == 1:
            
        return 1
        
    return 0
    
    
@jit(cache=True, nopython=True, nogil=True) 
def __Motif_2( AM ):
    
    if np.sum( AM ) != 6:
        return 0
    
    if  AM[0, 1] == 1 and AM[1, 0] == 1 and \
        AM[1, 2] == 1 and AM[2, 1] == 1 and \
        AM[0, 2] == 1 and AM[2, 0] == 1:
            
        return 1
        
    return 0


    
@jit(cache=True, nopython=True, nogil=True) 
def GetMotifs3B_Fast( AM ):
    
    resMotifs = np.zeros( (2, 1) )
    nn = AM.shape[0]
    
    for n1 in range(nn):
        for n2 in range(nn):
            for n3 in range(nn):
                    
                if n1 == n2 or n1 == n3 or n2 == n3:
                    continue
                
                subAM = AM[ np.array([n1, n2, n3]), :]
                subAM = subAM[ :, np.array([n1, n2, n3]) ]
                resMotifs[0] += __Motif_1( subAM )
                resMotifs[1] += __Motif_2( subAM )

    return resMotifs



