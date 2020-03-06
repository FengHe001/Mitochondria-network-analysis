# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 11:33:31 2017

@author: MZanin
"""

import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec



np.random.seed( 5 )

nEstimators = 1000
nIter = 500000


allData = np.load( 'allData.npy' )
allData[ :, 3: ] = normalize( allData[ :, 3: ] )
allProb = np.zeros( ( np.size( allData, 0 ), 1 ) )

superClass = allData[ :, :2 ]
superClasses = np.unique( superClass, axis = 0 )

for k in range( np.size( superClasses, 0 ) ):

    print( 'Processing super-class %d' % k )
    
    myfilter = np.all( ( allData[:, 0] == superClasses[k, 0], allData[:, 1] == superClasses[k, 1] ), axis = 0 )
    myfilter2 = np.any( ( allData[:, 0] != superClasses[k, 0], allData[:, 1] != superClasses[k, 1] ), axis = 0 )

    trainF = allData[ myfilter2, : ]
    trainF = trainF[ :, 2: ]
    testF = allData[ myfilter, : ]
    testF = testF[ :, 2: ]

    trainC = allData[ myfilter2, : ]
    trainC = trainC[ :, 0 ]
    testC = allData[ myfilter, : ]
    testC = testC[ :, 0 ]

    clf = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(20, 3), max_iter = nIter)

    clf.fit( trainF, trainC )
    aX = clf.predict_proba( testF )[:, 0]

    if np.size( aX, 0 ) > 1:
        while np.sum( np.abs( aX[1:] - aX[:-1] ) < 0.0001 ) > np.size( aX, 0 ) * 0.25:
            clf.fit( trainF, trainC )
            aX = clf.predict_proba( testF )[:, 0]

    allProb[ myfilter, 0 ] = aX

auc_global = roc_auc_score( allData[:, 0], allProb )
fpr_global, tpr_global, _ = roc_curve( allData[:, 0], allProb )




allData = np.load( 'allData.npy' )
allData[ :, 3: ] = normalize( allData[ :, 3: ] )
allData = allData[ allData[ :, 2 ] == 0, : ]
allProb = np.zeros( ( np.size( allData, 0 ), 1 ) )

superClass = allData[ :, :2 ]
superClasses = np.unique( superClass, axis = 0 )

for k in range( np.size( superClasses, 0 ) ):

    print( 'Processing super-class %d, Left' % k )
    
    myfilter = np.all( ( allData[:, 0] == superClasses[k, 0], allData[:, 1] == superClasses[k, 1] ), axis = 0 )
    myfilter2 = np.any( ( allData[:, 0] != superClasses[k, 0], allData[:, 1] != superClasses[k, 1] ), axis = 0 )

    trainF = allData[ myfilter2, : ]
    trainF = trainF[ :, 2: ]
    testF = allData[ myfilter, : ]
    testF = testF[ :, 2: ]

    trainC = allData[ myfilter2, : ]
    trainC = trainC[ :, 0 ]
    testC = allData[ myfilter, : ]
    testC = testC[ :, 0 ]

    clf = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(20, 3), max_iter = nIter)

    clf.fit( trainF, trainC )
    aX = clf.predict_proba( testF )[:, 0]

    if np.size( aX, 0 ) > 1:
        while np.sum( np.abs( aX[1:] - aX[:-1] ) < 0.0001 ) > np.size( aX, 0 ) * 0.25:
            clf.fit( trainF, trainC )
            aX = clf.predict_proba( testF )[:, 0]

    allProb[ myfilter, 0 ] = aX

auc_global_L = roc_auc_score( allData[:, 0], allProb )
fpr_global_L, tpr_global_L, _ = roc_curve( allData[:, 0], allProb )




allData = np.load( 'allData.npy' )
allData[ :, 3: ] = normalize( allData[ :, 3: ] )
allData = allData[ allData[ :, 2 ] == 1, : ]
allProb = np.zeros( ( np.size( allData, 0 ), 1 ) )

superClass = allData[ :, :2 ]
superClasses = np.unique( superClass, axis = 0 )

for k in range( np.size( superClasses, 0 ) ):

    print( 'Processing super-class %d, Right' % k )
    
    myfilter = np.all( ( allData[:, 0] == superClasses[k, 0], allData[:, 1] == superClasses[k, 1] ), axis = 0 )
    myfilter2 = np.any( ( allData[:, 0] != superClasses[k, 0], allData[:, 1] != superClasses[k, 1] ), axis = 0 )

    trainF = allData[ myfilter2, : ]
    trainF = trainF[ :, 2: ]
    testF = allData[ myfilter, : ]
    testF = testF[ :, 2: ]

    trainC = allData[ myfilter2, : ]
    trainC = trainC[ :, 0 ]
    testC = allData[ myfilter, : ]
    testC = testC[ :, 0 ]

    clf = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(20, 3), max_iter = nIter)

    clf.fit( trainF, trainC )
    aX = clf.predict_proba( testF )[:, 0]

    if np.size( aX, 0 ) > 1:
        while np.sum( np.abs( aX[1:] - aX[:-1] ) < 0.0001 ) > np.size( aX, 0 ) * 0.25:
            clf.fit( trainF, trainC )
            aX = clf.predict_proba( testF )[:, 0]

    allProb[ myfilter, 0 ] = aX

auc_global_R = roc_auc_score( allData[:, 0], allProb )
fpr_global_R, tpr_global_R, _ = roc_curve( allData[:, 0], allProb )




allData = np.load( 'allData.npy' )
allData = allData[ allData[ :, 3 ] < 10, : ]
allData[ :, 3: ] = normalize( allData[ :, 3: ] )
allProb = np.zeros( ( np.size( allData, 0 ), 1 ) )

superClass = allData[ :, :2 ]
superClasses = np.unique( superClass, axis = 0 )

for k in range( np.size( superClasses, 0 ) ):

    print( 'Processing super-class %d, size < 10' % k )
    
    myfilter = np.all( ( allData[:, 0] == superClasses[k, 0], allData[:, 1] == superClasses[k, 1] ), axis = 0 )
    myfilter2 = np.any( ( allData[:, 0] != superClasses[k, 0], allData[:, 1] != superClasses[k, 1] ), axis = 0 )

    trainF = allData[ myfilter2, : ]
    trainF = trainF[ :, 2: ]
    testF = allData[ myfilter, : ]
    testF = testF[ :, 2: ]

    trainC = allData[ myfilter2, : ]
    trainC = trainC[ :, 0 ]
    testC = allData[ myfilter, : ]
    testC = testC[ :, 0 ]

    clf = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(20, 3), max_iter = nIter)

    clf.fit( trainF, trainC )
    aX = clf.predict_proba( testF )[:, 0]

    if np.size( aX, 0 ) > 1:
        while np.sum( np.abs( aX[1:] - aX[:-1] ) < 0.0001 ) > np.size( aX, 0 ) * 0.25:
            clf.fit( trainF, trainC )
            aX = clf.predict_proba( testF )[:, 0]

    allProb[ myfilter, 0 ] = aX

auc_l10 = roc_auc_score( allData[:, 0], allProb )
fpr_l10, tpr_l10, _ = roc_curve( allData[:, 0], allProb )




allData = np.load( 'allData.npy' )
allData = allData[ np.all( ( allData[ :, 3 ] >= 10, allData[ :, 3 ] < 20 ), axis = 0) , : ]
allData[ :, 3: ] = normalize( allData[ :, 3: ] )
allProb = np.zeros( ( np.size( allData, 0 ), 1 ) )

superClass = allData[ :, :2 ]
superClasses = np.unique( superClass, axis = 0 )

for k in range( np.size( superClasses, 0 ) ):

    print( 'Processing super-class %d, 10 < size < 20' % k )
    
    myfilter = np.all( ( allData[:, 0] == superClasses[k, 0], allData[:, 1] == superClasses[k, 1] ), axis = 0 )
    myfilter2 = np.any( ( allData[:, 0] != superClasses[k, 0], allData[:, 1] != superClasses[k, 1] ), axis = 0 )

    trainF = allData[ myfilter2, : ]
    trainF = trainF[ :, 2: ]
    testF = allData[ myfilter, : ]
    testF = testF[ :, 2: ]

    trainC = allData[ myfilter2, : ]
    trainC = trainC[ :, 0 ]
    testC = allData[ myfilter, : ]
    testC = testC[ :, 0 ]

    clf = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(20, 3), max_iter = nIter)

    clf.fit( trainF, trainC )
    aX = clf.predict_proba( testF )[:, 0]

    if np.size( aX, 0 ) > 1:
        while np.sum( np.abs( aX[1:] - aX[:-1] ) < 0.0001 ) > np.size( aX, 0 ) * 0.25:
            clf.fit( trainF, trainC )
            aX = clf.predict_proba( testF )[:, 0]

    allProb[ myfilter, 0 ] = aX

auc_l20 = roc_auc_score( allData[:, 0], allProb )
fpr_l20, tpr_l20, _ = roc_curve( allData[:, 0], allProb )



allData = np.load( 'allData.npy' )
allData = allData[ allData[ :, 3 ] >= 20, : ]
allData[ :, 3: ] = normalize( allData[ :, 3: ] )
allProb = np.zeros( ( np.size( allData, 0 ), 1 ) )

superClass = allData[ :, :2 ]
superClasses = np.unique( superClass, axis = 0 )

for k in range( np.size( superClasses, 0 ) ):

    print( 'Processing super-class %d, size > 20' % k )
    
    myfilter = np.all( ( allData[:, 0] == superClasses[k, 0], allData[:, 1] == superClasses[k, 1] ), axis = 0 )
    myfilter2 = np.any( ( allData[:, 0] != superClasses[k, 0], allData[:, 1] != superClasses[k, 1] ), axis = 0 )

    trainF = allData[ myfilter2, : ]
    trainF = trainF[ :, 2: ]
    testF = allData[ myfilter, : ]
    testF = testF[ :, 2: ]

    trainC = allData[ myfilter2, : ]
    trainC = trainC[ :, 0 ]
    testC = allData[ myfilter, : ]
    testC = testC[ :, 0 ]

    clf = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(20, 3), max_iter = nIter)

    clf.fit( trainF, trainC )
    aX = clf.predict_proba( testF )[:, 0]

    if np.size( aX, 0 ) > 1:
        while np.sum( np.abs( aX[1:] - aX[:-1] ) < 0.0001 ) > np.size( aX, 0 ) * 0.25:
            clf.fit( trainF, trainC )
            aX = clf.predict_proba( testF )[:, 0]

    allProb[ myfilter, 0 ] = aX

auc_g20 = roc_auc_score( allData[:, 0], allProb )
fpr_g20, tpr_g20, _ = roc_curve( allData[:, 0], allProb )



allData = np.load( 'allData.npy' )[ :, :16 ]
allData[ :, 3: ] = normalize( allData[ :, 3: ] )
allProb = np.zeros( ( np.size( allData, 0 ), 1 ) )

superClass = allData[ :, :2 ]
superClasses = np.unique( superClass, axis = 0 )

for k in range( np.size( superClasses, 0 ) ):

    print( 'Processing super-class %d, topology' % k )
    
    myfilter = np.all( ( allData[:, 0] == superClasses[k, 0], allData[:, 1] == superClasses[k, 1] ), axis = 0 )
    myfilter2 = np.any( ( allData[:, 0] != superClasses[k, 0], allData[:, 1] != superClasses[k, 1] ), axis = 0 )

    trainF = allData[ myfilter2, : ]
    trainF = trainF[ :, 2: ]
    testF = allData[ myfilter, : ]
    testF = testF[ :, 2: ]

    trainC = allData[ myfilter2, : ]
    trainC = trainC[ :, 0 ]
    testC = allData[ myfilter, : ]
    testC = testC[ :, 0 ]

    clf = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(20, 3), max_iter = nIter)

    clf.fit( trainF, trainC )
    aX = clf.predict_proba( testF )[:, 0]

    if np.size( aX, 0 ) > 1:
        while np.sum( np.abs( aX[1:] - aX[:-1] ) < 0.0001 ) > np.size( aX, 0 ) * 0.25:
            clf.fit( trainF, trainC )
            aX = clf.predict_proba( testF )[:, 0]

    allProb[ myfilter, 0 ] = aX

auc_t = roc_auc_score( allData[:, 0], allProb )
fpr_t, tpr_t, _ = roc_curve( allData[:, 0], allProb )



allData = np.load( 'allData.npy' )
allData = np.hstack( ( allData[:, :3], allData[:, 16:18] ) )
allData[ :, 3: ] = normalize( allData[ :, 3: ] )
allProb = np.zeros( ( np.size( allData, 0 ), 1 ) )

superClass = allData[ :, :2 ]
superClasses = np.unique( superClass, axis = 0 )

for k in range( np.size( superClasses, 0 ) ):

    print( 'Processing super-class %d, topology' % k )
    
    myfilter = np.all( ( allData[:, 0] == superClasses[k, 0], allData[:, 1] == superClasses[k, 1] ), axis = 0 )
    myfilter2 = np.any( ( allData[:, 0] != superClasses[k, 0], allData[:, 1] != superClasses[k, 1] ), axis = 0 )

    trainF = allData[ myfilter2, : ]
    trainF = trainF[ :, 2: ]
    testF = allData[ myfilter, : ]
    testF = testF[ :, 2: ]

    trainC = allData[ myfilter2, : ]
    trainC = trainC[ :, 0 ]
    testC = allData[ myfilter, : ]
    testC = testC[ :, 0 ]

    clf = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(20, 3), max_iter = nIter)

    clf.fit( trainF, trainC )
    aX = clf.predict_proba( testF )[:, 0]

    if np.size( aX, 0 ) > 40:
        while np.sum( np.abs( aX[1:] - aX[:-1] ) < 0.0001 ) > np.size( aX, 0 ) * 0.25:
            clf.fit( trainF, trainC )
            aX = clf.predict_proba( testF )[:, 0]

    allProb[ myfilter, 0 ] = aX

auc_m3 = roc_auc_score( allData[:, 0], allProb )
fpr_m3, tpr_m3, _ = roc_curve( allData[:, 0], allProb )



allData = np.load( 'allData.npy' )
allData = np.hstack( ( allData[:, :3], allData[:, 18:] ) )
allData[ :, 3: ] = normalize( allData[ :, 3: ] )
allProb = np.zeros( ( np.size( allData, 0 ), 1 ) )

superClass = allData[ :, :2 ]
superClasses = np.unique( superClass, axis = 0 )

for k in range( np.size( superClasses, 0 ) ):

    print( 'Processing super-class %d, topology' % k )
    
    myfilter = np.all( ( allData[:, 0] == superClasses[k, 0], allData[:, 1] == superClasses[k, 1] ), axis = 0 )
    myfilter2 = np.any( ( allData[:, 0] != superClasses[k, 0], allData[:, 1] != superClasses[k, 1] ), axis = 0 )

    trainF = allData[ myfilter2, : ]
    trainF = trainF[ :, 2: ]
    testF = allData[ myfilter, : ]
    testF = testF[ :, 2: ]

    trainC = allData[ myfilter2, : ]
    trainC = trainC[ :, 0 ]
    testC = allData[ myfilter, : ]
    testC = testC[ :, 0 ]

    clf = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(20, 3), max_iter = nIter)

    clf.fit( trainF, trainC )
    aX = clf.predict_proba( testF )[:, 0]

    if np.size( aX, 0 ) > 40:
        while np.sum( np.abs( aX[1:] - aX[:-1] ) < 0.0001 ) > np.size( aX, 0 ) * 0.25:
            clf.fit( trainF, trainC )
            aX = clf.predict_proba( testF )[:, 0]

    allProb[ myfilter, 0 ] = aX

auc_m4 = roc_auc_score( allData[:, 0], allProb )
fpr_m4, tpr_m4, _ = roc_curve( allData[:, 0], allProb )



