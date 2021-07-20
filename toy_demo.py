import numpy as np
from jpp import JAL_NMF

def NMF_expt():
    maxIter=200 
    K=50    # no. of topics
    l1reg=1.0   # L1 regularization parameter
    computeLoss=True    # whether loss function should be computed per iteration
    
    D=10000
    V=5000
    np.random.seed(121)
    X=np.random.rand(D,V)   # fake toy data with dense matrix
    # Note: replace with sparse X for real data

    [W, H]=JAL_NMF(X, K, l1reg, maxIter, computeLoss)

    #topn=30     # no. of top words to show per topic
    #for k in range(0, K):
    #    sorted_wid=H[k,:].argsort()[::-1]


NMF_expt()
