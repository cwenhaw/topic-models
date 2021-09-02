from sklearn.decomposition import NMF
import numpy as np
import sys
from timeit import default_timer as timer
from numpy import linalg as LA
import scipy.sparse as sps
from scipy import sparse
import scipy
from numpy.core.umath_tests import inner1d

REDUCE_PREC=False   # doesn't work as timereg can be huge. dtype of matrices can get upcasted during operations

# equialent to trace(A' * B)
def tr(A, B):
    elementprod=np.multiply(A, B)
    value=np.sum(elementprod)
    return value

def demo():
    A=np.random.rand(10)
    print(tr(A,A))
    fro_norm=LA.norm(A)
    print(np.power(fro_norm,2))

# very expensive. adds a few hundred % to compute time
# timereg*trace(I) omitted as it's a constant
# omitted constants
#   timereg*trace(I)
#   trXX
def compute_loss_slow(X, W, H, M, R, l1reg, timereg):
    MR = np.matmul(M, R)
    WH = np.matmul(W, H)
    WMR = np.matmul(W, MR)
    tr1 = - 2*tr(X,WH) + tr(WH,WH)
    tr2 = - 2*tr(X,WMR) + tr(WMR,WMR)
    tr3 = timereg * (tr(M,M) - 2*np.trace(M))
    tr4 = l1reg * (np.sum(H) + np.sum(W) + np.sum(M))
    loss = tr1 + tr2 + tr3 + tr4
    return loss

# Huge matrices not computed since we only want trace
# sequence of matrix multiplication is designed s.t. only smaller matrices are in memory
# constant tr(X'X) can be excluded. Include to get exact value with naive compute
def compute_loss(X, W, H, M, R, l1reg, timereg):
    MR = np.matmul(M, R)
    if scipy.sparse.issparse(X):
        norm_X=sparse.linalg.norm(X)
    else:
        norm_X=LA.norm(X)
    TrXX = norm_X * norm_X  #tr(X'X)
    constant=2*TrXX

    XtW = (X.transpose()).dot(W)    # V x K
    tr1_1=np.sum(inner1d(XtW, H.T))
    Ht=H.transpose()
    WtW=(W.T).dot(W)    # K x K
    HtWtW=Ht.dot(WtW)   # V x K
    tr1_2=np.sum(inner1d(HtWtW, Ht))    # trace(H'W'WH)
    tr1=-2*tr1_1 + tr1_2

    XtWM=np.matmul(XtW, M)
    tr2_1=np.sum(inner1d(XtWM, R.T))
    RtMtWtW=np.matmul(MR.T, WtW)
    tr2_2=np.sum(inner1d(RtMtWtW, MR.T))    # trace(R'M'W'WMR)
    tr2=-2*tr2_1 + tr2_2

    norm_M = LA.norm(M)
    trMM = norm_M * norm_M
    tr3 = timereg * (trMM - 2*np.trace(M))
    tr4 = l1reg * (np.sum(H) + np.sum(W) + np.sum(M))
    loss = tr1 + tr2 + tr3 + tr4 + constant
    return loss
 
# omitted constants: timereg*trace(I)
# included constants: Tr(X'X)
# 35% faster than slow version
# mem intensive cos WH, WMR matrics are not sparse
def compute_loss_naive(X, W, H, M, R, l1reg, timereg):
    MR = np.matmul(M, R)
    WH = np.matmul(W, H)    # huge
    WMR = np.matmul(W, MR)  # huge
    tr1_norm = LA.norm(X-WH)
    tr1 = tr1_norm * tr1_norm
    tr2_norm = LA.norm(X-WMR)
    tr2 = tr2_norm * tr2_norm
   
    '''   
    # for checking: loss values will sync with the slow version / mem-light version ==
    norm_X = LA.norm(X)
    TrXX = norm_X * norm_X
    tr1 -= TrXX
    tr2 -= TrXX
    '''

    norm_M = LA.norm(M)
    trMM = norm_M * norm_M
    tr3 = timereg * (trMM - 2*np.trace(M))
    tr4 = l1reg * (np.sum(H) + np.sum(W) + np.sum(M))
    loss = tr1 + tr2 + tr3 + tr4
    return loss

def nmf_loss_naive(X, W, H, l1reg):
    WH = np.matmul(W, H)
    tr1_norm = LA.norm(X-WH)
    tr1 = tr1_norm * tr1_norm
    loss = tr1 + l1reg *(np.sum(H)+np.sum(W))
    return loss

def nmf_loss(X, W, H, l1reg):
    XtW = (X.transpose()).dot(W)    # V x K
    tr1_1=np.sum(inner1d(XtW, H.T))
    Ht=H.transpose()
    WtW=(W.T).dot(W)    # K x K
    HtWtW=Ht.dot(WtW)   # V x K
    tr1_2=np.sum(inner1d(HtWtW, Ht))    # trace(H'W'WH)
    tr1=-2*tr1_1 + tr1_2
    loss = tr1 + l1reg *(np.sum(H)+np.sum(W))
    '''
    if scipy.sparse.issparse(X):
        norm_X=sparse.linalg.norm(X)
    else:
        norm_X=LA.norm(X)
    TrXX = norm_X * norm_X # tr(X'X)
    loss+=TrXX
    '''
    return loss

# B not yet regularized
def gnmf_loss(X, W, H, Y, B, l1reg, Swt):
    # NMF part
    XtW = (X.transpose()).dot(W)    # V x K
    tr1_1=np.sum(inner1d(XtW, H.T))
    Ht=H.transpose()
    WtW=(W.T).dot(W)    # K x K
    HtWtW=Ht.dot(WtW)   # V x K
    tr1_2=np.sum(inner1d(HtWtW, Ht))    # trace(H'W'WH)
    tr1=-2*tr1_1 + tr1_2

    # supervision part
    YtB = (Y.transpose()).dot(B)
    tr2_1=np.sum(inner1d(YtB, H.T))
    BtB=(B.T).dot(B)
    HtBtB=Ht.dot(BtB)
    tr2_2=np.sum(inner1d(HtBtB, Ht))    # trace(H'B'BH)
    tr2=-2*tr2_1 + tr2_2
    
    loss = tr1 + Swt*tr2 + l1reg *(np.sum(H)+np.sum(W))
    '''
    # Constants
    if scipy.sparse.issparse(X):
        norm_X=sparse.linalg.norm(X)
    else:
        norm_X=LA.norm(X)
    TrXX = norm_X * norm_X # tr(X'X)
    
    if scipy.sparse.issparse(Y):
        norm_Y=sparse.linalg.norm(Y)
    else:
        norm_Y=LA.norm(Y)
    TrYY = norm_Y * norm_Y # tr(Y'Y)
    constant = TrXX + Swt*TrYY
    loss+=constant
    '''
    return loss
    
def gnmf_loss_naive(X, W, H, Y, B, l1reg, Swt):
    WH = np.matmul(W, H)
    tr1_norm = LA.norm(np.subtract(X, WH))
    tr1 = tr1_norm * tr1_norm
    
    BH = np.matmul(B, H)
    tr2_norm = LA.norm(np.subtract(Y, BH))
    tr2 = tr2_norm * tr2_norm
    
    loss = tr1 + Swt*tr2 + l1reg *(np.sum(H)+np.sum(W))
    return loss    

def GNMF(X, Y, K, l1reg, Swt, maxIter, computeLoss):    
    (N, V)=np.shape(X)
    (S, V)=np.shape(Y)
    W=np.random.rand(N,K)
    H=np.random.rand(K,V)
    B=np.random.rand(S,K)  # topic supervision matrix
    
    eps=sys.float_info.epsilon
    for iter in range(0, maxIter):
        # W =  W .* ( X*H' ./ max( WHH'+ lambda, eps) )        
        if scipy.sparse.issparse(X):
            numer = X.dot(H.transpose())
        else:
            numer = np.matmul(X, H.transpose())
        temp = np.matmul(H, H.transpose()) 
        denom = np.matmul(W, temp) + l1reg
        denom = np.maximum(denom, eps)    
        W = np.multiply(W, np.divide(numer,denom))
    
        # H = H .* (W'X./max(W'WH + B'BH + lambda,eps))
        WtW = np.matmul(W.transpose(), W)
        if scipy.sparse.issparse(X):
            XtW = (X.transpose()).dot(W)
            WtX = XtW.transpose()
        else:
            WtX = np.matmul(W.transpose(), X)
        numer = WtX + Swt*np.matmul(B.transpose(),Y)  # W'X + B'Y
        BtB = np.matmul(B.tranpose(), B)
        denom = np.matmul(WtW + Swt*BtB, H) + l1reg   # (W'W + B'B)HH
        denom = np.maximum(denom, eps)
        H = np.multiply(H, np.divide(numer,denom))
        
        # B = B.* ( YH' / BHH' )  # analogus to updates for WH
        numer = np.matmul(Y, H.transpose())
        HHt = np.matmul(H, H.transpose())
        denom = np.matmul(B, HHt)
        denom = np.maximum(denom, eps) # cannot divide by 0
        B = np.multiply(B, np.divide(numer, denom))
        
        if computeLoss==True:
            loss = gnmf_loss(X, W, H, Y, B, l1reg, Swt)
            print(loss)
    return [W, H, B]

def JAL_NMF(X, K, l1reg, maxIter, computeLoss):    
    (N, V)=np.shape(X)
    W=np.random.rand(N,K)
    H=np.random.rand(K,V)
    
    eps=sys.float_info.epsilon
    for iter in range(0, maxIter):
        # W =  W .* ( X*H' ./ max( WHH'+ lambda, eps) )
        if scipy.sparse.issparse(X):
            numer = X.dot(H.transpose())
        else:
            numer = np.matmul(X, H.transpose())
        temp = np.matmul(H, H.transpose()) 
        denom = np.matmul(W, temp) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        W = np.multiply(W, np.divide(numer,denom))
        
        # H = H .* (WtX./max(WtW*H+lambda,eps))
        WtW = np.matmul(W.transpose(), W)
        if scipy.sparse.issparse(X):
            XtW = (X.transpose()).dot(W)
            WtX = XtW.transpose()
        else:
            WtX = np.matmul(W.transpose(), X)
        
        denom = np.matmul(WtW, H) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        H = np.multiply(H, np.divide(WtX,denom))
        if computeLoss==True:
            loss=nmf_loss(X, W, H, l1reg)
            print('iter', iter, 'loss', loss)
        
    return [W, H]


# handles both dense, sparse X matrix
# objective function in compute_loss_slow()
def JPP(X, R, K, timereg, l1reg, loss_delta, maxIter):
    # random init param    
    (N, V)=np.shape(X)
    W=np.random.rand(N, K) # doc df over topics
    H=np.random.rand(K, V)  # topic df over words
    M=np.random.rand(K, K)  # topic-topic mapping

    scaled_I = timereg * np.identity(K)    # constants
    eps=sys.float_info.epsilon

    loss=float('inf')
    start = timer()
    for iter in range(0, maxIter):
        J=np.matmul(M, R)

        # From Matlab: W = W .* ( X(H'+J') ./ max( W( JJ'+HH'+lambda ), eps) )
        # implemented here: W = W .* ( X(H'+J') ./ max( W(JJ'+HH')+lambda, eps) )
        H_J = np.add(H.transpose(), J.transpose())
        if scipy.sparse.issparse(X):
            numer = X.dot(H_J)
        else:
            numer = np.matmul(X, H_J)
        temp = np.matmul(J, J.transpose()) + np.matmul(H, H.transpose()) 
        denom = np.matmul(W, temp) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        W = np.multiply(W, np.divide(numer,denom))

        # M = M .* ( (WtX*R' + alpha*I) ./ max( WtW*M*R*R' + alpha*M+lambda,eps) ); 
        WtW = np.matmul(W.transpose(), W)
        if scipy.sparse.issparse(X):
            XtW = (X.transpose()).dot(W)
            WtX = XtW.transpose()
        else:
            WtX = np.matmul(W.transpose(), X)
        WtXRt = np.matmul(WtX,R.transpose())
        numer = np.add( WtXRt, scaled_I )
        #np.fill_diagonal(WtXRt, WtXRt.diagonal() + timereg)    # matrix is small -> immaterial
        #numer=WtXRt
        WtWM = np.matmul(WtW,M)
        RRt = np.matmul(R, R.transpose())
        WtWMRRt = np.matmul(WtWM, RRt)      #   WtW*M*R*R'

        denom = np.add(WtWMRRt, timereg*M) + l1reg    # + alpha*M + lambda
        denom = np.maximum(denom, eps)    # ensure non-negativity
        M = np.multiply(M, np.divide(numer,denom))

        # H = H .* (WtX./max(WtW*H+lambda,eps));
        denom = np.matmul(WtW, H) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        H = np.multiply(H, np.divide(WtX,denom))
        '''              
        prev_loss=loss
        loss=compute_loss_naive(X, W, H, M, R, l1reg, timereg)    # expensive to eval
        print(loss)
        if prev_loss - loss < loss_delta:   # reduction in loss < threshold
            print("break at iter", iter)
            break
        '''
        print(iter)                     
    endtime=timer()
    print('JPP elapsed', endtime-start)
    return [H,W,M]

def sparse_verify():
    np.random.seed(123)
    X=np.random.rand(3,3) 
    Y=np.random.rand(3,3)
    XY=np.matmul(X,Y)
    print(XY)

    print('sparse')
    X = sps.csr_matrix(X)
    Y = sps.csr_matrix(Y)
    r = X.dot(Y)
    print(r.toarray(), '\n')

    print(r.transpose().toarray())

def expt():
    K=20
    N1=1000
    V=5000

    start = timer()
    np.random.seed(123)
    X=np.random.rand(N1,V)   # data at time t-1
    model = NMF(n_components=K, init='random', random_state=0, solver='mu', l1_ratio=1, alpha=5 )
    W = model.fit_transform(X)
    R = model.components_   # H(t-1)
    endtime=timer()
    print('NMF elapsed', endtime-start)
    N2=1000
    X=np.random.rand(N2,V)   # data at time t

    timereg=10000000   # temporal regularization 
    l1reg=0.05      # L1 regularization
    maxIter=100
    loss_delta=0.01     # stopping criterion
    [H,W,M]=JPP(X, R, K, timereg, l1reg, loss_delta, maxIter)
    print('final-W', W[110,0:10])
    print('final-H', H[10,0:10])
    print('final-M', M[10,0:10])

def sparse_expt():
    K=20
    N1=1000
    V=5000

    start = timer()
    np.random.seed(123)
    X=np.random.rand(N1,V)   # data at time t-1
    X = sps.csr_matrix(X)
    model = NMF(n_components=K, init='random', random_state=0, solver='mu', l1_ratio=1, alpha=5 )
    W = model.fit_transform(X)      # slow if using sparse structure for non-sparse data
    R = model.components_   # H(t-1)
    endtime=timer()
    print('NMF elapsed', endtime-start)

    N2=1000
    X=np.random.rand(N2,V)   # data at time t
    
    X = sps.csr_matrix(X)

    timereg=10000000   # temporal regularization 
    l1reg=0.05      # L1 regularization
    maxIter=100
    loss_delta=0.01     # stopping criterion
    [H,W,M]=JPP_sparse(X, R, K, timereg, l1reg, loss_delta, maxIter)
    print('final-W', W[110,0:10])
    print('final-H', H[10,0:10])
    print('final-M', M[10,0:10])

def GNMF_expt():
    np.random.seed(123)
    N=10000
    V=6000
    X=np.random.rand(N,V)
    S=5  # no. of seed topics
    n=3  # no. of words for each seed topic
    Y=np.zeros((S,V))
    for s in range(0, S):
        kw=np.random.permutation(V)
        kw=kw[:n]
        for w in kw:
            Y[s,w]=1
    K=10
    computeLoss=True
    maxIter=200
    l1reg=1.0
    Swt=50.0    # supervision strength
    topn=30
    [W, H, B]=GNMF(X, Y, K, l1reg, Swt, maxIter, computeLoss)
    for k in range(0, K):
        sorted_wid=H[k,:].argsort()[::-1]
    
    
def NMF_expt():
    maxIter=200
    K=50
    l1reg=1.0	# L1 regularization parameter
    topn=30
    for run in range(0, 1):
        np.random.seed(run)
        [W, H]=JAL_NMF(X, K, l1reg, maxIter, computeLoss)
        
        for k in range(0, K):
            sorted_wid=H[k,:].argsort()[::-1]    


#sparse_verify()
#expt()
#sparse_expt()
