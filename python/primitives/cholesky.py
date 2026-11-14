# Following recursive cholesky on a Gram matrix
import numpy as np
def cholesky(G):
    if G.shape[0] == 1:
        return np.array([[np.sqrt(G[0, 0])]])
    m,n = G.shape
    mh = m//2
    nh = n//2
    RA = cholesky(G[:mh, :nh])
    S = np.linalg.solve(RA.T, G[:mh, nh:])
    RC = cholesky(G[mh:, nh:] - S.T.dot(S))
    R = np.block([
        [RA, S],
        [np.zeros(S.T.shape), RC]
    ])
    return R
    
def qr_cholesky(B):
    B = B.astype(np.float64)
    return cholesky(B.T.dot(B))
    
def qr_cholesky_sing(B): #for singular matrices
    B = B.astype(np.float64)
    return cholesky(B.T.dot(B) + 1e-10 * np.eye(B.shape[1]))