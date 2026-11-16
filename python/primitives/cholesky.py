# Following recursive cholesky on a Gram matrix
import numpy as np
import scipy as sp
def cholesky(G):
    if G.shape[0] == 1:
        return np.array([[np.sqrt(G[0, 0])]])
    m = G.shape[0]
    mh = m//2
    RA = cholesky(G[:mh, :mh])
    if mh < 64:
        S = np.linalg.solve(RA.T, G[:mh, mh:])
    else:
        #S = sp.linalg.inv(RA.T) @ G[:mh, mh:]
        S = sp.linalg.solve_triangular(RA.T, G[:mh, mh:], lower=True, check_finite=False)
    RC = cholesky(G[mh:, mh:] - S.T.dot(S))
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