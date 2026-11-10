"""
INPUT: 
    Lattice basis B of size n as an array of integer arrays
    Parameter δ with 1/4 < δ < 1, default 3/4
OUTPUT of LLL: 
    Reduced basis for B
"""

import numpy as np

# Todo: express as calls to Babai
def size_reduce(B, R, k):
    for j in range(k-1, -1, -1):
        mu = R[j][k]/R[j][j]
        if abs(mu) > 0.5:
            B[:, k] -= round(mu) * B[:, j]
            R[:, k] -= round(mu) * R[:, j]
    return B, R
        
def swap_reduce(B, delta, R, k):
    i = 0
    V = R[k-1][k]**2
    while i < k and delta*(R[k-i-1][k-i-1]**2) > R[k][k]**2 + V:
        i += 1
        if i < k:
            V += R[k-i-1][k]**2
    if i > 0:
        # Insert b_k before b_{k-i}, shifting everything forward
        B[:, k-i : k+1] = np.roll(B[:, k-i : k+1], shift=1, axis=1)
        R[:, k-i : k+1] = np.roll(R[:, k-i : k+1], shift=1, axis=1)
        
        Givens, _ = np.linalg.qr(R[k-i:k+1, k-i:k+1])
        R[k-i:k+1, :] = Givens.T @ R[k-i:k+1, :]
        k -= i
    return B, R, k
    
def lll_naive(B, delta=0.75):
    n = B.shape[1]
    R = np.linalg.qr(B, mode="r")
    k = 1
    while k < n:
        B, R = size_reduce(B, R, k)
        B, R, k = swap_reduce(B, delta, R, k)
        k = k + 1
    return B
    
import numpy as np
def lll_reduction(B, delta=0.75):
    n = B.shape[0]
    U = np.eye(n)
    R = np.linalg.qr(B, mode = "r")
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            mu = R[j][k]/R[j][j]
            if abs(mu) > 0.5:
                B[:,k] -= round(mu) * B[:,j]
                U[:,k] -= round(mu) * U[:,j]
                R[:,k] -= round(mu) * R[:,j]
        if R[k][k]**2 + R[k-1][k]**2 >= delta * R[k-1][k-1]**2:
            k += 1
        else:
            B[:,[k, k - 1]] = B[:,[k - 1, k]].copy()
            U[:,[k, k - 1]] = U[:,[k - 1, k]].copy()
            R[:,[k, k - 1]] = R[:,[k - 1, k]].copy()
            Givens, _ = np.linalg.qr(R[k-1:k+1,k-1:k+1])
            R[k-1:k+1, :] = Givens.T @ R[k-1:k+1, :]
            k = max(k - 1, 1)
    return U.astype(np.int64)
        

#print(lll_reduction(B))
#B = np.array([
#[1,5], [2,3]
#])
#lll_naive(B, 0.9)
#B = np.array([[6,7],[1,1]])
#lll_naive(B, 0.9)
#B = np.array([[7,2,3],[4,5,6],[7,8,9]])
#print(lll_naive(B))
#print(lll_reduction(B))
