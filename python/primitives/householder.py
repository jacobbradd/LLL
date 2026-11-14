# Householder is just vector reflection: H_u(v) = v - 2\langle v,u \rangle/\|u\|^2 u.
# In matrices: H_u = I - 2uu^T/\|u\|^2
# This can be used to stably compute the R in QR: 
#   recursively, on R[i:n, i:n], reflect the first column onto the basis vector e_1 = (1, 0, .., 0)
#   for first column b_1, this is H_u with u = b_1 - \|b_1\|e_1 
#   to avoid fp "cancellation", use u = b_1 + \|b_1\|e_1 if b_1 < 0
#   All together this gives matrices H_1, \ldots, H_n, and R = H_n \cdots H_1 A.

import numpy as np

# used in the "level 3" version
def householder(u):
    u=u.copy().astype(np.float64) #just in case
    u[0] += np.copysign(np.linalg.norm(u), u[0])
    tau = 2 / np.dot(u, u)
    return u, tau
    

def householder_matrix(u):
    u=u.copy().astype(np.float64) #just in case
    if np.all(u[1:] == 0):
        return np.eye(len(u))
    u[0] += np.copysign(np.linalg.norm(u), u[0])
    return np.eye(len(u)) - 2*np.outer(u,u)/np.dot(u,u)
    
# This is slow but very easy
def qr_householder(B):
    m, n = B.shape
    Q = np.eye(m)
    R = B.copy()
    for k in range(0, n-1):
        H_k = np.eye(m)
        H_k[k:,k:] = householder_matrix(R[k:, k])
        Q = Q @ H_k
        R = H_k @ R
    return Q, R
    
# Combines Householder and QR in one step. Surprisingly decent.
# Future exercise is to bring it to "Level 3"
def qr_combined(B):
    m, n = B.shape
    Q = np.eye(m)
    R = B.copy().astype(np.float64)
    for k in range(0, n-1):
        u = R[k:, k].copy()
        if np.all(u[1:] == 0):
            continue
        u[0] += np.copysign(np.linalg.norm(u), u[0])
        tau = (2/np.dot(u,u))
        Q[:,k:] -= tau*np.outer(Q[:,k:] @ u, u)
        R[k:,k:] -= tau*np.outer(u, u @ R[k:,k:])
    return Q, R

# The "WY" method (named for whatever reason by Bischof-Van Loan)
# The point is you can assemble the product H_i .. H_j by hand given the Householder vectors u_i .. u_j
# Bischof-Van Loan calculated this product, a small trick is to keep track of the \tau = 2/\|u\|^2
# The u's assemble into a matrix V, the \tau's help assemble a matrix T,  then the product is I - VT(V^T)
# Batch 64-128 looks decent for 4000+, 32 for a good lower range
def qr_wy(B, b = 64, r_mode = False):
    if r_mode == True:
        return qr_wy_r_mode(B, b)
    m, n = B.shape
    Q, R = np.eye(m), B.copy().astype(np.float64)
    for k in range(0, n, b):
        p = min(b, n-k)
        V, T = np.zeros((m-k, p)), np.zeros((p, p))
        for j in range(p):
            u, tau = householder(R[k+j:, k+j]) 
            V[j:, j] = u
            T[j,j] = tau
            if j > 0:
                T[:j,j] = -tau * T[:j,:j] @ (V[:, :j].T @ V[:, j])
            # Need to update R the least amount to compute the next householder (see qr_combined)
            R[k+j:, k+j:k+p] -= tau * np.outer(u, u @ R[k+j:, k+j:k+p])
        R[k:, k+p:] -= V @ (T.T @ (V.T @ R[k:, k+p:]))
        Q[:, k:] -= ((Q[:, k:] @ V) @ T) @ V.T
    return Q,R
    
def qr_wy_r_mode(B, b = 64):
    m, n = B.shape
    R = B.copy().astype(np.float64)
    for k in range(0, n, b):
        p = min(b, n-k)
        V, T = np.zeros((m-k, p)), np.zeros((p, p))
        for j in range(p):
            u, tau = householder(R[k+j:, k+j])
            V[j:, j] = u 
            T[j,j] = tau
            if j > 0:
                T[:j,j] = -tau * T[:j,:j] @ (V[:, :j].T @ V[:, j])
            # Need to update R the least amount to compute the next householder (see qr_combined)
            R[k+j:, k+j:k+p] -= tau * np.outer(u, u @ R[k+j:, k+j:k+p]) 
        R[k:, k+p:] -= V @ (T.T @ (V.T @ R[k:, k+p:]))
    return R   