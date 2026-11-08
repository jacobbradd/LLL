# Householder is just vector reflection: H_u(v) = v - 2\langle v,u \rangle/\|u\|^2 u.
# In matrices: H_u = I - 2uu^T/\|u\|^2
# This can be used to stably compute the R in QR: 
#   recursively, on R[i:n, i:n], reflect the first column onto the basis vector e_1 = (1, 0, .., 0)
#   for first column b_1, this is H_u with u = b_1 - \|b_1\|e_1 
#   to avoid fp "cancellation", use u = b_1 + \|b_1\|e_1 if b_1 < 0

import numpy as np

def householder(u):
    u=u.copy().astype(np.float64) #just in case
    if np.all(u[1:] == 0):
        return np.eye(len(u))
    u[0] += np.copysign(np.linalg.norm(u), u[0])
    return np.eye(len(u)) - 2*np.outer(u,u)/np.dot(u,u)
    
# This is slow but very easy
def householder_qr(B):
    m, n = B.shape
    Q = np.eye(m)
    R = B.copy()
    for k in range(0, n-1):
        H_k = np.eye(m)
        H_k[k:,k:] = householder(R[k:, k])
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
        Q[:,k:] -= (2/np.dot(u,u))*np.outer(Q[:,k:] @ u, u)
        R[k:,k:] -= (2/np.dot(u,u))*np.outer(u, u @ R[k:,k:])
    return Q, R
    
B = np.array([
[3,5],
[4,2]
])
print(householder(B[:,0]))

print(householder_qr(B))
print(qr_combined(B))
print(np.linalg.qr(B))


rng = np.random.default_rng()

B = rng.standard_normal((500, 500))
B = rng.integers(low=1, high=101, size=(500, 500))



import time
#start = time.perf_counter()
#_,_ = QR(B)
#print("QR:", time.perf_counter()-start)
start = time.perf_counter()
_,_ = qr_combined(B)
print("qr_Combined:", time.perf_counter()-start)
start = time.perf_counter()
_,_ = np.linalg.qr(B, mode='reduced')
print("numpy:", time.perf_counter()-start)   

