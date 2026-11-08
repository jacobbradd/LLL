# Idea: recurse on upper triangular matrices
import numpy as np
def seysen(R):
    n = R.shape[0]
    if n == 1:
        return np.eye(1, dtype=int), R
    lh = n//2
    UA, A = seysen(R[:lh, :lh])
    UC, C = seysen(R[lh:, lh:])
    B = R[:lh, lh:]
    X = -np.rint(np.linalg.solve(A, B @ UC)).astype(int)
    U = np.block([
        [UA, UA @ X],
        [np.zeros((n-lh, lh), dtype=int), UC]
    ])
    B[:] = A @ X + B @ UC
    return U, R
print(seysen(np.array([[15,55,93],[0,12,29],[0,0,2]])))
A = np.array([
    [8, 21, 55, 132],
    [0, 5, 22, 64],
    [0, 0, 9, 38],
    [0, 0, 0, 2]
])
print(seysen(A))