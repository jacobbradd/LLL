from householder import *
import scipy as sp
from cholesky import qr_cholesky, qr_cholesky_sing


import time
time_fn = lambda label, f, *args, **kwargs: (
    s := time.perf_counter(), 
    f(*args, **kwargs), 
    print(f"{label}: {time.perf_counter() - s:.6f}s")
)

rng = np.random.default_rng()
B = rng.integers(low=-2**32, high=2**32, size=(2000, 2000)).astype(np.float64)

sp.linalg.solve(B, B) #warmup sp
np.linalg.solve(B,B) #warmup np
print("B:", B)
time_fn("qr_wy", qr_wy_r_mode, B)
time_fn("qr_lq", qr_lq_r_mode, B)
time_fn("numpy", np.linalg.qr, B, mode="r")
time_fn("qr_cholesky", qr_cholesky, B)

print("WY:", qr_wy(B, r_mode=True))
print("LQ:",qr_lq_r_mode(B))
print("NP:",np.linalg.qr(B, mode="r"))
print("CH:", qr_cholesky(B))