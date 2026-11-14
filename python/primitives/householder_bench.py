from householder import *

B = np.array([[4, 1, 2, -1], [3, 5, -2, 2], [1, -2, 6, 3], [2, 1, 1, 4]])
print(qr_wy(B)[1])


rng = np.random.default_rng()


import time
time_fn = lambda label, f, *args, **kwargs: (
    s := time.perf_counter(), 
    f(*args, **kwargs), 
    print(f"{label}: {time.perf_counter() - s:.6f}s")
)

B = rng.integers(low=10000, high=1000000000, size=(500, 500))
time_fn("qr_wy", qr_wy, B)
time_fn("numpy", np.linalg.qr, B, mode="reduced")


print(qr_wy(B))

print(np.linalg.qr(B))