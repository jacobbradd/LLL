# Just pseudocode essentially of Ducas-Pulles-Stevens BLASter algorithm
# Definitely unoptimised and currently slower than just running lll_reduction


import numpy as np
from numpy_naive_lll import lll_reduction
from primitives.seysen import seysen

    
def lovasz_check(R, delta=0.75):
    d = np.diag(R)
    return np.all(delta * d[:-1]**2 <= d[1:]**2 + np.diag(R, k=1)**2)


def blaster(B, delta=0.75, block=64, iter_budget=-1):
    i0 = 0
    m, n = B.shape
    while iter_budget != 0:
        iter_budget -= 1
        i0 = block//2 - i0
        R = np.linalg.qr(B, mode="r")
        Vs = []
        for i in range(i0, n, block):
            p = min(block, n - i)
            Vs.append((i, p, lll_reduction(R[i:i+p, i:i+p], delta))) #should do in parallel
        for i, p, V in Vs:
            B[:, i:i+p] = B[:, i:i+p] @ V
        R = np.linalg.qr(B, mode="r")
        W, R = seysen(R)
        B = B @ W
        if lovasz_check(R, delta):
            return B
    print("Failed to converge in time")
    return B

### Parallel attempt
from concurrent.futures import ProcessPoolExecutor
def _lll_worker(task):
    i, p, slice, delta = task
    return i, p, lll_reduction(slice, delta)


def blaster_parallel_attempt(B, delta=0.75, block=64, iter_budget=-1, workers=None):
    i0 = 0
    m, n = B.shape
    with ProcessPoolExecutor(max_workers=workers) as executor: # Parallel
        while iter_budget != 0:
            iter_budget -= 1
            i0 = block//2 - i0
            R = np.linalg.qr(B, mode="r")
            ### Parallel part: should really be multiplying B in parallel too? (It's not in the paper)
            tasks = [(i, min(block, n - i), R[i:i+min(block, n-i), i:i+min(block, n-i)], delta)
                        for i in range(i0, n, block)]
            for i, p, V in executor.map(_lll_worker,tasks):
                B[:, i:i+p] = B[:, i:i+p] @ V
            ###
            R = np.linalg.qr(B, mode="r")
            W, R = seysen(R)
            B = B @ W
            if lovasz_check(R, delta):
                return B
    print("Failed to converge in time")
    return B
###


def check_weak_reduced(B, delta=0.75):
    R = np.linalg.qr(B, mode="r")
    return lovasz_check(R, delta)
    
import os    
if __name__ == "__main__":
    rng = np.random.default_rng(43)
    B = rng.integers(-10**5, 10**5, size=(1000, 1000))
    
    print(B)
    import time
    time_fn = lambda label, f, *args, **kwargs: (
        s := time.perf_counter(), 
        f(*args, **kwargs), 
        print(f"{label}: {time.perf_counter() - s:.6f}s")
    )
    time_fn("blaster", blaster, B.copy())
    time_fn("blaster (parallel attempt)", blaster_parallel_attempt, B.copy(), workers=os.cpu_count())
    time_fn("lll_reduction", lll_reduction, B.copy())
    print(check_weak_reduced(blaster(B.copy())))
    print(check_weak_reduced(B @ lll_reduction(B.copy())))
    
