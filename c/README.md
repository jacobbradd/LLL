# C Code

Testing out [FLINT](https://github.com/flintlib/flint) for arbitrary precision matrix multiplication.

`householder` is analogous to `python/primitives/householder.py`, particularly `qr_wy_r_mode`.
It is currently inefficient, partly because I decided to do everything mathematically (with column vectors) instead of using rows, which FLINT is faster with.

To install FLINT: follow [the instructions](https://github.com/flintlib/flint#building-from-source) with linux/WSL.
Then run `make` on the folder.