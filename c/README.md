# C Code

Testing out [FLINT](https://github.com/flintlib/flint) for arbitrary precision matrix multiplication.

`householder` is analogous to `python/primitives/householder.py`, particularly `qr_wy_r_mode`. I tried the LQ one which is only slightly faster on C. I suspect FLINT was already smart enough to handle memory layout. 
Either way, it's slower than FLINT's one even though it only computes R (I'm not sure what's wrong).

To use: install FLINT by following [the instructions](https://github.com/flintlib/flint#building-from-source) with linux/WSL.
Then run `make` on the folder. 

Run `./bench` and input matrices in numpy fashion, e.g. `[[1,2],[3,4]]`.
Arguments: `-t` to time, `-b [num bits]` to set precision, `-f` runs FLINT's QR, `-q` suppresses printing the matrices.