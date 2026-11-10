# LLL Expository Code
This repository is primarily intended to consist of "easy" implementations of LLL, for understanding the algorithm. This assumes the rank of the lattice is equal to the ambient dimension (i.e. the basis matrix is a square matrix), since this is intended for cryptographic purposes.

- `python/LLL_Pseudocode.py` contains a python-like adaptation of the pseudocode on wikipedia.
- `python/naive_lll_numpy.py` contains an actual, minimal python implementation of the naive LLL algorithm. There are a couple "improvements": instead of computing the unnormalized Gram-Schmidt $B^*$ and the $\mu$, we only compute the $R$ in the $QR$ decomposition, which contains all relevant information. Second, we perform multiple swaps in one pass using the `roll` function, following the LLL algorithm given in Nguyen and Stehlé (not the $L^2$ algorithm). It is "naive" because it performs floating point arithmetic without any regard for precision.
- `python/blaster.py` is "working pseudocode" of the BLASter algorithm by Ducas, Pulles and Stevens. I currently do not have a strong explanation, other than it uses Householder matrices for $QR$ (as `numpy` does) and Seysen reduction (done recursively on block halves) for numerical stability, and that it reduces the lattice matrix on blocks to perform the reduction in parallel (and let BLAS do its job).  One drawback is that it is currently for machine-precision numbers, which is not suitable for cryptographic purposes. Also, my pseudocode is slow.


### Note 1
A note on Nguyen and Stehlé: they check the Lovasz condition without swaps, eventually inserting $b_k$ at the first admissible position. The relevant line of code reads as follows:

- `4.` $\kappa' \leftarrow \kappa$. While $\kappa \geq 2$ and $\delta \cdot r_{\kappa-1, \kappa-1} > r_{\kappa', \kappa'} + \sum_{i = \kappa-1}^{\kappa'-1} \mu^2_{\kappa', i}r_{i,i}$ do $\kappa \leftarrow \kappa-1$.

Note that $\mu_{ij} = R_{ji}/R_{ii}$ and $r_{ii} = R_{ii}$, so $\mu^2_{\kappa',i}r_{i,i} = R_{\kappa',i}^2$.
I found this line difficult to read, so I performed the substitution $(\kappa', \kappa, i) \to (k, k-i, j)$. The line becomes

- `4.` $i \leftarrow 0$. While $i < k$ and $\delta \cdot R_{k-i-1,k-i-1}^2 > R_{k,k} + \sum_{j=0}^i R_{k-j-1,k}^2$ do $i \leftarrow i+1$.

More efficient is to perform one swap at a time, using the Givens matrix (essentially the $Q$ in a $2\times2$ $QR$ decomposition). The idea is that, when you update $R$, you just swap two rows of an upper-triangular matrix, and the only non-upper-triangular part is of the form

$$\begin{bmatrix}R_{k-1,k}& R_{kk}\\\ R_{k-1,k-1} & 0\end{bmatrix},$$

so effectively you get the $Q$ for this and multiply $R$ by $Q$ (note that it also affects the rest of the two rows of $R$). I suppose you can do this for the combined swaps version too, computing the relevant $Q$ and multiplying $R$ by this $Q$.