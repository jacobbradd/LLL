"""
INPUT: 
    Lattice basis B of size n as an array of integer arrays
    Parameter δ with 1/4 < δ < 1, usually 3/4
OUTPUT of LLL: 
    Reduced basis for B
INTERMEDIATE VARIABLES: 
    Unnormalized Gram-Schmidt basis B* of B, 
    μ[i][j] = <B[i], B*[j]>/<B*[i], B*[j]>
"""

def SizeReduce(B, B*, μ, k):
    for j in range(k-1, 1, -1):
        if abs(μ[k][j]) > 1/2:
            B[k] -= round(μ[k][j]) * B[j]
            # Update B* and μ from B
            UpdateAfterSize(B, B*, μ)
def SwapReduce(B, δ, B*, μ, k):
    if InnerProduct(B*[k], B*[k]) > (δ - μ[k][k-1]**2) * InnerProduct(B*[k-1], B*[k-1]):
        k -= 1
    else:
        B[k], B[k-1] = B[k-1], B[k]
        UpdateAfterSwap(B, B*, μ)
        k = max(k-1, 2)

def LLL(B, δ):
    B* = GramSchmidt(B)
    μ[i,j] = InnerProduct(B[i],B*[j])/InnerProduct(B*[j],B*[j])
    k = 2
    while k <= n:
        SizeReduce(B, B*, μ, k)
        SwapReduce(B, δ, B*, μ, k)
    return B

            
        