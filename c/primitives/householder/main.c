#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <flint/nfloat.h>
#include <flint/gr.h>
#include <flint/gr_vec.h>
#include <flint/gr_mat.h>
#include "house_qr.h"
bool status = 0;
#ifndef STAT
#define STAT(x) status |= x
#endif
#ifndef TESTPR
#define TESTPR(x) printf("\n====\nTESTING %s\n====\n", x);
#endif
slong rows = 10, cols = 10;


void condition_number(gr_ptr res, gr_mat_t A, gr_ctx_t ctx){
    slong m = A->r, n = A->c;
    gr_ptr normA, normAp;
    GR_TMP_INIT2(normA, normAp,ctx);
    gr_mat_t At, B, Binv, Ap;
    gr_mat_init(At, n, m, ctx);
    gr_mat_init(B, n < m ? n : m, n < m ? n : m, ctx);
    gr_mat_init(Binv, n < m ? n : m, n < m ? n : m, ctx);
    gr_mat_init(Ap, n, m, ctx);

    STAT(gr_mat_transpose(At, A, ctx));

    if (m >= n) {
        // Ap = (A^T A)^(-1) A^T
        STAT(gr_mat_mul(B, At, A, ctx));
        STAT(gr_mat_inv(Binv, B, ctx));
        STAT(gr_mat_mul(Ap, Binv, At, ctx));
    } else {
        // Ap = A^T (A A^T)^(-1)
        STAT(gr_mat_mul(B, A, At, ctx));
        STAT(gr_mat_inv(Binv, B, ctx));
        STAT(gr_mat_mul(Ap, At, Binv, ctx));
    }

    // cond_F(A) = ||A||_F * ||Ap||_F
    STAT(gr_mat_norm_frobenius(normA, A, ctx));
    STAT(gr_mat_norm_frobenius(normAp, Ap, ctx));
    STAT(gr_mul(res, normA, normAp, ctx));
    GR_TMP_CLEAR2(normA, normAp, ctx);
}

int main(){
    gr_ctx_t RR;
    nfloat_ctx_init(RR, 256, 0);

    gr_mat_t A;
    gr_mat_init(A, 2,2, RR);
    STAT(gr_mat_set_str(A, "[[10.,1.],[2.,2.]]", 0, RR));
    TESTPR("REPEATED SQUARING");
    for(int i = 0; i < 3; i++){
        STAT(gr_mat_mul(A,A,A,RR));
	}
    flint_printf("%{gr_mat}\n", A, RR);
    TESTPR("CONDITION NUMBER"); 
    gr_mat_t AT, B;
    GR_MAT_TMP_INIT_SHALLOW_TRANSPOSE(AT, A, RR);
    gr_mat_init(B, AT->r, A->c, RR);
    STAT(gr_mat_mul(B, AT, A, RR));
    gr_ptr cond;
    GR_TMP_INIT(cond, RR);
    condition_number(cond, B, RR);
    flint_printf("condition number %{gr}\n", cond, RR); 
    GR_TMP_CLEAR(cond, RR);
    GR_MAT_TMP_CLEAR_SHALLOW_TRANSPOSE(AT, RR);


    gr_mat_clear(A, RR);
    gr_ctx_clear(RR);
    return 0;	
}
