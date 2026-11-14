#ifndef __HOUSE_QR_H__
#define __HOUSE_QR_H__

#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <flint/nfloat.h>
#include <flint/gr.h>
#include <flint/gr_vec.h>
#include <flint/gr_mat.h>
extern bool status;
#ifndef STAT
#define STAT(x) status |= x
#endif
#ifndef TESTPR
#define TESTPR(x) printf("\n====\nTESTING %s\n====\n", x);
#endif

void householder(gr_mat_t res, gr_ptr tau, gr_mat_t vect, gr_ctx_t ctx){
    STAT(gr_mat_set(res, vect, ctx));
    gr_ptr norm, two, nsign;
    GR_TMP_INIT3(norm,two,nsign, ctx);
    STAT(gr_mat_norm_frobenius(norm, res, ctx));
    gr_ptr u0 = gr_mat_entry_ptr(res,0,0,ctx);
	int sign;
	STAT(gr_zero(nsign, ctx));
	STAT(gr_cmp(&sign, u0, nsign, ctx));
	STAT(gr_set_si(nsign, sign < 0 ? -1 : 1, ctx));
    STAT(gr_addmul(u0, norm, nsign, ctx)); 
    STAT(_gr_vec_dot_strided(tau, NULL, 0, res->entries,
                res->stride, res->entries,res->stride,
                res->r, ctx));
    STAT(gr_set_ui(two, 2, ctx));
    STAT(gr_div(tau, two, tau, ctx));
    GR_TMP_CLEAR3(norm, two, nsign, ctx);
}
void house_qr(gr_mat_t R, gr_mat_t A, gr_ctx_t ctx, unsigned block){
    slong m = gr_mat_nrows(A, ctx), n = gr_mat_ncols(A, ctx);
	
    gr_mat_t Vst,Tst, ust;
    gr_mat_init(Vst,m,block,ctx);
    gr_mat_init(Tst,block,block,ctx);
    gr_mat_init(ust, m, 1, ctx);
	
    gr_ptr tau, negtau;
    GR_TMP_INIT2(tau, negtau, ctx);
	//<scratch>
	gr_mat_t wtempst, outerst, W1st, W2st, W3st;
	gr_mat_init(outerst, m, n, ctx);
	gr_mat_init(wtempst, 1, n, ctx);	
	gr_mat_init(W1st, block, n, ctx);
	gr_mat_init(W2st, block, n, ctx);
	gr_mat_init(W3st, m, n, ctx);
	//</scratch>
    STAT(gr_mat_set(R, A, ctx));
    for(unsigned k = 0; k < n; k+=block){
        unsigned p = block;
        if(p > n-k) p = n-k;
        gr_mat_t V, T;
        gr_mat_window_init(V, Vst, 0, 0, m-k, p, ctx);
        gr_mat_window_init(T, Tst, 0, 0, p, p, ctx);
        STAT(gr_mat_zero(V, ctx));
        STAT(gr_mat_zero(T, ctx));

        for(unsigned j = 0; j < p; j++){
            gr_mat_t u;
            gr_mat_window_init(u, ust, 0, 0, m-k-j, 1, ctx); 
            gr_mat_t Vvect, Vright, Vvect2, Tleft, Tblock, Rvect, Rblock1;
            gr_mat_window_init(Vvect, V, j, j, V->r, j+1, ctx); //V[j:,j]
            gr_mat_window_init(Vright, V, 0, 0, V->r, j, ctx); //V[:,:j]
            gr_mat_window_init(Vvect2, V, 0, j, V->r, j+1, ctx); //V[:,j]
            gr_mat_window_init(Rvect, R, k+j, k+j, R->r, k+j+1, ctx); //R[k+j:,k+j]
            gr_mat_window_init(Rblock1, R, k+j, k+j, R->r, k+p, ctx); //R[k+j:,k+j:k+p]
            gr_mat_window_init(Tleft, T, 0, j, j, j+1, ctx); //T[:j,j]
            gr_mat_window_init(Tblock, T, 0, 0, j, j, ctx); //T[:j,:j]
            householder(u, tau, Rvect, ctx); //u, tau = householder(R[k+j:,k+j])
            STAT(gr_mat_set(Vvect, u, ctx)); //V[j:,j] = u
            STAT(gr_set(gr_mat_entry_ptr(T, j, j,ctx), tau,ctx)); //T[j,j] = tau
			STAT(gr_neg(negtau, tau, ctx));
            if(j > 0){ //T[:j,j] = -tau * T[:j,:j] @ (V[:, :j].T @ V[:, j])
                gr_mat_t temp;
                gr_mat_t Vtransp;
                GR_MAT_TMP_INIT_SHALLOW_TRANSPOSE(Vtransp,Vright,ctx);
                gr_mat_init(temp, Vtransp->r, Vvect2->c, ctx);
                STAT(gr_mat_mul(temp, Vtransp, Vvect2, ctx));
                STAT(gr_mat_mul(Tleft, Tblock, temp, ctx));
                STAT(gr_mat_scalar_mul(Tleft, negtau, Tleft,ctx));
                gr_mat_clear(temp, ctx);
                GR_MAT_TMP_CLEAR_SHALLOW_TRANSPOSE(Vtransp, ctx);
            }
            {//R[k+j:, k+j:k+p] -= tau * np.outer(u, u @ R[k+j:, k+j:k+p]) (-= tau * (u @ (u.T @ R[])))
				gr_mat_t uT;
				GR_MAT_TMP_INIT_SHALLOW_TRANSPOSE(uT, u, ctx);
				gr_mat_t wtemp, outer;
				gr_mat_window_init(wtemp, wtempst, 0, 0, uT->r, Rblock1->c, ctx);
				STAT(gr_mat_mul(wtemp, uT, Rblock1, ctx));
				gr_mat_window_init(outer, outerst, 0, 0, u->r, wtemp->c, ctx);
				STAT(gr_mat_mul(outer, u, wtemp, ctx));
				STAT(gr_mat_scalar_mul(outer, negtau, outer, ctx));
				STAT(gr_mat_add(Rblock1, Rblock1, outer, ctx));
				GR_MAT_TMP_CLEAR_SHALLOW_TRANSPOSE(uT, ctx);
				gr_mat_window_clear(wtemp, ctx);
				gr_mat_window_clear(outer, ctx);
			}
            gr_mat_window_clear(Vright, ctx);
            gr_mat_window_clear(Vvect, ctx);
            gr_mat_window_clear(Vvect2, ctx);
            gr_mat_window_clear(Tleft, ctx);
            gr_mat_window_clear(Tblock, ctx);
            gr_mat_window_clear(Rvect, ctx);
            gr_mat_window_clear(Rblock1, ctx);
            gr_mat_window_clear(u, ctx);
        }
		{// R[k:, k+p:] -= V @ (T.T @ (V.T @ R[k:, k+p:]))
			gr_mat_t Rblock3, W1, W2, W3, VT, TT;
			gr_mat_window_init(Rblock3, R, k, k+p, R->r, R->c, ctx);
			GR_MAT_TMP_INIT_SHALLOW_TRANSPOSE(TT, T, ctx);
			GR_MAT_TMP_INIT_SHALLOW_TRANSPOSE(VT, V, ctx);
			gr_mat_window_init(W1, W1st, 0, 0, VT->r, Rblock3->c, ctx);
			STAT(gr_mat_mul(W1, VT, Rblock3, ctx));
			gr_mat_window_init(W2, W2st, 0, 0, TT->r, W1->c, ctx);
			STAT(gr_mat_mul(W2, TT, W1, ctx));
			gr_mat_window_init(W3, W3st, 0, 0, V->r, W2->c, ctx);
			STAT(gr_mat_mul(W3, V, W2, ctx)); 
			STAT(gr_mat_sub(Rblock3, Rblock3, W3, ctx));
			GR_MAT_TMP_CLEAR_SHALLOW_TRANSPOSE(TT,ctx);
			GR_MAT_TMP_CLEAR_SHALLOW_TRANSPOSE(VT,ctx);
			gr_mat_window_clear(Rblock3, ctx);
			gr_mat_window_clear(W1, ctx);
			gr_mat_window_clear(W2, ctx);
			gr_mat_window_clear(W3, ctx);
		}
		gr_mat_window_clear(V, ctx);
		gr_mat_window_clear(T, ctx);
    }
    gr_mat_clear(Vst, ctx);
    gr_mat_clear(Tst, ctx);
    gr_mat_clear(ust, ctx);
    GR_TMP_CLEAR2(tau, negtau, ctx);
}
#endif
