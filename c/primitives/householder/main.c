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
    TESTPR("WINDOW MULT");
    gr_mat_t Asub;
    gr_mat_window_init(Asub,A,0,2,0,1,RR);
    gr_mat_t B, C;
    gr_mat_init(B, 2,2, RR);
    gr_mat_init(C, 2,2, RR);
    STAT(gr_mat_set_str(B, "[[5,5],[1,2]]", 0, RR));
    STAT(gr_mat_mul(C, B, A, RR));
    STAT(gr_mat_mul(Asub, B, Asub, RR));
    flint_printf("C: %{gr_mat}\n", C, RR);
    flint_printf("A: %{gr_mat}\n", A, RR);
    TESTPR("QR");
    gr_mat_t Q,R;
    gr_mat_init(Q, 2,2, RR);
    gr_mat_init(R, 2,2, RR);
    STAT(gr_mat_qr(Q, R, A, RR));
    flint_printf("Q: %{gr_mat}\nR: %{gr_mat}\n", Q, RR, R, RR);
	gr_mat_clear(A, RR);
	gr_mat_init(A, 4,4, RR);
	gr_mat_clear(R, RR);
	gr_mat_init(R, 4,4, RR);
	STAT(gr_mat_set_str(A, "[[4, 1, 2, -1], [3, 5, -2, 2], [1, -2, 6, 3], [2, 1, 1, 4]]", 0, RR));
    TESTPR("HOMEGROWN QR");
    house_qr(R, A, RR, 4);
    flint_printf("A: %{gr_mat}\n", A, RR);
    flint_printf("R: %{gr_mat}\n", R, RR);
	
	gr_mat_window_clear(Asub, RR);
	gr_mat_clear(A, RR);
	gr_mat_clear(B, RR);
	gr_mat_clear(C, RR);
	gr_mat_clear(Q, RR);
	gr_mat_clear(R, RR);
	gr_ctx_clear(RR);
    return 0;	
}
