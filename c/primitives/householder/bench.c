#include "house_qr.h"
#include <flint/gr_mat.h>
#include <flint/nfloat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
bool status = 0;
bool quiet = false;
bool flint_qr = false;
bool time_it = false;
void process_args(int argc, char **argv){
    int c= 0;
    while((c = getopt(argc, argv, "fqt")) != -1){
        switch(c){
            case 'q':
                quiet = true;
                break;
            case 'f':
                flint_qr = true;
                break;
            case 't':
                time_it = true;
                break;
            case '?':
                printf("Unknown command -%c\n", optopt);
        }
    }
 
}

double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(int argc, char **argv){
    process_args(argc, argv);
    gr_ctx_t RR;
    nfloat_ctx_init(RR, 64, 0);
    
    gr_mat_t A, Q, R;
    gr_mat_init(A, 0,0, RR);
    
    char* matrix_input = 0;
    size_t capacity = 0;
    ssize_t length = getline(&matrix_input, &capacity, stdin);
    STAT(gr_mat_set_str(A, matrix_input, 1, RR));
    free(matrix_input);
    
    double start = get_time();

    if(flint_qr){
        gr_mat_init(R, A->r, A->r, RR);
        gr_mat_init(Q, A->r, A->c, RR);
        STAT(gr_mat_qr(Q, R, A, RR));
        if(!quiet){
            flint_printf("A: %{gr_mat}\nR: %{gr_mat}\n", A, RR, R, RR); 
        } 
        gr_mat_clear(Q, RR);
    }else{
        gr_mat_init(R, A->r, A->c, RR);
        house_qr(R, A, RR, 64);
        if(!quiet){
            flint_printf("R: %{gr_mat}\n", R, RR);
        }
    }
    double end = get_time();
    if(time_it){
        printf("Time elapsed: %.9f seconds.\n", end - start);
    }
    gr_mat_clear(A, RR);
    gr_mat_clear(R, RR);
}
