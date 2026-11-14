#include "house_qr.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
bool status = 0;
void print_random_matrix(slong n, flint_bitcnt_t bits)
{
    flint_rand_t state;
    fmpz_mat_t A;

    flint_rand_init(state);
    fmpz_mat_init(A, n, n);
    fmpz_mat_randtest(A, state, bits);
    flint_printf("%{fmpz_mat}\n", A);

    fmpz_mat_clear(A);
    flint_rand_clear(state);
}
int main(int argc, char **argv){
    int opt;
    int dim = 10;
    int bits = 128;
    opterr = 0;
    while((opt = getopt(argc, argv, "n:d:b:")) != -1){
        switch(opt){
            case 'n':
            case 'd':
                dim = atoi(optarg);
                break;
            case 'b':
                bits = atoi(optarg);
                break;
            case '?':
            default:
                printf("Usage: %s -n [dim size] -b [bit length]\n", argv[0]);
                exit(1);
        
        }
    }
    if(optind < argc){
        printf("Unexpected argument: %s\n", argv[optind]);
        exit(1);
    }
    print_random_matrix(dim, bits);
}
