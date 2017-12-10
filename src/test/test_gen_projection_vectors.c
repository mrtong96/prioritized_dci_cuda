#include <stdio.h>
#include "dci.h"
#include <malloc.h>


int main() {
    int num_indices = 100;
    int dim = 200;
    double epsilon = 1e-6; // tolerance of error

    double * a = (double *)memalign(64, sizeof(double)*num_indices*dim);
    dci_gen_proj_vec(a, dim, num_indices);

    for(int i = 0; i < num_indices; ++i) {
        double total = 0.0;
        for(int j = 0; j < dim; ++j) {
            total += a[j + i * dim] * a[j + i * dim];
        }
        double error = abs(total - 1.0);
        if(error > epsilon) {
            printf("error, not normalized");
            return 1;
        }
    }

    return 0;
}