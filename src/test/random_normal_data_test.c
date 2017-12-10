#include<stdio.h>
#include "util.h"
#include <malloc.h>

int main() {

    int ambient_dim = 250;
    int intrinsic_dim = 50;
    int num_points = 10000;

    double * data = (double *)memalign(64, sizeof(double)*num_points*ambient_dim);
    gen_data(data, ambient_dim, intrinsic_dim, num_points);

    print_max_min(data, num_points*ambient_dim);
    print_mean_var(data, num_points*ambient_dim);
}