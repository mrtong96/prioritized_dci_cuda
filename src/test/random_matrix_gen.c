#include<stdio.h>
#include "util.h"
#include <malloc.h>

int main() {
    int n = 1000000;  // number of points to randomly generate

    // because parallelism can generate ~4e8 gaussian numbers/sec
    double * a = (double *)memalign(64, sizeof(double)*n);
    for(int i = 0; i < 100; ++i) {
        rng_parallel(a, n, GAUSS_RAND);
    }

    // print statistics of this

    double first_moment = 0.0;
    double second_moment = 0.0;

    for(int i = 0; i < n; ++i) {
        first_moment += a[i];
        second_moment += a[i] * a[i];
    }

    printf("%f: %f\n", first_moment, second_moment);

    first_moment /= n;
    second_moment /= n;

    for(int i = 0; i < 10; ++i) {
        printf("%f, ", a[i]);
    }
    printf("\n");

    // 1e-9 for numerical stability
    printf("n: %d\nmean: %f\n var: %f\n",
        n, first_moment, second_moment - (first_moment * first_moment + 1e-9));

}