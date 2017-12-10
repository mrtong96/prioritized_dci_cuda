#include "dci.h"
#include "util.h"
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <stdlib.h>

// generate the random seed
#include <inttypes.h>

dci test_master;

void small_scale(const int num_points) {
    const int intrinsic_dim = 50;
    const int ambient_dim = 250;
    const int num_query_points = 1; // number of points to query for K-NN
    const int num_comp_indices = 2;
    const int num_simp_indices = 5;
    const int num_neighbours = 4; // k in k-NN
    const int num_outer_iterations = 5000;
    const int max_num_candidates = 10*num_neighbours; // do this so num_neighbors < max_num_candidates

    double* const data = (double *)memalign(64, sizeof(double)*ambient_dim*num_points);
    double* const query = (double *)memalign(64, sizeof(double)*ambient_dim*num_query_points);
    int* const final_outputs = (int *)memalign(64, sizeof(int)*num_neighbours*num_query_points);
    double* const final_distances = (double *)memalign(64, sizeof(double)*num_neighbours*num_query_points);

    printf("small_scale\n");
    printf("init\n");

    // populate with random data
    gen_data(data, ambient_dim, intrinsic_dim, num_points);
    gen_data(query, ambient_dim, intrinsic_dim, num_query_points);
    //rng_parallel(query, ambient_dim*num_query_points, UNIFORM_RAND);

    printf("proj_vec generated\n");

    printf("%f, %f\n", query[0], query[1]);

    dci_init_master(&test_master, ambient_dim, num_comp_indices, num_simp_indices);
    dci_add_master(&test_master, ambient_dim, num_points, data);

    dci_query_config query_config = {false, num_outer_iterations, max_num_candidates};

    clock_t begin=clock();
    dci_master_query(&test_master, num_neighbours, &query_config, num_query_points, query, final_outputs, final_distances);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("=====================\n");
    printf("Query time: %f seconds \n", time_spent);
    printf("=====================\n");


    for(int i = 0; i < num_neighbours; ++i) {
        printf("i: %d, %d, distance: %f\n", i, final_outputs[i], final_distances[i]);
    }

    printf("Reached end.\n");
    printf("done with %d points\n", num_points);

    free(data);
}

int main() {
    srand(time(0));
    small_scale(102400);
    return 0;
}