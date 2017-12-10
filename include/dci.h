/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 *
 * This code implements the method described by Li et al., which can be found at https://arxiv.org/abs/1703.00440
 * This code also builds off of code written by Ke Li.
 */

#ifndef DCI_H
#define DCI_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef struct idx_elem {
    double key;  // value of the projection of point onto vector
    int value;  // index of the point


} idx_elem;

// sorting alg we are using
__device__
void mixSort(idx_elem arr[], int n);

typedef struct dci {
    int dim;                // (Ambient) dimensionality of data
    int num_comp_indices;   // Number of composite indices
    int num_simp_indices;   // Number of simple indices in each composite index
    int num_points;
    idx_elem* indices;      // Assuming row-major layout, matrix of size required_num_points x (num_comp_indices*num_simp_indices)
    double* d_proj_vec;     // Assuming row-major layout, matrix of size dim x (num_comp_indices*num_simp_indices)
    double* d_data_proj;    // Device copy of data_proj
    double* data;
    double* d_data;
    int devID;              // To initialize CUDA's matmul, set to 0
} dci;

typedef struct dci_query_config {
    bool blind;
    int num_outer_iterations;
    int max_num_candidates;
} dci_query_config;

void dci_gen_proj_vec(double* const proj_vec, const int dim, const int num_indices);

void dci_init_master(dci* const dci_inst, const int dim, const int num_comp_indices, const int num_simp_indices);
void dci_init_slave(dci* const dci_inst, const int dim, const int num_comp_indices, const int num_simp_indices, const int max_num_points_per_block);

__device__
void insertionSort(idx_elem arr[], int n);

// // Note: the data itself is not kept in the index and must be kept in-place
void dci_add_master(dci* const dci_inst, const int dim, const int num_points, double* const data);

// call *_slave functions to init and add the data to slave instances then run queries
// candidate_points are the indices of the nearest neighbor candidates (size is k*num_blocks)
// candidate_points is the return value
__global__ 
void split_blocks_and_run_queries(const int k, const int num_blocks, dci* const dci_master, 
    double* const candidate_data_proj, const int num_candidate_points, dci* const dci_slave_list, // from old header
    dci_query_config* const query_config, const int num_queries, const double* const query, // variables I added
    double* const query_proj, // this one just have to malloc from the outside
    int * output_candidates, double* const output_candidates_distances, int* const output_num_candidates);

void dci_master_query(dci * const dci_master, const int num_neighbors, dci_query_config* const query_config, 
    const int num_queries, const double* const query, int* final_outputs, double* final_distances);

__device__
void dci_query(dci* const dci_inst, const int dim, const int num_queries, const double* const query,
    const int num_neighbours, const dci_query_config query_config, int* const nearest_neighbours,
    const double* const query_proj,
    double* const nearest_neighbour_dists, int* const num_candidates);

__device__
static int dci_query_single_point(const dci* const dci_inst, const int num_neighbours, const double* const query,
    const double* const query_proj, const dci_query_config query_config, idx_elem* const top_candidates,
    double* const index_priority, int* const left_pos, int* const right_pos, int* const cur_points,
    int* const counts, double* const candidate_dists, int* const all_candidates);

void dci_clear(dci* const dci_inst);

// Clear indices and reset the projection directions
void dci_reset(dci* const dci_inst);

void dci_free(const dci* const dci_inst);

void dci_dump(const dci* const dci_inst);

#ifdef __cplusplus
}
#endif

#endif // DCI_H
