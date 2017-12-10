/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 *
 * This code implements the method described by Li et al., which can be found at https://arxiv.org/abs/1703.00440
 * This code also builds off of code written by Ke Li.
 */

#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "dci.h"
#include "util.h"

/* CUDA and CUBLAS functions */
#include <helper_functions.h>
#include <helper_cuda.h>

/* CUDA runtime */
#include <cuda_runtime.h>
#include <cublas_v2.h>

__device__
static inline double abs_d(double x) {
    return x > 0 ? x : -x;
}

/* Normalize the input projection vectors. Vectors are normalized along each row. */
__global__ void
normalize_proj_vecs(double* const proj_vec, const int dim, const int num_indices) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    /* Note: Assumes num_blocks = num_threads */
    int chunk_size = (num_indices + blockDim.x * blockDim.x - 1) / (blockDim.x * blockDim.x);
    int vec_index;
    double sq_norm, norm;
    for(int j = 0; j < chunk_size; ++j) {
        vec_index = i*chunk_size+j;
        if(vec_index < num_indices) {
            sq_norm = 0.0;
            for(int k = 0; k < dim; ++k) {
                sq_norm += proj_vec[vec_index*dim + k] * proj_vec[vec_index*dim + k];
            }
            norm = sqrt(sq_norm);
            for(int k = 0; k < dim; ++k) {
                proj_vec[vec_index*dim + k] /= norm;
            }
        }
    }
}


/* Create matrix with proj_vec dim-dimensional normalized gaussian vectors.
   vectors are normalized along each row */
void dci_gen_proj_vec(double* const proj_vec, const int dim, const int num_indices) {
    /* Generate the random indices */
    rng_parallel(proj_vec, dim*num_indices, GAUSS_RAND);

    /* Initialize CUDA memory */
    double *d_proj_vec;
    checkCudaErrors(cudaMalloc((void **) &d_proj_vec, sizeof(double)*dim*num_indices));

    /* Copy memory to device */
    checkCudaErrors(cudaMemcpy(d_proj_vec, proj_vec, sizeof(double)*dim*num_indices, cudaMemcpyHostToDevice));

    /* Normalize */
    int block_size = 32;
    int thread_size = 32;
    normalize_proj_vecs<<<block_size, thread_size>>>(d_proj_vec, dim, num_indices);

    /* Copy memory back to host */
    checkCudaErrors(cudaMemcpy(proj_vec, d_proj_vec, sizeof(double)*dim*num_indices, cudaMemcpyDeviceToHost));

    /* Free memory */
    checkCudaErrors(cudaFree(d_proj_vec));
}


/* Device function to generate projection vectors. */
void dci_gen_proj_vec_device(double* const proj_vec, const int dim, const int num_indices) {
    /* Generate the random indices */
    rng_parallel_device(proj_vec, dim*num_indices, GAUSS_RAND);

    /* Normalize */
    int block_size = 32;
    int thread_size = 32;
    normalize_proj_vecs<<<block_size, thread_size>>>(proj_vec, dim, num_indices);
}

/* Initializes the master DCI data structure.  */
void dci_init_master(dci* const dci_inst, const int dim, const int num_comp_indices, const int num_simp_indices) {
    int num_indices = num_comp_indices*num_simp_indices;

    dci_inst->dim = dim;
    dci_inst->num_comp_indices = num_comp_indices;
    dci_inst->num_simp_indices = num_simp_indices;

    checkCudaErrors(cudaMalloc((void **) &dci_inst->d_proj_vec, sizeof(double)*dim*num_indices));
    dci_gen_proj_vec_device(dci_inst->d_proj_vec, dim, num_indices);

    /* Variables that initialize to default values */
    dci_inst->num_points = 0;
    dci_inst->indices = NULL;
    dci_inst->d_data_proj = NULL;
    dci_inst->data = NULL;
    dci_inst->d_data = NULL;
    dci_inst->devID = 0;
}

/* Initializes the slave DCI data strucutre. Each slave DCI structure represents one block of data in our parallelized
algorithm. */
void dci_init_slave(dci* const dci_inst, const int dim, const int num_comp_indices, const int num_simp_indices,
    const int max_num_points_per_block) {
    int num_indices = num_comp_indices*num_simp_indices;

    dci_inst->dim = dim;
    dci_inst->num_comp_indices = num_comp_indices;
    dci_inst->num_simp_indices = num_simp_indices;

    /* Variables that initialize to default values */
    dci_inst->num_points = 0;
    checkCudaErrors(cudaMalloc((void **)(&dci_inst->indices), sizeof(idx_elem) * max_num_points_per_block * num_indices));
    dci_inst->d_proj_vec = NULL;
    dci_inst->d_data_proj = NULL;
    dci_inst->data = NULL;
    dci_inst->d_data = NULL;
    dci_inst->devID = 0;
}

/* Add data to the master DCI data structure.  */
void dci_add_master(dci* const dci_inst, const int dim, const int num_points, double* const data){
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    checkCudaErrors(cudaMalloc((void **) &dci_inst->d_data_proj, sizeof(double)*num_points*num_indices));

    assert(dim == dci_inst->dim);

    dci_inst->data = data;
    checkCudaErrors(cudaMallocManaged((void **) &dci_inst->d_data, sizeof(double)*num_points*dim));
    checkCudaErrors(cudaMemcpy(
        dci_inst->d_data,
        (void *)dci_inst->data,
        sizeof(double)*num_points*dim, cudaMemcpyHostToDevice));

    dci_inst->num_points = num_points;

    /* Get dot products.
    proj_vec is num_indices x dim, data is num_points x dim
    data_proj is num_indices x num_points */
    matmul_device(CUBLAS_OP_N, CUBLAS_OP_T, num_indices, num_points, dci_inst->dim,
        dci_inst->d_proj_vec, dci_inst->d_data, dci_inst->d_data_proj, dci_inst->devID);
}

/* Copy data in proj_vec to indices */
__global__ void
copy_to_indices(dci* const dci_inst, double* const data_proj, const int num_indices, const int num_points) {
    int i = threadIdx.x;
    int n = num_indices * num_points;
    int chunk_size = (n + blockDim.x * blockDim.x - 1) / (blockDim.x * blockDim.x);
    int index;
    for(int j = 0; j < chunk_size; ++j) {
        index = i*chunk_size+j;
        if(index < n) {
            dci_inst->indices[index].key = data_proj[index];
            dci_inst->indices[index].value = index % num_indices;
        }
    }
}

/* Insertion sort. Taken from StackOverflow */
__device__
void insertionSort(idx_elem arr[], int n) {
   int i, j;
   idx_elem key;
   for (i = 1; i < n; i++) {
       key = arr[i];
       j = i-1;

       /* Move elements of arr[0..i-1], that are
          greater than key, to one position ahead
          of their current position */
       while (j >= 0 && arr[j].key > key.key) {
           arr[j+1] = arr[j];
           j = j-1;
       }
       arr[j+1] = key;
   }
}

/* Modified Quicksort to use "MixSort" below. */
__device__
void quickSort(idx_elem arr[], int n) {
    // arbitrary pivot
    double pivot_key = arr[n/2].key;
    idx_elem swp;
    int low = 0;
    int high = n-1;
    while(low < n || high > 0) {
        while(arr[low].key < pivot_key && low < n) {
            low++;
        }
        while(arr[high].key > pivot_key && high > 0) {
            high--;
        }
        if(low<=high) {
            swp = arr[low];
            arr[low] = arr[high];
            arr[high] = swp;
            low++;
            high--;
        }
        else{
            if(high > 0) {
                mixSort(arr, high+1);
            }
            if(low < n-1) {
                mixSort(&arr[low], n-low);
            }
            return;
        }
    }
}

/* Sorting algorithm. If the number of data points is fewer than 64, then it does
 Insertion Sort. Otherwise, it uses Quick Sort. The reasoning is that if there are
 too few data points, then Quick Sort's overhead may be too large. */
__device__
void mixSort(idx_elem arr[], int n) {
    if(n > 64) {
        quickSort(arr, n);
    } else {
        insertionSort(arr, n);
    }
}

/* Runs a single iteration for k-NN with block size for all queries. Taken from Li et. al's
implementation and slightly modified. */
__global__
void split_blocks_and_run_queries(const int k, const int num_blocks, dci* const dci_master,
    double* const candidate_data_proj, const int num_candidate_points, dci* const dci_slave_list,
    dci_query_config* const query_config, const int num_queries, const double* const query,
    double* const query_proj, // this one just have to malloc from the outside
    int * output_candidates, double* const output_candidates_distances, int* const output_num_candidates) {

    // NOTE: num_blocks must be the same as N in <<N, num_threads>>
    dci* dci_slave_inst = &(dci_slave_list[blockIdx.x]);
    int num_indices = dci_master->num_comp_indices * dci_master->num_simp_indices;
    int num_points_per_block = (num_candidate_points + num_blocks - 1)/num_blocks;

    dci_slave_inst->num_points = MIN(num_points_per_block, num_candidate_points - blockIdx.x*num_points_per_block);
    int candidate_data_proj_col;

    for(int i = 0; i < num_indices; ++i) {
        for(int j = 0; j < dci_slave_inst->num_points; ++j) {
            candidate_data_proj_col = blockIdx.x * num_points_per_block + j;
            dci_slave_inst->indices[i*dci_slave_inst->num_points + j].key = candidate_data_proj[i*num_candidate_points + candidate_data_proj_col];
            dci_slave_inst->indices[i*dci_slave_inst->num_points + j].value = j;
        }
        mixSort(&(dci_slave_inst->indices[i*dci_slave_inst->num_points]), dci_slave_inst->num_points);
    }

    // set data index here
    dci_slave_inst->data = &dci_master->d_data[num_points_per_block * blockIdx.x * dci_slave_inst->dim];

    int num_queries_per_thread = (num_queries + blockDim.x - 1) / blockDim.x;
    int num_queries_in_thread = MIN(num_queries_per_thread, num_queries - threadIdx.x * num_queries_per_thread);
    int query_offset = num_queries_per_thread * threadIdx.x;
    // image output to be matrix of (num_blocks) x (num_queries x k)
    int output_offset = k * num_queries * blockIdx.x + query_offset * k;
    // image output to be matrix of (num_blocks) x (num_queries)
    int candidate_offset = blockIdx.x * num_queries + query_offset;

    dci_query(dci_slave_inst, dci_master->dim, num_queries_in_thread, &query[query_offset * dci_master->dim],
        k, *query_config, &output_candidates[output_offset],
        query_proj,
        &output_candidates_distances[output_offset], &output_num_candidates[candidate_offset]);

    // add the offset since incrementing the index seems to break things downstream
    for(int i = 0; i < k; ++i) {
        output_candidates[i + output_offset] += blockIdx.x * num_points_per_block;
    }
}

/* output_candidates is the number of the points, the output_candidate_map is the map for index i -> point number in full data set */
void update_candidate_mapping(int* const output_candidates, int* const output_candidate_map, const int size, const int iter_num) {
    if(iter_num == 0) {
        for(int i = 0; i < size; ++i) {
            output_candidate_map[i] = output_candidates[i];
        }
    } else {
        int* tmp = (int *)memalign(64, sizeof(int)*size);
        for(int i = 0; i < size; ++i) {
            tmp[i] = output_candidate_map[output_candidates[i]];
        }
        for(int i = 0; i < size; ++i) {
            output_candidate_map[i] = tmp[i];
        }
        free(tmp);
    }
}

void update_dci_master(int* const output_candidates, dci* const dci_master, int size) {
    double* new_data = (double *)memalign(64, sizeof(double)*dci_master->dim*size);
    for(int i = 0; i < size; ++i) {
        int index = output_candidates[i];
        for(int j = 0; j < dci_master->dim; ++j) {
            new_data[i*dci_master->dim + j] = dci_master->data[index*dci_master->dim + j];
        }
    }
    cudaFree(dci_master->d_data);
    cudaFree(dci_master->d_data_proj);
    dci_add_master(dci_master, dci_master->dim, size, new_data);
}

void dci_master_query(dci * const dci_master, const int num_neighbors,
    dci_query_config* const query_config, const int num_queries, const double* const query,
    int * final_outputs, double* final_distances) {

    // hyparameter that can be tuned
    const int len_slave_nums = 3;
    int slave_nums [len_slave_nums] = {128, 64, 1};
    int max_slaves = slave_nums[0];

    // do all the mallocing outside the main loop
    dci* dci_slave_inst_list = (dci*)(malloc(sizeof(dci) * max_slaves));
    dci* device_dci_slave_list;
    checkCudaErrors(cudaMalloc((void **) &device_dci_slave_list, sizeof(dci) * max_slaves));

    dci * dev_master;
    checkCudaErrors(cudaMalloc((void **) &dev_master, sizeof(dci)));
    cudaMemcpy(dev_master, dci_master, sizeof(dci), cudaMemcpyHostToDevice);

    int* output_candidates;
    int* output_candidate_map;
    double* output_candidates_distances;
    int* output_num_candidates;
    cudaMallocManaged((void **)(&output_candidates), sizeof(int)*num_neighbors*max_slaves*num_queries);
    cudaMallocManaged((void **)(&output_candidate_map), sizeof(int)*num_neighbors*max_slaves*num_queries);
    cudaMallocManaged((void **)(&output_candidates_distances), sizeof(double)*num_neighbors*max_slaves*num_queries);
    cudaMallocManaged((void **)(&output_num_candidates), sizeof(int)*max_slaves*num_queries);

    double* d_query;
    cudaMallocManaged((void **)(&d_query), sizeof(double)*dci_master->dim*num_queries);
    checkCudaErrors(cudaMemcpy(d_query, query, sizeof(double)*dci_master->dim*num_queries, cudaMemcpyHostToDevice));

    // calculate query_proj
    int devId = 0;
    int num_indices = dci_master->num_simp_indices*dci_master->num_comp_indices;
    double* d_query_proj;
    cudaMallocManaged((void **)(&d_query_proj), sizeof(double)*num_queries*num_indices);
    matmul_device(CUBLAS_OP_N, CUBLAS_OP_T, num_queries, num_indices, dci_master->dim, d_query, dci_master->d_proj_vec, d_query_proj, devId);

    // copy query config to device pointer
    dci_query_config* d_query_config;
    cudaMalloc((void **) &d_query_config, sizeof(dci_query_config));
    cudaMemcpy(d_query_config, query_config, sizeof(dci_query_config), cudaMemcpyHostToDevice);

    // main loop
    for(int i = 0; i < len_slave_nums; ++i) {
        int num_slaves = slave_nums[i];
        int num_candidate_points = dci_master->num_points;

        for (int j = 0; j < num_slaves; j++) {
            dci_init_slave(&dci_slave_inst_list[j], dci_master->dim, dci_master->num_comp_indices,
                dci_master->num_simp_indices, (num_candidate_points + num_slaves - 1)/num_slaves);
        }
        checkCudaErrors(cudaMemcpy(device_dci_slave_list, dci_slave_inst_list, sizeof(dci) * num_slaves, cudaMemcpyHostToDevice));

        split_blocks_and_run_queries<<<num_slaves, num_queries>>>(num_neighbors, num_slaves, dev_master,
            dci_master->d_data_proj, dci_master->num_points, device_dci_slave_list,
            d_query_config, num_queries, d_query,
            d_query_proj,
            output_candidates, output_candidates_distances, output_num_candidates);
        cudaDeviceSynchronize();

        // update map
        update_candidate_mapping(output_candidates, output_candidate_map, num_neighbors*num_slaves, i);

        // update dci_master
        if(num_slaves != 1) {
            // reconstruct data in dci_master_inst
            update_dci_master(output_candidates, dci_master, num_neighbors*num_slaves);
            cudaMemcpy(dev_master, dci_master, sizeof(dci), cudaMemcpyHostToDevice);
        }
    }

    // when we finally get this working, use this
    checkCudaErrors(cudaMemcpy(final_outputs, output_candidate_map, sizeof(int)*num_neighbors*num_queries, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(final_distances, output_candidates_distances, sizeof(double)*num_neighbors*num_queries, cudaMemcpyDeviceToHost));

    // Need to do a bunch more stuff here, but for now let's make sure it's even able to run up to this point.
    cudaFree(device_dci_slave_list);
    cudaFree(output_candidates);
    cudaFree(output_candidates_distances);
    cudaFree(output_num_candidates);
    cudaFree(d_query_config);
    cudaFree(d_query);
    cudaFree(d_query_proj);
}

__device__
static inline int dci_next_closest_proj(const idx_elem* const index, int* const left_pos, int* const right_pos, const double query_proj, const int num_elems) {
    int cur_pos;
    if (*left_pos == -1 && *right_pos == num_elems) {
        cur_pos = -1;
    } else if (*left_pos == -1) {
        cur_pos = *right_pos;
        ++(*right_pos);
    } else if (*right_pos == num_elems) {
        cur_pos = *left_pos;
        --(*left_pos);
    } else if (index[*right_pos].key - query_proj < query_proj - index[*left_pos].key) {
        cur_pos = *right_pos;
        ++(*right_pos);
    } else {
        cur_pos = *left_pos;
        --(*left_pos);
    }
    return cur_pos;
}

// Returns the index of the element whose key is the largest that is less than the key
// Returns an integer from -1 to num_elems - 1 inclusive
// Could return -1 if all elements are greater or equal to key
__device__
static inline int dci_search_index(const idx_elem* const index, const double key, const int num_elems) {
    int start_pos, end_pos, cur_pos;

    start_pos = -1;
    end_pos = num_elems - 1;
    cur_pos = (start_pos + end_pos + 2) / 2;

    while (start_pos < end_pos) {
        if (index[cur_pos].key < key) {
            start_pos = cur_pos;
        } else {
            end_pos = cur_pos - 1;
        }
        cur_pos = (start_pos + end_pos + 2) / 2;
    }

    return start_pos;
}

// Blind querying does not compute distances or look at the values of indexed vectors
// For blind querying, top_candidates is not used; all_candidates is used to store candidates in the order of retrieval
__device__
static int dci_query_single_point(const dci* const dci_inst, const int num_neighbours, const double* const query,
    const double* const query_proj, const dci_query_config query_config, idx_elem* const top_candidates,
    double* const index_priority, int* const left_pos, int* const right_pos, int* const cur_points, int* const counts,
    double* const candidate_dists, int* const all_candidates) {

    int i, j, k, m, h, top_h;
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    int cur_pos;
    double cur_dist, top_index_priority;
    int num_candidates = 0;
    double last_top_candidate_dist = -1.0;   // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;

    for (i = 0; i < dci_inst->num_comp_indices*dci_inst->num_points; i++) {
        counts[i] = 0;
    }

    for (i = 0; i < dci_inst->num_points; i++) {
        candidate_dists[i] = -2.0;
    }

    for (i = 0; i < num_indices; i++) {
        left_pos[i] = dci_search_index(&(dci_inst->indices[i*(dci_inst->num_points)]), query_proj[i], dci_inst->num_points);
        right_pos[i] = left_pos[i] + 1;
    }
    for (i = 0; i < num_indices; i++) {
        cur_pos = dci_next_closest_proj(&(dci_inst->indices[i*(dci_inst->num_points)]), &(left_pos[i]), &(right_pos[i]), query_proj[i], dci_inst->num_points);
        assert(cur_pos >= 0);    // There should be at least one point in the index
        index_priority[i] = abs_d(dci_inst->indices[cur_pos+i*(dci_inst->num_points)].key - query_proj[i]);
        cur_points[i] = dci_inst->indices[cur_pos+i*(dci_inst->num_points)].value;
    }

    k = 0;
    while (k < dci_inst->num_points*dci_inst->num_simp_indices) {
        for (m = 0; m < dci_inst->num_comp_indices; m++) {
            top_index_priority = DBL_MAX;
            top_h = -1;
            for (h = 0; h < dci_inst->num_simp_indices; h++) {
                if (index_priority[h+m*dci_inst->num_simp_indices] < top_index_priority) {
                    top_index_priority = index_priority[h+m*dci_inst->num_simp_indices];
                    top_h = h;
                }
            }

            if (top_h >= 0) {
                i = top_h+m*dci_inst->num_simp_indices;
                counts[cur_points[i]+m*(dci_inst->num_points)]++;

                if (counts[cur_points[i]+m*(dci_inst->num_points)] == dci_inst->num_simp_indices) {
                    if (candidate_dists[cur_points[i]] == -2.0) {

                        if (query_config.blind) {
                            candidate_dists[cur_points[i]] = -1.0;
                            all_candidates[num_candidates] = cur_points[i];
                        } else {
                            // Compute distance
                            cur_dist = compute_dist(&(dci_inst->data[cur_points[i]*dci_inst->dim]), query, dci_inst->dim);
                            candidate_dists[cur_points[i]] = cur_dist;
                            if (num_candidates < num_neighbours) {
                                top_candidates[num_candidates].key = cur_dist;
                                top_candidates[num_candidates].value = cur_points[i];
                                if (cur_dist > last_top_candidate_dist) {
                                    last_top_candidate_dist = cur_dist;
                                    last_top_candidate = num_candidates;
                                }
                            } else if (cur_dist < last_top_candidate_dist) {
                                top_candidates[last_top_candidate].key = cur_dist;
                                top_candidates[last_top_candidate].value = cur_points[i];
                                last_top_candidate_dist = -1.0;
                                for (j = 0; j < num_neighbours; j++) {
                                    if (top_candidates[j].key > last_top_candidate_dist) {
                                        last_top_candidate_dist = top_candidates[j].key;
                                        last_top_candidate = j;
                                    }
                                }
                            }
                        }
                        num_candidates++;
                    } else {
                        if (!query_config.blind) {
                            cur_dist = candidate_dists[cur_points[i]];
                        }
                    }
                }

                cur_pos = dci_next_closest_proj(&(dci_inst->indices[i*(dci_inst->num_points)]), &(left_pos[i]), &(right_pos[i]), query_proj[i], dci_inst->num_points);

                if (cur_pos >= 0) {
                    index_priority[i] = abs_d(dci_inst->indices[cur_pos+i*(dci_inst->num_points)].key - query_proj[i]);
                    cur_points[i] = dci_inst->indices[cur_pos+i*(dci_inst->num_points)].value;
                } else {
                    index_priority[i] = DBL_MAX;
                    cur_points[i] = -1;
                }
            }
        }
        // Stopping condition
        if (num_candidates >= num_neighbours) {
            if (k + 1 >= query_config.num_outer_iterations*dci_inst->num_simp_indices || num_candidates >= query_config.max_num_candidates) {
                break;
            }
        }
        k++;
    }
    if (!query_config.blind) {
        mixSort(top_candidates, num_neighbours);
    }

    return num_candidates;
}

// If blind querying is used, nearest_neighbours must be of size num_queries * max_possible_num_candidates; otherwise, it must be of size num_queries * num_neighbours
// nearest_neighbour_dists can be NULL when blind querying is used
__device__
void dci_query(dci* const dci_inst, const int dim, const int num_queries, const double* const query,
    const int num_neighbours, const dci_query_config query_config, int* const nearest_neighbours,
    const double* const query_proj,
    double* const nearest_neighbour_dists, int* const num_candidates) {

    int j;
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    int max_possible_num_candidates = MIN(query_config.max_num_candidates, query_config.num_outer_iterations);

    assert(dim == dci_inst->dim);
    assert(num_neighbours > 0);
    assert(num_neighbours <= dci_inst->num_points);

    // intialize all the memory things for dci_query_single_point
    int* counts;
    cudaMalloc((void **)&counts, sizeof(int)*dci_inst->num_comp_indices*dci_inst->num_points);
    double* candidate_dists;
    cudaMalloc((void **)&candidate_dists, sizeof(double)*dci_inst->num_points);
    idx_elem* top_candidates;
    cudaMalloc((void **)&top_candidates, sizeof(idx_elem)*num_neighbours);  // Maintains the top-k candidates
    int* left_pos;
    cudaMalloc((void **)&left_pos, sizeof(int)*num_indices);
    int* right_pos;
    cudaMalloc((void **)&right_pos, sizeof(int)*num_indices);
    double* index_priority;
    cudaMalloc((void **)&index_priority, sizeof(double)*num_indices);
    int* cur_points;
    cudaMalloc((void **)&cur_points, sizeof(int)*num_indices);

    for (j = 0; j < num_queries; j++) {
        int cur_num_candidates, k;
        int* all_candidates;    // Set to &(nearest_neighbours[j*max_possible_num_candidates]) when blind querying is used, and NULL otherwise

        if (query_config.blind) {
            all_candidates = &(nearest_neighbours[j*max_possible_num_candidates]);
        } else {
            all_candidates = NULL;
        }

        cur_num_candidates = dci_query_single_point(dci_inst, num_neighbours, &(query[j*dim]), &(query_proj[j*num_indices]),
            query_config, top_candidates, index_priority, left_pos, right_pos, cur_points, counts, candidate_dists, all_candidates);

        if (!query_config.blind) {
            for (k = 0; k < num_neighbours; k++) {      // cur_num_candidates should be equal to num_neighbours
                nearest_neighbours[k+j*num_neighbours] = top_candidates[k].value;
                nearest_neighbour_dists[k+j*num_neighbours] = top_candidates[k].key;
            }
        }
        if (num_candidates) {
            num_candidates[j] = cur_num_candidates;
        }
    }

    // free the dci_single_point_query
    cudaFree(counts);
    cudaFree(candidate_dists);
    cudaFree(top_candidates);
    cudaFree(left_pos);
    cudaFree(right_pos);
    cudaFree(index_priority);
    cudaFree(cur_points);
}


void dci_clear(dci* const dci_inst) {
    if (dci_inst->indices) {
        free(dci_inst->indices);
        dci_inst->indices = NULL;
    }
    dci_inst->data = NULL;
    dci_inst->num_points = 0;
}

void dci_reset(dci* const dci_inst) {
    dci_clear(dci_inst);
    dci_gen_proj_vec_device(dci_inst->d_proj_vec, dci_inst->dim, dci_inst->num_comp_indices*dci_inst->num_simp_indices);
}

void dci_free(const dci* const dci_inst) {
    if (dci_inst->indices) {
        free(dci_inst->indices);
    }
    if (dci_inst->d_proj_vec) {
        cudaFree(dci_inst->d_proj_vec);
    }
}

void dci_dump(const dci* const dci_inst) {
    int i, j;
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    for (j = 0; j < num_indices; j++) {
        for (i = 0; i < dci_inst->num_points; i++) {
            printf("%f[%d],", dci_inst->indices[i+j*(dci_inst->num_points)].key, dci_inst->indices[i+j*(dci_inst->num_points)].value);
        }
        printf("\n");
    }
}
