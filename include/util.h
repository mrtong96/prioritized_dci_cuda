/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 *
 * This code implements the method described by Li et al., which can be found at https://arxiv.org/abs/1703.00440
 * This code also builds off of code written by Ke Li.
 */

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// CUDA random
#include <curand.h>
#include <curand_kernel.h>
#include <unistd.h>


#ifndef UTIL_H
#define UTIL_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#define GAUSS_RAND 0
#define UNIFORM_RAND 1

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
        unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

void initializeCUDA(int &devID);

// put in host pointer, get host pointer
void matmul(const cublasOperation_t op_A, const cublasOperation_t op_B,
    const int M, const int N, const int K, const double* const A, const double* const B, double* const C, int &devID);

// put in device pointers. Saves on memcpy operations
void matmul_device(const cublasOperation_t op_A, const cublasOperation_t op_B,
    const int M, const int N, const int K, const double* const A, const double* const B, double* const C, int &devID);

void gen_data(double* const data, const int ambient_dim, const int intrinsic_dim, const int num_points);

// put in device pointers. Saves on memcpy operations
void gen_data_device(double* const data, const int ambient_dim, const int intrinsic_dim, const int num_points);

__device__
double compute_dist(const double* const vec1, const double* const vec2, const int dim);

double gaussrand();

void rng_parallel(double* const vec, const int n, const int rng_type);

// put in device pointers. Saves on memcpy operations
void rng_parallel_device(double* const vec, const int n, const int rng_type);

__global__ void init_curand_state(unsigned int seed, curandState_t* states);

__global__ void gauss_parallel_rng(curandState_t* states, double *vec, const int n);

__global__ void uniform_parallel_rng(curandState_t* states, double *vec, const int n);

void print_matrix(const double* const data, const int num_rows, const int num_cols);

// debugging functions
void print_max_min(double* const data, const int size);
void print_mean_var(double* const data, const int size);

#ifdef __cplusplus
}
#endif

#endif // UTIL_H
