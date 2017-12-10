/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 *
 * This code implements the method described by Li et al., which can be found at https://arxiv.org/abs/1703.00440
 * This code also builds off of code written by Ke Li.
 */

#include "util.h"
// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
#include <malloc.h>
#include <math.h>

// generate the random seed
#include <inttypes.h>

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


#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

// taken from 0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp  in the NVIDIA samples
void initializeCUDA(int &devID) {
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess) {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
}    

// Given matrices A, B, output C s.t.
// C = (A/A^T) * (B/B^T)
// after the optional transpose operations, A/A^T is dimension M*K, B/B^T is dimension K*N
// M, N, and K can be thought of as dim(C) = M*N, K is shared dimension.
// Can specify whether or not to take transpose by using op_A/B = CUBLAS_OP_N/CUBLAS_OP_T
// Note this is in row-major format. 
void matmul(const cublasOperation_t op_A, const cublasOperation_t op_B,
    const int M, const int N, const int K, const double* const A, const double* const B, double* const C, int &devID) {
    // convenience variables
    unsigned int mem_size_A = sizeof(double)*M*K;
    unsigned int mem_size_B = sizeof(double)*K*N;
    unsigned int mem_size_C = sizeof(double)*M*N;

    // initialize the CUDA variables
    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    int block_size = 32;  // size 16 has also been used. Think 32 is faster

    // allocate device memory
    double *d_A, *d_B, *d_C;

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(N / threads.x, M / threads.y);

    // CUBLAS version 2.0
    const double alpha = 1.0f;
    const double beta  = 0.0f;
    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    int lda, ldb;
    if(op_A == CUBLAS_OP_N) {
        lda = K;
    } else {
        lda = M;
    }
    if(op_B == CUBLAS_OP_N) {
        ldb = N;
    } else {
        ldb = K;
    }


    cublasDgemm(handle, op_B, op_A, N, M, K, &alpha, d_B, ldb, d_A, lda, &beta, d_C, N);

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    // Destroy the handle
    checkCudaErrors(cublasDestroy(handle));

    // free memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}

// uses device pointers, save on malloc ops
void matmul_device(const cublasOperation_t op_A, const cublasOperation_t op_B,
    const int M, const int N, const int K, const double* const A, const double* const B, double* const C, int &devID) {
    // initialize the CUDA variables
    cudaDeviceProp deviceProp; 

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    int block_size = 32;  // size 16 has also been used. Think 32 is faster

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(N / threads.x, M / threads.y);

    // CUBLAS version 2.0
    const double alpha = 1.0f;
    const double beta  = 0.0f;
    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    int lda, ldb;
    if(op_A == CUBLAS_OP_N) {
        lda = K;
    } else {
        lda = M;
    }
    if(op_B == CUBLAS_OP_N) {
        ldb = N;
    } else {
        ldb = K;
    }

    cublasDgemm(handle, op_B, op_A, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);

    // Destroy the handle
    checkCudaErrors(cublasDestroy(handle));
}

// for debugging
void print_max_min(double* const data, const int size) {
    double max = -10000000.0;
    double min = 10000000.0;
    for(int i = 0; i < size; ++i) {
        if(max < data[i]) {
            max = data[i];
        }
        if(min > data[i]) {
            min = data[i];
        }
    }
    printf("max: %f\n", max);
    printf("min: %f\n", min);
}


// for debugging
void print_mean_var(double* const data, const int size) {
    double m1 = 0.0;
    double m2 = 0.0;
    for(int i = 0; i < size; ++i) {
        m1 += data[i];
        m2 += data[i] * data[i];
    }
    printf("mean: %f\n", m1 / size);
    printf("var: %f\n", (m2 / size) - ((m1 * m1) / (size * size)));
}

void gen_data(double* const data, const int ambient_dim, const int intrinsic_dim, const int num_points) {
    // initialize space
    unsigned int latent_data_size = sizeof(double)*intrinsic_dim*num_points;
    unsigned int transformation_data_size = sizeof(double)*ambient_dim*intrinsic_dim;
    double * latent_data_mat = (double *)memalign(64, sizeof(double)*latent_data_size);
    double * transformation_data_mat = (double *)memalign(64, sizeof(double)*transformation_data_size);

    // populate with random gaussian data
    rng_parallel(latent_data_mat, latent_data_size, UNIFORM_RAND);
    rng_parallel(transformation_data_mat, transformation_data_size, UNIFORM_RAND);

    // matrix multiply
    int devId = 0;
    initializeCUDA(devId);
    // (num_points x intrinsic_dim) x (intrinsic_dim x ambient_dim) = num_points x ambient_dim
    matmul(CUBLAS_OP_N, CUBLAS_OP_N, num_points, ambient_dim, intrinsic_dim, latent_data_mat, transformation_data_mat, data, devId);
}

void gen_data_device(double* const data, const int ambient_dim, const int intrinsic_dim, const int num_points) {
    // initialize space
    unsigned int latent_data_size = sizeof(double)*intrinsic_dim*num_points;
    unsigned int transformation_data_size = sizeof(double)*ambient_dim*intrinsic_dim;

    double* latent_data_mat;
    checkCudaErrors(cudaMalloc((void **) &latent_data_mat, latent_data_size));
    double* transformation_data_mat;
    checkCudaErrors(cudaMalloc((void **) &transformation_data_mat, transformation_data_size));

    // populate with random gaussian data
    rng_parallel_device(latent_data_mat, latent_data_size, UNIFORM_RAND);
    rng_parallel_device(transformation_data_mat, transformation_data_size, UNIFORM_RAND);

    // matrix multiply
    int devId = 0;
    // (num_points x intrinsic_dim) x (intrinsic_dim x ambient_dim) = num_points x ambient_dim
    matmul_device(CUBLAS_OP_N, CUBLAS_OP_N, num_points, ambient_dim, intrinsic_dim, latent_data_mat, transformation_data_mat, data, devId);

    // free memory
    cudaFree(latent_data_mat);
    cudaFree(transformation_data_mat);
}

// copied cpu implementation, probably not worth the GPU memory overhead.
__device__
double compute_dist(const double* const vec1, const double* const vec2, const int dim) {
    int i;
    double sq_dist = 0.0;
    for (i = 0; i < dim; i++) {
        sq_dist += (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);
    }
    return sqrt(sq_dist);
}

// From http://c-faq.com/lib/gaussian.html
// Marsaglia_polar_method
double gaussrand() {
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if(phase == 0) {
        do {
            double U1 = drand48();
            double U2 = drand48();
            
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
            } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

// taken from https://developer.download.nvidia.com/compute/DevZone/docs/html/CUDALibraries/doc/CURAND_Library.pdf
__global__ void init_curand_state(unsigned int seed, curandState_t* states) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}


// gauss random variables in parallel
__global__ void
gauss_parallel_rng(curandState_t* states, double* vec, const int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // Note assumes num_blocks = num_threads
    int chunk_size = (n + blockDim.x * blockDim.x - 1) / (blockDim.x * blockDim.x);
    int index;
    for(int j = 0; j < chunk_size; ++j) {
        index = i*chunk_size+j;
        if(index < n) {
            vec[i*chunk_size+j] = curand_normal_double(&states[i]);
        }
    }
}

// uniform distribution in [-1, 1] in parallel
__global__ void
uniform_parallel_rng(curandState_t* states, double *vec, const int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // Note assumes num_blocks = num_threads
    int chunk_size = (n + blockDim.x * blockDim.x - 1) / (blockDim.x * blockDim.x);
    int index;
    for(int j = 0; j < chunk_size; ++j) {
        index = i*chunk_size+j;
        if(index < n) {
            vec[i*chunk_size+j] = (curand_uniform_double(&states[i]) * 2.0) - 1.0;
        }
    }
}

void rng_parallel(double* const vec, const int n, const int rng_type) {
    int num_blocks = 64;  //  for now using num_blocks blocks, num_blocks threads per block

    // curand initialization
    curandState_t* states;
    long long seed = 0;
    for(int i = 0; i < 4; ++i) {
        seed = (seed << 32) | rand();
    }
    cudaMalloc((void**) &states, num_blocks * num_blocks * sizeof(curandState_t));
    init_curand_state<<<num_blocks, num_blocks>>>(seed, states);

    // allocate device memory
    double *d_vec;
    checkCudaErrors(cudaMalloc((void **) &d_vec, sizeof(double)*n));

    // generate random numbers
    if(rng_type == GAUSS_RAND) {
        gauss_parallel_rng<<<num_blocks, num_blocks>>>(states, d_vec, n);
    } else {
        uniform_parallel_rng<<<num_blocks, num_blocks>>>(states, d_vec, n);
    }

    // copy result to host
    checkCudaErrors(cudaMemcpy(vec, d_vec, sizeof(double)*n, cudaMemcpyDeviceToHost));

    // free memory
    checkCudaErrors(cudaFree(d_vec));
}

// helper functon, assumes vec is device pointer
void rng_parallel_device(double* const vec, const int n, const int rng_type) {
    int num_blocks = 64;  //  for now using num_blocks blocks, num_blocks threads per block

    // curand initialization
    curandState_t* states;
    long long seed = 0;
    for(int i = 0; i < 4; ++i) {
        seed = (seed << 32) | rand();
    }
    cudaMalloc((void**) &states, num_blocks * num_blocks * sizeof(curandState_t));
    init_curand_state<<<num_blocks, num_blocks>>>(seed, states);

    // generate random numbers
    if(rng_type == GAUSS_RAND) {
        gauss_parallel_rng<<<num_blocks, num_blocks>>>(states, vec, n);
    } else {
        uniform_parallel_rng<<<num_blocks, num_blocks>>>(states, vec, n);
    }
}


// Prints the matrix assuming column-major layout
void print_matrix(const double* const data, const int num_rows, const int num_cols) {
    int i, j;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_cols; j++) {
            printf("%.4f\t", data[i+j*num_rows]);
        }
        printf("\n");
    }
}
