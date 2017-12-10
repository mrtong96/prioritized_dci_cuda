
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include<stdio.h>
#include "util.h"
#include <malloc.h>

// print an MXN matrix, for debugging
void print_matrix(int M, int N, double* mat) {
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            printf("%f\t", mat[N*i+j]);
        }
        printf("\n");
    }
}

double abs_double(double x) {
    if (x > 0) return x;
    return -x;
}

// sanity check with CPU
// Check C = A * B
// dim(A) = N X M, dim(B) = M X K
double check_mat_mul_cpu(int M, int N, int K, double* const A, double* const B, double* const C) {
    double total_error = 0.0;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < K; ++j) {
            double tmp = 0.0;
            for(int k = 0; k < M; ++k) {
                tmp += A[k + M * i] * B[k * K + j];
            }
            total_error += abs_double(tmp - C[i * K + j]);
        }
    }
    return total_error;
}

void test1() {
    // a is 3x3
    double * a = (double *)memalign(64, sizeof(double)*9);
    // b is 3x3
    double * b = (double *)memalign(64, sizeof(double)*9);
    // c is 3x3
    double * c = (double *)memalign(64, sizeof(double)*9);

    // a = 0-8, b = 0-8
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            a[3*i + j] = 3*i + j;
            b[3*i + j] = 3*i - j;
        }
    }

    printf("Matrix A: \n");
    print_matrix(3,3, a);

    printf("Matrix B: \n");
    print_matrix(3,3, b);
    int devId = 0;
    // A^T B
    // matmul_test(CUBLAS_OP_N, CUBLAS_OP_T, 3, 3, 3, b, 3, a, 3, c, 3, devId); // Note that it does (a^T)(b) , not (a)(b)
    matmul(CUBLAS_OP_T, CUBLAS_OP_N, 3, 3, 3, a, b, c, devId); // Note that it does (a^T)(b) , not (a)(b)
    printf("Matrix C = A^T B: \n");
    print_matrix(3,3, c);
    // should be [[45,36,27],[54,42,30],[63,48,33]]

    // A B^T
    // matmul_test(CUBLAS_OP_T, CUBLAS_OP_N, 3, 3, 3, b, 3, a, 3, c, 3, devId); // Note that it does (a^T)(b) , not (a)(b)
    matmul(CUBLAS_OP_N, CUBLAS_OP_T, 3, 3, 3, a, b, c, devId); // Note that it does (a^T)(b) , not (a)(b)
    printf("Matrix C = A B^T: \n");
    print_matrix(3,3, c);
    // should be [[-5,4,13],[-14,22,58],[-23,40,103]]

    printf("==================\n");
}

void test2() {
    // a is 3x2
    double * a = (double *)memalign(64, sizeof(double)*6);
    // b is 3x2
    double * b = (double *)memalign(64, sizeof(double)*6);
    // c is 3x3
    double * c = (double *)memalign(64, sizeof(double)*9);

    // a = 0-8, b = 0-8
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            a[2*i + j] = 2*i + j;
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            b[2*i + j] = 2*i + j;
        }
    }

    printf("Matrix A: \n");
    print_matrix(3, 2, a);

    printf("Matrix B: \n");
    print_matrix(3, 2, b);
    // A B^T
    int devId = 0;

    //matmul_test(CUBLAS_OP_T, CUBLAS_OP_N, 3, 3, 2, b, 2, a, 2, c, 3, devId);
    matmul(CUBLAS_OP_N, CUBLAS_OP_T, 3, 3, 2, a, b, c, devId);
    printf("Matrix C = A B^T: \n");
    print_matrix(3, 3, c);
    // should be [[1,3,5],[3,13,23],[5,23,41]]

    printf("==================\n");

}

void test3(){
    // a is 1x3
    double * a = (double *)memalign(64, sizeof(double)*3);
    // b is 1x2
    double * b = (double *)memalign(64, sizeof(double)*2);
    // c is 3x2
    double * c = (double *)memalign(64, sizeof(double)*6);

    // a = 0-8, b = 0-8
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 3; j++) {
            a[1*i + j] = 1*i + j;
        }
    }
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 2; j++) {
            b[1*i + j] = 1*i + j;
        }
    }

    printf("Matrix A: \n");
    print_matrix(1, 3, a);

    printf("Matrix B: \n");
    print_matrix(1, 2, b);
    // A^T B
    int devId = 0;

    // matmul_test(CUBLAS_OP_N, CUBLAS_OP_T, 2, 3, 1, b, 2, a, 3, c, 2, devId);
    matmul(CUBLAS_OP_T, CUBLAS_OP_N, 3, 2, 1, a, b, c, devId);
    printf("Matrix C = A^T B: \n");
    print_matrix(3, 2, c);
    // should be [[0,0],[0,1],[0,2]]


    printf("==================\n");
}

void test4(){
    // a is 3x1
    double * a = (double *)memalign(64, sizeof(double)*3);
    // b is 2x1
    double * b = (double *)memalign(64, sizeof(double)*2);
    // c is 3x2
    double * c = (double *)memalign(64, sizeof(double)*6);

    // a = 0-8, b = 0-8
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 1; j++) {
            a[1*i + j] = 3*i + j;
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 1; j++) {
            b[1*i + j] = 1*i + j;
        }
    }

    printf("Matrix A: \n");
    print_matrix(3, 1, a);

    printf("Matrix B: \n");
    print_matrix(2, 1, b);
    // A^T B
    int devId = 0;

    // matmul_test(CUBLAS_OP_T, CUBLAS_OP_N, 2, 3, 1, b, 1, a, 1, c, 2, devId);
    matmul(CUBLAS_OP_N, CUBLAS_OP_T, 3, 2, 1, a, b, c, devId);
    printf("Matrix C = A B^T: \n");
    print_matrix(3, 2, c);    
    // should be [[0,0],[0,3],[0,6]]

    printf("==================\n");
}

void test5(){
    // a is 3x1
    double * a = (double *)memalign(64, sizeof(double)*3);
    // b is 1x2
    double * b = (double *)memalign(64, sizeof(double)*2);
    // c is 3x2
    double * c = (double *)memalign(64, sizeof(double)*6);

    // a = 0-8, b = 0-8
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 1; j++) {
            a[1*i + j] = 3*i + j;
        }
    }
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 2; j++) {
            b[1*i + j] = 1*i + j;
        }
    }

    printf("Matrix A: \n");
    print_matrix(3, 1, a);

    printf("Matrix B: \n");
    print_matrix(1, 2, b);
    // A^T B
    int devId = 0;

    // matmul_test(CUBLAS_OP_N, CUBLAS_OP_N, 2, 3, 1, b, 2, a, 1, c, 2, devId);
    matmul(CUBLAS_OP_N, CUBLAS_OP_N, 3, 2, 1, a, b, c, devId);
    printf("Matrix C = A B: \n");
    print_matrix(3, 2, c);
    // should be [[0,0],[0,3],[0,6]]

    printf("==================\n");
}


// matmul_test(transpose B, transpose A, dim(only in B), dim(only in A), shared_dim (A,B), b, 2nd dim of B, a, 2nd dim of a, c, 2nd dim of C)

int main() {

    test1();
    test2();
    test3();
    test4();
    test5();

    return 0;
}

