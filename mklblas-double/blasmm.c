#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <mkl.h> // Intel MKL Header
#include <omp.h> // Use OpenMP when possible

// Macro for calculating GFLOPS: 2*N*K*N operations for N x K * K x N = N x N
#define GFLOPS(N, K, time_s) (2.0 * (double)(N) * (double)(K) * (double)(N) / (time_s) / 1e9)


#include <sys/time.h>
double get_time() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec / 1000000.0;
}


// --- Naive C Implementation (Non-BLAS, Column-Major) ---
// Note: This order (I-J-K) is generally cache-unfriendly for column-major data
void naive_matrix_mult(int M, int N, int K, const double *A, const double *B, double *C) {
    // Assuming Column-Major storage: C[row][col] is C[col * M + row]
    
    // C(i, j) = SUM(k) A(i, k) * B(k, j)
#pragma omp parallel for 
    for (int i = 0; i < M; i++) { // Loop over rows of C (and A)
       for (int j = 0; j < N; j++) { // loop over columns of C (and B)
            double sum = 0.0;
            for (int k = 0; k < K; k++) { // Inner loop for the dot product
                // Accessing A[i][k]: A[k * M + i]
                // Accessing B[k][j]: B[j * K + k]
                sum += A[k * M + i] * B[j * K + k];
            }
            // C[i][j]: C[j * M + i]
            C[j * M + i] = sum;
        }
    }
}

void initialize_matrix(double *M, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        M[i] = (double)rand() / (double)RAND_MAX;
    }
}

int test( int N) {
    // --- 1. Define Matrix Dimensions ---
    // Use a large size to see the performance difference clearly
    const int M = N, K = N; 
    const int LDC = N, LDA = N, LDB = N; // Leading dimensions (Col-major)
    const int INCX = 1, INCY = 1;         // Increments for vectors

    // --- 2. Allocate and Initialize Matrices ---
    // Use mkl_malloc for all matrices to ensure fair comparison regarding alignment
    double *A = (double *)mkl_malloc(M * K * sizeof(double), 64);
    double *B = (double *)mkl_malloc(K * N * sizeof(double), 64);
    double *C_dgemm = (double *)mkl_malloc(M * N * sizeof(double), 64);
    double *C_dgemv = (double *)mkl_malloc(M * N * sizeof(double), 64);
    double *C_naive = (double *)mkl_malloc(M * N * sizeof(double), 64);
    
    if (!A || !B || !C_dgemm || !C_dgemv || !C_naive) {
        printf("ERROR: Failed to allocate memory.\n");
        return 1;
    }

    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);
    
    printf("--- Matrix Multiplication Performance Comparison ---\n");
    printf("Matrix size: N=%d x N=%d\n\n", N, N);

    // --- 3. DGEMM (Level 3 BLAS) ---
    memset(C_dgemm, 0, M * N * sizeof(double)); 
    double start_dgemm = get_time();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0, A, LDA,
                B, LDB,
                0.0, C_dgemm, LDC);
    double end_dgemm = get_time();
    double time_dgemm = end_dgemm - start_dgemm;
    double gflops_dgemm = GFLOPS(N, K, time_dgemm);


    // --- 4. DGEMV Loop (Level 2 BLAS) ---
    memset(C_dgemv, 0, M * N * sizeof(double)); 
    double start_dgemv = get_time();

    // Matrix C is of size M*N, A is of size M*K, B is of size K*N
    // Col C1+ ColC2+ ... ColCN= C= A*B = A * ColB1 + A* ColB2+ ... + A*ColBN
    for (int j = 0; j < N; j++) {
        const double *B_col; 
        double *C_col; 
	//Your solution to call cblas_dgemv
	
    }

    double end_dgemv = get_time();
    double time_dgemv = end_dgemv - start_dgemv;
    double gflops_dgemv = GFLOPS(N, K, time_dgemv);
    

    // --- 5. Naive C Loop (Unoptimized) ---
    memset(C_naive, 0, M * N * sizeof(double)); 
    double start_naive = get_time();
    naive_matrix_mult(M, N, K, A, B, C_naive);
    double end_naive = get_time();
    double time_naive = end_naive - start_naive;
    double gflops_naive = GFLOPS(N, K, time_naive);

    // --- 6. Print Results ---
    // printf("| Method   | Time (s) | GFLOPS | Speedup vs. Naive |\n");
    printf("MKL DGEMM     : Time %.6f sec. GFLOPS %.2f.  %.2fx\n", time_dgemm, gflops_dgemm, time_naive / time_dgemm);
    printf("MKL DGEMV Loop: Time %.6f sec. GFLOPS %.2f.  %.2fx\n", time_dgemv, gflops_dgemv, time_naive / time_dgemv);
    printf("Naive 3 loops : Time %.6f sec. GFLOPS %.2f.  1.00x\n", time_naive,  gflops_naive);


    // --- 7. Verification  ---
    // Compare one element to ensure correctness (MKL DGEMM is the ground truth)
    double dgemm_val = C_dgemm[M / 2 * N + N / 2]; // Mid-point
    double dgemv_val = C_dgemv[M / 2 * N + N / 2]; 
    double naive_val = C_naive[M / 2 * N + N / 2];
    if(fabs(dgemm_val - dgemv_val) >0.00001 || fabs(dgemm_val - naive_val) >0.00001)
    	printf("\nError! Unequal mid-points:");
    else
    	printf("\nMid-point verification looks OK:");
    printf(" DGEMM=%.4f, DGEMV=%.4f, Naive=%.4f\n\n", dgemm_val, dgemv_val, naive_val);


    // --- 8. Cleanup ---
    mkl_free(A);
    mkl_free(B);
    mkl_free(C_dgemm);
    mkl_free(C_dgemv);
    mkl_free(C_naive);

}

int main(){

   //mkl_set_num_threads(4); Disabled as it requires compilation to change 
   //When needed, use  shell commands "export MKL_NUM_THREADS=4" and  "export OMP_NUM_THREADS=4" 
   //before running this binary
   
   int th=mkl_get_max_threads();
   printf("Maximum number of threads allowed for MKL: %d\n",th); 
   test(50);
   test(200);
   test(800);
   test(1600);
   return 0;
}
