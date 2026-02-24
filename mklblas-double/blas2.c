#include <stdio.h>
#include <stdlib.h>
#include "mkl.h" // Include the MKL header file

/**
 * MKL Matrix-Vector Multiplication Example (dgemv)
 * * Computes y = alpha * A * x + beta * y
 * Matrix A: 3x2 (M=3, N=2)
 * Vector x: 2 elements (N)
 * Vector y: 3 elements (M)
 */
int main() {
    // --- 1. Define Matrix and Vector Dimensions ---
    int M = 3; // Number of rows in A (length of y)
    int N = 2; // Number of columns in A (length of x)
    int lda = M; // Leading dimension of A (for column-major storage, this is M)
    
    // --- 2. Define Scalars for the operation ---
    double alpha = 1.0; // Scaling factor for A*x
    double beta = 0.0;  // Scaling factor for y (setting to 0.0 effectively calculates y = A*x)

    // --- 3. Initialize Input Data ---
    // Matrix A (3x2) - Stored in Column-Major order (MKL default)
    // Conceptually:
    // A = | 1.0  4.0 |
    //     | 2.0  5.0 |
    //     | 3.0  6.0 |
    // Stored in memory as: {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Vector x (length 2)
    // x = {1.0, 2.0}
    double x[] = {1.0, 2.0};
    int incx = 1; // Increment for x (contiguous)

    // Vector y (length 3) - Will store the result
    // y = {0.0, 0.0, 0.0}
    double y[] = {0.0, 0.0, 0.0};
    int incy = 1; // Increment for y (contiguous)

    // --- 4. Call the MKL dgemv function ---
    // CblasNoTrans: Use A as is (not transposed)
    // M: Number of rows in A
    // N: Number of columns in A
    // alpha: The scalar alpha
    // A: Pointer to the matrix A
    // lda: Leading dimension of A
    // x: Pointer to vector x
    // incx: Stride for x
    // beta: The scalar beta
    // y: Pointer to vector y (input/output)
    // incy: Stride for y
    cblas_dgemv(CblasColMajor, CblasNoTrans, 
                M, N, 
                alpha, A, lda, 
                x, incx, 
                beta, y, incy);

    // --- 5. Print the Result ---
    printf("Result y = A*x:\n");
    for (int i = 0; i < M; i++) {
        printf("y[%d] = %lf\n", i, y[i]);
    }
    
    // Expected Result:
    // y[0] = (1.0 * 1.0) + (4.0 * 2.0) = 1.0 + 8.0 = 9.0
    // y[1] = (2.0 * 1.0) + (5.0 * 2.0) = 2.0 + 10.0 = 12.0
    // y[2] = (3.0 * 1.0) + (6.0 * 2.0) = 3.0 + 12.0 = 15.0
    
    return 0;
}
