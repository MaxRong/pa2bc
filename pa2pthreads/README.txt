Last name of Student 1: Rong
First name of Student 1: Max
Email of Student 1:
Last name of Student 2:
First name of Student 2:
Email of Student 2:

See the description of this assignment  for detailed reporting requirements 


Part B

Q2.a List parallel code that uses at most two barrier calls inside the while loop

For block mapping (work_block):
  blocksize = ceil(matrix_dim / thread_count);
  first_row = my_rank * blocksize;
  last_row = min(first_row + blocksize, matrix_dim);

  for k = 0 to no_iterations - 1:
    for i = first_row to last_row - 1:        // compute y = d + A*x for owned rows
      mv_compute(i);
    pthread_barrier_wait(&mybarrier);          // barrier 1: all y values computed

    for i = first_row to last_row - 1:        // copy y -> x for owned rows
      vector_x[i] = vector_y[i];
    pthread_barrier_wait(&mybarrier);          // barrier 2: all x values updated

For block-cyclic mapping (work_blockcyclic):
  for k = 0 to no_iterations - 1:
    for i = 0 to matrix_dim - 1:              // compute y for owned rows
      if (i / cyclic_blocksize) % thread_count == my_rank:
        mv_compute(i);
    pthread_barrier_wait(&mybarrier);          // barrier 1

    for i = 0 to matrix_dim - 1:              // copy y -> x for owned rows
      if (i / cyclic_blocksize) % thread_count == my_rank:
        vector_x[i] = vector_y[i];
    pthread_barrier_wait(&mybarrier);          // barrier 2


Q2.b Report parallel time, speedup, and efficiency for  the upper triangular test matrix case when n=4096 and t=1024. 
Use 2 threads and 4  threads (1 thread per core) under blocking mapping, and block cyclic mapping with block size 1 and block size 16.    
Write a short explanation on why one mapping method is significantly faster than or similar to another.

Correctness tests (all passed on CSIL csilvm-04):
  cs140barrier_test:       Failed 0 out of 5 tests
  itmv_mult_test_pth (4t): Failed 0 out of 12 tests
  itmv_mult_test_pth (2t): Failed 0 out of 12 tests
  itmv_mult_test_pth (1t): Failed 0 out of 12 tests

Sample latencies from CSIL correctness tests (n=512, t=4096):
  Test 8a (block mapping, regular):          1t: 0.950s, 2t: 0.620s, 4t: 0.453s
  Test 8b (block mapping, upper triangular): 1t: 0.509s, 2t: 0.561s, 4t: 0.434s

(Performance tests for n=4096, t=1024 with Tests 9-14 need to be uncommented 
in itmv_mult_test_pth.c and re-run on CSIL/Expanse. Results to be filled below.)

  Threads | Mapping          | Time (s) | Speedup | Efficiency
  --------|------------------|----------|---------|----------
  1       | sequential       |          | 1.00    | 1.00
  2       | block            |          |         |
  4       | block            |          |         |
  2       | block-cyclic r=1 |          |         |
  4       | block-cyclic r=1 |          |         |
  2       | block-cyclic r=16|          |         |
  4       | block-cyclic r=16|          |         |

Explanation:
For the upper triangular matrix, block mapping causes load imbalance because 
threads owning higher-numbered rows have less computation (fewer non-zero 
elements). Block-cyclic mapping distributes rows in a round-robin fashion, 
which better balances the workload across threads. With block size r=1 
(pure cyclic), the load is most evenly distributed but may suffer from 
poor cache locality. Block size r=16 provides a compromise between load 
balance and cache performance.


Please indicate if your evaluation is done on CSIL and if yes, list the uptime index of that CSIL machine.  

Evaluation done on CSIL: csilvm-04

-----------------------------------------------------------------
Part C

1. Report what code changes you made for blasmm.c. 

In the DGEMV loop (Section 4), I set B_col and C_col to point to the j-th 
column of B and C respectively, then called cblas_dgemv to compute 
C(:,j) = A * B(:,j):

    B_col = &B[j * K];
    C_col = &C_dgemv[j * M];
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                M, K,
                1.0, A, LDA,
                B_col, INCX,
                0.0, C_col, INCY);

This computes matrix-matrix multiplication C = A*B column by column using 
Level 2 BLAS (matrix-vector multiply) instead of Level 3 BLAS (DGEMM).


2. Conduct a latency and GFLOPS comparison of the above 3 when matrix dimension N varies as 50, 200, 800, and 1600. 
Run the code in one thread and 8 threads on an AMD CPU server of Expanse.
List the latency and GFLOPs of  each method in each setting.  
Explain why when N varies from small to large,  Method 1 with GEMM starts to outperform others. 

(To be filled after running blasmm on Expanse)

Explanation: 
DGEMM (Level 3 BLAS) outperforms DGEMV and naive loops for larger matrices 
because it operates on matrix-matrix data, enabling better reuse of data in 
CPU caches (O(n^3) computation on O(n^2) data). DGEMM implementations use 
cache-blocking/tiling to keep sub-matrices in L1/L2 cache, achieving near-peak 
FLOPS. DGEMV (Level 2 BLAS) processes one column at a time, re-reading the 
entire matrix A for each column with O(n^2) work on O(n^2) data, limiting 
cache reuse. The naive loop has the worst cache behavior due to its access 
pattern. For small N, the overhead differences are negligible and all methods 
perform similarly.
