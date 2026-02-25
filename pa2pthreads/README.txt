Last name of Student 1: Rong
First name of Student 1: Max
Email of Student 1:
Last name of Student 2:
First name of Student 2:
Email of Student 2:

See the description of this assignment for detailed reporting requirements


Part B

Q2.a List parallel code that uses at most two barrier calls inside the while loop

work_block (block mapping):
  blocksize = ceil(n / p)
  first = my_rank * blocksize
  last = min(first + blocksize, n)

  for each iteration k:
      for i = first to last-1:
          mv_compute(i)             // compute y[i] = d[i] + A[i]*x
      barrier_wait()                // wait for everyone to finish y

      for i = first to last-1:
          x[i] = y[i]              // update x from y
      barrier_wait()                // wait before next iteration

work_blockcyclic (block-cyclic mapping):
  for each iteration k:
      for i = 0 to n-1:
          if (i/cyclic_blocksize) % p == my_rank:
              mv_compute(i)
      barrier_wait()

      for i = 0 to n-1:
          if (i/cyclic_blocksize) % p == my_rank:
              x[i] = y[i]
      barrier_wait()

Two barrier calls per iteration in both cases.


Q2.b Performance results for upper triangular, n=4096, t=1024

Sequential baseline (1 thread, block mapping, Test 12): 11.44 sec

  Threads | Mapping            | Time (s) | Speedup | Efficiency
  --------|--------------------|----------|---------|----------
  1       | block              | 11.44    | 1.00    | 1.00
  2       | block              |  8.65    | 1.32    | 0.66
  4       | block              |  4.81    | 2.38    | 0.59
  2       | block-cyclic (r=1) |  6.31    | 1.81    | 0.91
  4       | block-cyclic (r=1) |  3.49    | 3.28    | 0.82
  2       | block-cyclic (r=16)|  7.03    | 1.63    | 0.81
  4       | block-cyclic (r=16)|  3.42    | 3.35    | 0.84

Block mapping is the slowest of the three. The upper triangular matrix
means the last rows have way fewer nonzeros, so whichever thread gets
stuck with those bottom rows finishes way earlier than the others. With
4 threads the speedup is only 2.38x (eff = 0.59), pretty bad.

Block-cyclic does better since it spreads rows out across threads, so
each thread gets a mix of heavy and light rows. r=1 and r=16 both get
around 3.3x speedup at 4 threads. r=16 is marginally faster, probably
because bigger blocks means better cache behavior without hurting
the load balance too much.

All 18 tests passed at 1, 2, and 4 threads.


Evaluation done on CSIL machine: csilvm-04

-----------------------------------------------------------------
Part C

1. Code changes to blasmm.c:

Added pointers for the j-th column of B and C, then called cblas_dgemv
to do C(:,j) = A * B(:,j) inside the loop:

    B_col = &B[j * K];
    C_col = &C_dgemv[j * M];
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                M, K,
                1.0, A, LDA,
                B_col, INCX,
                0.0, C_col, INCY);

Basically doing C = A*B one column at a time with Level 2 BLAS (dgemv).


2. Latency and GFLOPS comparison (run on Expanse, AMD EPYC)

1 Thread results:

  N    | Method         | Time (s)   | GFLOPS | Speedup
-------|----------------|------------|--------|--------
  50   | MKL DGEMM      | 0.042028   |  0.01  |  0.00x
  50   | MKL DGEMV      | 0.000040   |  6.24  |  2.15x
  50   | Naive          | 0.000086   |  2.90  |  1.00x
 200   | MKL DGEMM      | 0.000966   | 16.57  |  3.23x
 200   | MKL DGEMV      | 0.001286   | 12.44  |  2.43x
 200   | Naive          | 0.003123   |  5.12  |  1.00x
 800   | MKL DGEMM      | 0.029261   | 35.00  | 13.38x
 800   | MKL DGEMV      | 0.082539   | 12.41  |  4.74x
 800   | Naive          | 0.391456   |  2.62  |  1.00x
1600   | MKL DGEMM      | 0.216402   | 37.86  | 48.63x
1600   | MKL DGEMV      | 1.924734   |  4.26  |  5.47x
1600   | Naive          | 10.523589  |  0.78  |  1.00x

8 Thread results:

  N    | Method         | Time (s)   | GFLOPS | Speedup
-------|----------------|------------|--------|--------
  50   | MKL DGEMM      | 0.067656   |  0.00  |  0.01x
  50   | MKL DGEMV      | 0.000058   |  4.32  |  7.65x
  50   | Naive          | 0.000443   |  0.56  |  1.00x
 200   | MKL DGEMM      | 0.010839   |  1.48  |  0.42x
 200   | MKL DGEMV      | 0.001783   |  8.97  |  2.56x
 200   | Naive          | 0.004563   |  3.51  |  1.00x
 800   | MKL DGEMM      | 0.025533   | 40.10  |  3.25x
 800   | MKL DGEMV      | 0.095666   | 10.70  |  0.87x
 800   | Naive          | 0.082994   | 12.34  |  1.00x
1600   | MKL DGEMM      | 0.074987   |109.25  | 21.00x
1600   | MKL DGEMV      | 1.778943   |  4.60  |  0.89x
1600   | Naive          | 1.574509   |  5.20  |  1.00x

All verifications passed.

At small N (50, 200), DGEMM is actually slower -- it has startup overhead
that doesn't pay off when there's not much work to do. But once N gets
big enough (800+), DGEMM pulls way ahead. At N=1600 with 8 threads it
hits 109 GFLOPS, 21x faster than naive.

The reason is that DGEMM (Level 3) can tile the matrices and keep small
blocks in L1/L2 cache, reusing data many times. It does n^3 ops on n^2
data so there's lots of reuse potential. DGEMV (Level 2) has to re-read
the whole matrix A for every single column of B, so it can't get the
same cache reuse. The naive loop is even worse since its access pattern
is bad for column-major layout. With 8 threads the naive loop does speed
up (OpenMP parallelizes the outer loop), but DGEMV doesn't benefit much
since the for-loop over columns isn't parallelized.
