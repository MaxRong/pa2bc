Last name of Student 1:
First name of Student 1:
Email of Student 1:
Last name of Student 2:
First name of Student 2:
Email of Student 2:

See the description of this assignment  for detailed reporting requirements 


Part B

Q2.a List parallel code that uses at most two barrier calls inside the while loop





Q2.b Report parallel time, speedup, and efficiency for  the upper triangular test matrix case when n=4096 and t=1024. 
Use 2 threads and 4  threads (1 thread per core) under blocking mapping, and block cyclic mapping with block size 1 and block size 16.    
Write a short explanation on why one mapping method is significantly faster than or similar to another.











Please indicate if your evaluation is done on CSIL and if yes, list the uptime index of that CSIL machine.  


-----------------------------------------------------------------
Part C

1. Report what code changes you made for blasmm.c. 




2. Conduct a latency and GFLOPS comparison of the above 3 when matrix dimension N varies as 50, 200, 800, and 1600. 
Run the code in one thread and 8 threads on an AMD CPU server of Expanse.
List the latency and GFLOPs of  each method in each setting.  
Explain why when N varies from small to large,  Method 1 with GEMM starts to outperform others. 
