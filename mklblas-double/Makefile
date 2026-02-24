#Expanse
#Load Intel compiler first	module load gpu/0.15.4 intel/19.0.5.281
#MKLPATH=/opt/intel/Compiler/11.1/072/mkl
#Also limit Intel MKL with one thread
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1

MKLPATH=${MKLROOT}
MKL_ROOT=${MKLROOT}

CC=icpc
GCC = gcc
CFLAGS = -I$(MKLPATH)/include  -O3 -fopenmp 

MKLFLAGS= -mkl 

LDFLAGS = $(MKLFLAGS) -lm  -lpthread -O3 -fopenmp
#LDFLAGS1 = -std=gnu99 -O3 DNDEBUG -g0 -msse4.2 -masm=intel -lm



all:	blasmm blas2 

blasmm: blasmm.o 
	$(CC) -o $@ $^ $(LDFLAGS)


blas2: blas2.o 
	$(CC) -o $@ $^ $(LDFLAGS)

localrunblasmm:
	./blasmm

runblasmm:
	sbatch -v run-blasmm.sh

localrunblas2:
	./blas2

cpu:
	lscpu

status:
	squeue -u `whoami`

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f blas2 blasmm *.o
