#include <mpi.h>

#ifdef ENABLE_CUDA
#include "cuda_sub.h"
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_OPENACC
#include "myopenacc.h"
#endif

#include "jacobi.h"

#include <stdio.h>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int numprocs,namelen,rank;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &namelen);
	printf("Hello from %d on %s out of %d\n",(rank+1),processor_name,numprocs);
	
	#ifdef ENABLE_CUDA
    int num_gpus = -1;
    GetCudaDeviceCount( num_gpus );
	#endif
	
	#ifdef ENABLE_OPENACC
	test_open_acc();
	#endif
	Jacobi_Test();
	
	#ifdef ENABLE_OPENMP
	printf("number of host CPUs:\t%d\n", omp_get_num_procs());	
	#pragma omp parallel num_threads(4)
	{
	    int nt = omp_get_thread_num();
		printf("Hello OneFLOW CFD: Hybrid MPI+Cuda+OpenACC+OpenMP! Thread = %d on CPU Rank%d on Machine %s\n", nt, rank + 1, processor_name);
	}
	#endif	
	
	MPI_Finalize();
    return 0;
}
