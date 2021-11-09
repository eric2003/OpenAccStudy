#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
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
	
	fstream * file = new fstream();
	delete file;
	
	MPI_Finalize();
    return 0;
}
