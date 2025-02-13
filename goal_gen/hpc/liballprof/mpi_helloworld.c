#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>


int main(int argc, char **argv) {
	/*  ------ MPI specific ------- */
	int rank; /*  MPI rank */
  int procs; /*  number of mpi procs */
  double i;
  int ret;
  MPI_Request reqs[2];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

  i = (double)rand();
  
  MPI_Isend(&rank, 1, MPI_INT, (rank+1)%procs, 0, MPI_COMM_WORLD, &reqs[0]);
  MPI_Irecv(&rank, 1, MPI_INT, (rank-1+procs)%procs, 0, MPI_COMM_WORLD, &reqs[1]);
  MPI_Waitall(2,reqs,MPI_STATUSES_IGNORE);

  printf("before rank %u: i=%f\n", rank, i);
  ret = MPI_Allreduce(MPI_IN_PLACE, &i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  printf("after rank %u: i=%f\n", rank, i);
      

	printf("Hello from rank %u\n", rank);
	fflush(stdout);

	MPI_Finalize();

	return 0;
}
