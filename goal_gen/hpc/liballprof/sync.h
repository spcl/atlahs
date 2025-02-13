#include <mpi.h>
#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif
double sync_tree(MPI_Comm comm);
double sync_lin(MPI_Comm comm);
#ifdef __cplusplus
}
#endif
