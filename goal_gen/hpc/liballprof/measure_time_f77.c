/*************************************************************************
 * liballprof MPIP Wrapper 
 *
 * Copyright: Indiana University
 * Author: Torsten Hoefler <htor@cs.indiana.edu>
 * 
 *************************************************************************/
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/utsname.h>
#include <mpi.h>

#include "config.h"
#include "allprof.h"
#include "numbers.h"
#include "sync.h"

#define true 1
#define false 0

#ifdef HAVE_NBC
#include <nbc.h>
#endif

#ifdef WRITER_THREAD
#include <pthread.h>
#include <semaphore.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Override three functions to compute the execution time of each rank
 **/
double start_time = 0;

/* parsing >int MPI_Init(int *argc, char ***argv)< */
void F77_FUNC(pmpi_init,PMPI_INIT)(int *argc, int *argv, int *ierr); 
void F77_FUNC(mpi_init,MPI_INIT)(int *argc, int *argv, int *ierr) { 
  F77_FUNC(pmpi_init,PMPI_INIT)(argc, argv, ierr);
  start_time = PMPI_Wtime();
}

/* parsing >int MPI_Finalize(void)< */
void F77_FUNC(pmpi_finalize,PMPI_FINALIZE)(int *ierr); 
void F77_FUNC(mpi_finalize,MPI_FINALIZE)(int *ierr) { 
  printf("[DEBUG] Total runtime: %f\n", PMPI_Wtime() - start_time);
  F77_FUNC(pmpi_finalize,PMPI_FINALIZE)(ierr);
}

/* parsing >int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)< */
void F77_FUNC(pmpi_init_thread,PMPI_INIT_THREAD)(int *argc, int *argv, int *required, int *provided, int *ierr); 
void F77_FUNC(mpi_init_thread,MPI_INIT_THREAD)(int *argc, int *argv, int *required, int *provided, int *ierr) { 
  F77_FUNC(pmpi_init_thread,PMPI_INIT_THREAD)(argc, argv, required, provided, ierr);
  start_time = PMPI_Wtime();
}


#ifdef __cplusplus
}
#endif

