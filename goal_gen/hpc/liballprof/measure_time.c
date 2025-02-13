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
#include <assert.h>
#include <errno.h>

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
/* some function definitions for F77 */

/* parsing >int MPI_Init(int *argc, char ***argv)< */
int MPI_Init(int *argc, char ***argv) { 
  int ret;
  ret = PMPI_Init(argc, argv);
  start_time = PMPI_Wtime();
  return ret;
}

/* parsing >int MPI_Finalize(void)< */
int MPI_Finalize(void) { 
  int ret;
  printf("[DEBUG] Total runtime: %f\n", PMPI_Wtime() - start_time);
  ret = PMPI_Finalize();
  return ret;
}

/* parsing >int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)< */
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) { 
  int ret;
  ret = PMPI_Init_thread(argc, argv, required, provided);
  start_time = PMPI_Wtime();
  return ret;
}

#ifdef __cplusplus
}
#endif

