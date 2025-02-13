/*************************************************************************
 * liballprof MPIP Wrapper 
 *
 * Copyright: Indiana University
 * Author: Torsten Hoefler <htor@cs.indiana.edu>
 * 
 *************************************************************************/

/* undef to disable banner printing */
#define PRINT_BANNER 

/* trace file prefix (relative to run directory) and suffix */
#define FILE_PREFIX "/users/sshen/workspace/traces/pmpi-trace-rank-"

#define FILE_SUFFIX ".txt"

/* undef to disable writer thread */
#define WRITER_THREAD
#define BUFSIZE 10485760
#define THRESHOLD 8388608

/* IBM only implements a subset of MPI-2 */
/* #define IBM_BROKEN_MPI */
