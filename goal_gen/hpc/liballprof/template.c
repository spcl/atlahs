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

/* some function definitions for F77 */
void F77_FUNC_(pmpi_type_extent, PMPI_TYPE_EXTENT)(int *datatype, MPI_Aint *extent, int *ierr);
void F77_FUNC_(pmpi_type_size, PMPI_TYPE_SIZE) (int *datatype, int *size, int *ierr);
void F77_FUNC_(pmpi_comm_size, PMPI_COMM_SIZE)(int *comm, int *i, int *ierr);
void F77_FUNC_(pmpi_comm_rank, PMPI_COMM_RANK)(int *comm, int *i, int *ierr );


#ifdef WRITER_THREAD
#define VOLATILE volatile
/* have a second buffer to swap */
static volatile char *buf1, *buf2, 
  *curbuf, /* current buffer base address */
  *bufptr; /* current position on buffer */
static volatile char exitflag=0;
static sem_t threadsem, usersem;
#else
#define VOLATILE 
static char *buf1,
  *curbuf, /* current buffer base address */
  *bufptr; /* position in current buf */
#endif
static char buf_initialized = false;

static int world_rank, world_size;
    
static FILE *fp;
static char mpi_initialized = false;

static void resetbuffer(void *buffer) {
  memset(buffer, '\0', BUFSIZE);
  buf_initialized = true;
}
 
#ifdef WRITER_THREAD
static void *writer_thread(void* arg) {
  /* loops infinitely - until exit notification received */
  while(1) {
    char *tmpbuf;

    /* wait on semaphore to be notified */
    sem_wait(&threadsem);

    /* check if exit flag is set :) */
    if(exitflag) {
      /* write */
      fputs((char*)curbuf, fp);
      sem_post(&usersem);
      break;
    }
    /* swap buffers */
    if(curbuf == buf1) {
      curbuf=buf2;
      bufptr=buf2;
      tmpbuf=(char*)buf1;
    } else {
      curbuf=buf1;
      bufptr=buf1;
      tmpbuf=(char*)buf2;
    }
    
    /* notify user thread */
    sem_post(&usersem);

    /* write buffer to disk */
    fputs(tmpbuf, fp);
    resetbuffer(tmpbuf);
  }
}
#endif

void print_banner(int rank, char *bindings, char *name, int size) {
#ifdef PRINT_BANNER
  if(!rank) {
    char *env = getenv("HTOR_PMPI_FILE_PREFIX");
    if(env == NULL) printf("*** htor's mpiplog in %s - %s bindings, logging %i processes to %s*%s!\n", name, bindings, size, FILE_PREFIX, FILE_SUFFIX);
    else            printf("*** htor's mpiplog in %s - %s bindings, logging %i processes to %s*%s!\n", name, bindings, size, env, FILE_SUFFIX);
  }
#endif
}

/* get \ceil log_base(i) \ceil with integer arithmetic */
int logi(int base, int x) {
  int log=0;
  int y=1;
  while(y <= x) { log++; y*=base; }
  return log;
}

/* pretty print numbers in buffer (add 0's to fill up to max) */
int pprint(char *buf, int len, int x, int max) {
  int log10x=logi(10,x);
  if(x==0) log10x=1; /* log_x(0) is undefined but has a single digit ;) */
  int log10max=logi(10,max);
  int i; for(i=0; i<log10max-log10x; ++i) {
    *buf = '0';
    buf++;
    len--;
  }
  snprintf(buf, len, "%i", x);
  return log10max-log10x;
}


static void mpi_initialize(void) {
#define BUFSZ 1024
  char buf[ BUFSZ ];
  time_t tim;
  char *string;
  double diff;
 
  PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

  char numbuf[1024];
  pprint(numbuf, 1024, world_rank, world_size);   

  /* open file */
  char *env = getenv("HTOR_PMPI_FILE_PREFIX");
  if(env == NULL) snprintf(buf, BUFSZ-1, "%s%s%s", FILE_PREFIX, numbuf, FILE_SUFFIX);
  else            snprintf(buf, BUFSZ-1, "%s%s%s", env, numbuf, FILE_SUFFIX);
  fp = fopen(buf, "w");
 
  /* print greeting */
  snprintf(buf, BUFSZ-1, "# htor's PMPI Tracer %s Output File\n", VERSION);
  fputs(buf, fp);
  
  /* print local time */
  time(&tim);
  string = ctime(&tim);
  snprintf(buf, BUFSZ-1, "# time: %s", string);
  fputs(buf, fp);
  
  /* print hostname */
  snprintf(buf, BUFSZ-1, "# hostname: ");
  fputs(buf, fp);
  gethostname(buf, BUFSZ-1);
  fputs(buf, fp);
  snprintf(buf, BUFSZ-1, ".");
  fputs(buf, fp);
  getdomainname(buf, BUFSZ-1);
  fputs(buf, fp);
  snprintf(buf, BUFSZ-1, "\n");
  fputs(buf, fp);

  /* print uname information */
  struct utsname uname_info;
  uname(&uname_info);
  snprintf(buf, BUFSZ-1, "# uname: %s %s %s %s %s\n", uname_info.sysname, uname_info.nodename, uname_info.release, uname_info.version, uname_info.machine);
  fputs(buf, fp);


  diff = sync_tree(MPI_COMM_WORLD);
  snprintf(buf, BUFSZ-1, "# clockdiff: %lf s (relative to rank 0)\n", diff);
  fputs(buf, fp);
 
  snprintf(buf, BUFSZ - 1, "# Current timestamp: %.0f\n", PMPI_Wtime() * 1e6);
  fputs(buf, fp);

  /* end of greeting -- empty line*/
  snprintf(buf, BUFSZ-1, "\n");
  fputs(buf, fp);

#ifdef WRITER_THREAD
  sem_init(&threadsem, 0, 0);
  sem_init(&usersem, 0, 0);
  
  {
    pthread_attr_t attr;
    pthread_t thread;
    int rc;

    pthread_attr_init(&attr);
    rc = pthread_create(&thread, &attr, writer_thread, (void *)0);
    if(rc) { MPI_Abort(MPI_COMM_WORLD, 55); }
  }

#endif

  mpi_initialized = true;
}


static void writebuf(void* buffer) {
  /* either notify thread or write yourself */
#ifdef WRITER_THREAD
  sem_post(&threadsem);
  /* wait until writer swapped buffers */
  sem_wait(&usersem);
#else
  fputs(buffer, fp);
  resetbuffer(buffer);
  bufptr=buffer;
#endif
}

static void mpi_finalize(void) {
  double diff;

  diff = sync_tree(MPI_COMM_WORLD);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((unsigned long)bufptr-(unsigned long)curbuf), "# Finalize clockdiff: %lf\n", diff);
#ifdef WRITER_THREAD
  /* tell thread to write finalize and finish */
  exitflag = 1;
  sem_post(&threadsem);
  sem_wait(&usersem);
#else
  writebuf(curbuf);
#endif
  resetbuffer((char*)curbuf);
  bufptr=curbuf;

  mpi_initialized = false;
}

static __inline__ void check() {
#ifdef WRITER_THREAD
  /* initialize both buffers */
  if(!buf_initialized) { 
    /* allocate buffers */
    buf1 = (volatile char*)malloc(BUFSIZE); if(buf1 == NULL) printf("malloc error\n");
    buf2 = (volatile char*)malloc(BUFSIZE); if(buf2 == NULL) printf("malloc error\n");
    curbuf=buf1;
    bufptr=buf1;
    resetbuffer((char*)buf1); 
    resetbuffer((char*)buf2); 
  }
#else
  if(!buf_initialized) {
    buf1 = malloc(BUFSIZE); if(buf1 == NULL) printf("malloc error\n");
    curbuf=buf1;
    bufptr=buf1;
    resetbuffer(buf1);
  }
#endif
  if((unsigned long)bufptr > ((unsigned long)curbuf)+BUFSIZE-THRESHOLD) writebuf((char*)curbuf);
}

#define IFDTYPE(DTYPE, dtypenum) \
  if(type == DTYPE) { \
    return snprintf(buffer, length, ":%i", dtypenum); \
  } \

static int printdatatype(MPI_Datatype type, char *buffer, int length) {

  IFDTYPE(MPI_INT, LOG_MPI_INT) else
  IFDTYPE(MPI_INTEGER, LOG_MPI_INTEGER) else 
  IFDTYPE(MPI_CHARACTER, LOG_MPI_CHARACTER) else 
  IFDTYPE(MPI_LONG, LOG_MPI_LONG) else 
  IFDTYPE(MPI_SHORT, LOG_MPI_SHORT) else 
  IFDTYPE(MPI_UNSIGNED, LOG_MPI_UNSIGNED) else 
  IFDTYPE(MPI_UNSIGNED_LONG, LOG_MPI_UNSIGNED_LONG) else 
  IFDTYPE(MPI_UNSIGNED_SHORT, LOG_MPI_UNSIGNED_SHORT) else 
  IFDTYPE(MPI_FLOAT, LOG_MPI_FLOAT) else 
  IFDTYPE(MPI_REAL, LOG_MPI_REAL) else 
  IFDTYPE(MPI_DOUBLE, LOG_MPI_DOUBLE) else 
  IFDTYPE(MPI_DOUBLE_PRECISION, LOG_MPI_DOUBLE_PRECISION) else 
  IFDTYPE(MPI_LONG_DOUBLE, LOG_MPI_LONG_DOUBLE) else 
  IFDTYPE(MPI_BYTE, LOG_MPI_BYTE) else 
  IFDTYPE(MPI_FLOAT_INT, LOG_MPI_FLOAT_INT) else 
  IFDTYPE(MPI_DOUBLE_INT, LOG_MPI_DOUBLE_INT) else 
  IFDTYPE(MPI_LONG_INT, LOG_MPI_LONG_INT) else 
  IFDTYPE(MPI_2INT, LOG_MPI_2INT) else 
  IFDTYPE(MPI_SHORT_INT, LOG_MPI_SHORT_INT) else 
  IFDTYPE(MPI_LONG_DOUBLE_INT, LOG_MPI_LONG_DOUBLE_INT) else 
  IFDTYPE(MPI_LOGICAL, LOG_MPI_LOGICAL) else 
  IFDTYPE(MPI_COMPLEX, LOG_MPI_COMPLEX) else 
  IFDTYPE(MPI_DOUBLE_COMPLEX, LOG_MPI_DOUBLE_COMPLEX) else 
  return snprintf(buffer, length, ":%lu", (unsigned long)type); 
}

#define IFOP(OP, opnum) \
  if(op == OP) { \
    return snprintf(buffer, length, ":%i", opnum); \
  } \

static int printop(MPI_Op op, char *buffer, int length) {

  IFOP(MPI_MIN, LOG_MPI_MIN) else
  IFOP(MPI_MAX, LOG_MPI_MAX) else
  IFOP(MPI_SUM, LOG_MPI_SUM) else
  IFOP(MPI_PROD, LOG_MPI_PROD) else
  IFOP(MPI_LAND, LOG_MPI_LAND) else
  IFOP(MPI_BAND, LOG_MPI_BAND) else
  IFOP(MPI_LOR, LOG_MPI_LOR) else
  IFOP(MPI_BOR, LOG_MPI_BOR) else
  IFOP(MPI_LXOR, LOG_MPI_LXOR) else
  IFOP(MPI_BXOR, LOG_MPI_BXOR) else
  IFOP(MPI_MINLOC, LOG_MPI_MINLOC) else
  IFOP(MPI_MAXLOC, LOG_MPI_MAXLOC) else
  return snprintf(buffer, length, ":%lu", (unsigned long)op); 
}





