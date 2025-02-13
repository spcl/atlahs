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


static double start_time = 0;
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
  // pprint(numbuf, 1024, world_rank, world_size);
  snprintf(numbuf, 1024, "%i", world_rank);

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


  /* diff = sync_tree(MPI_COMM_WORLD); */
  /* snprintf(buf, BUFSZ-1, "# clockdiff: %lf s (relative to rank 0)\n", diff); */
  /* fputs(buf, fp); */
  start_time = PMPI_Wtime();
  snprintf(buf, BUFSZ - 1, "# Current timestamp: %.0f\n", start_time * 1e6);
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

  /* diff = sync_tree(MPI_COMM_WORLD); */

  /* bufptr += snprintf((char*)bufptr, BUFSIZE-((unsigned long)bufptr-(unsigned long)curbuf), "# Finalize clockdiff: %lf\n", diff); */
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






/* parsing >int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)< */
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Send");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Send(buf, count, datatype, dest, tag, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)< */
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Recv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Recv(buf, count, datatype, source, tag, comm, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count)< */
int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_count");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Get_count(status, datatype, count);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)count);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)< */
int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Bsend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Bsend(buf, count, datatype, dest, tag, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)< */
int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ssend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ssend(buf, count, datatype, dest, tag, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)< */
int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rsend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Rsend(buf, count, datatype, dest, tag, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Buffer_attach(void *buffer, int size)< */
int MPI_Buffer_attach(void *buffer, int size) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Buffer_attach");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Buffer_attach(buffer, size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buffer);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Buffer_detach(void *buffer_addr, int *size)< */
int MPI_Buffer_detach(void *buffer_addr, int *size) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Buffer_detach");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Buffer_detach(buffer_addr, size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buffer_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Isend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ibsend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ibsend(buf, count, datatype, dest, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Issend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Issend(buf, count, datatype, dest, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Irsend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Irsend(buf, count, datatype, dest, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Irecv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Wait(MPI_Request *request, MPI_Status *status)< */
int MPI_Wait(MPI_Request *request, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Wait");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Wait(request, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)< */
/* int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {  */
/*   int ret; */
/*   check(); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Test"); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6); */

/*   ret = PMPI_Test(request, flag, status); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6); */

/*   return ret; */
/* } */

/* parsing >int MPI_Request_free(MPI_Request *request)< */
/* int MPI_Request_free(MPI_Request *request) {  */
/*   int ret; */
/*   check(); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Request_free"); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6); */

/*   ret = PMPI_Request_free(request); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6); */

/*   return ret; */
/* } */

/* parsing >int MPI_Waitany(int count, MPI_Request array_of_requests[], int *indx, MPI_Status *status)< */
int MPI_Waitany(int count, MPI_Request array_of_requests[], int *indx, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Waitany");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Waitany(count, array_of_requests, indx, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Testany(int count, MPI_Request array_of_requests[], int *indx, int *flag, MPI_Status *status)< */
int MPI_Testany(int count, MPI_Request array_of_requests[], int *indx, int *flag, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Testany");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Testany(count, array_of_requests, indx, flag, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])< */
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Waitall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Waitall(count, array_of_requests, array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[])< */
/* int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[]) {  */
/*   int ret; */
/*   check(); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Testall"); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6); */

/*   ret = PMPI_Testall(count, array_of_requests, flag, array_of_statuses); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_statuses); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6); */

/*   return ret; */
/* } */

/* parsing >int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])< */
int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Waitsome");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)incount);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outcount);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_indices);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])< */
int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Testsome");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)incount);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outcount);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_indices);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status)< */
/* int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status) {  */
/*   int ret; */
/*   check(); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iprobe"); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6); */

/*   ret = PMPI_Iprobe(source, tag, comm, flag, status); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)source); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag); */
  
/* bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm); */
/* { */
/*   int i; */
/*   PMPI_Comm_rank(comm, &i); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i); */
/*   PMPI_Comm_size(comm, &i); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i); */
/* } */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6); */

/*   return ret; */
/* } */

/* parsing >int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status)< */
/* int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status) {  */
/*   int ret; */
/*   check(); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Probe"); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6); */

/*   ret = PMPI_Probe(source, tag, comm, status); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)source); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag); */
  
/* bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm); */
/* { */
/*   int i; */
/*   PMPI_Comm_rank(comm, &i); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i); */
/*   PMPI_Comm_size(comm, &i); */
/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i); */
/* } */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status); */

/*   bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6); */

/*   return ret; */
/* } */

/* parsing >int MPI_Cancel(MPI_Request *request)< */
int MPI_Cancel(MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cancel");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Cancel(request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Test_cancelled(const MPI_Status *status, int *flag)< */
int MPI_Test_cancelled(const MPI_Status *status, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Test_cancelled");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Test_cancelled(status, flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Send_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Send_init(buf, count, datatype, dest, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Bsend_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Bsend_init(buf, count, datatype, dest, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ssend_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ssend_init(buf, count, datatype, dest, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Rsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Rsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rsend_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Rsend_init(buf, count, datatype, dest, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)< */
int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Recv_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Recv_init(buf, count, datatype, source, tag, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Start(MPI_Request *request)< */
int MPI_Start(MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Start");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Start(request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Startall(int count, MPI_Request array_of_requests[])< */
int MPI_Startall(int count, MPI_Request array_of_requests[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Startall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Startall(count, array_of_requests);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)< */
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Sendrecv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendtag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvtag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status)< */
int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Sendrecv_replace");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendtag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvtag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_contiguous");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_contiguous(count, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_vector");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_vector(count, blocklength, stride, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)stride);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_hvector");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_hvector(count, blocklength, stride, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)stride);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_indexed(int count, const int *array_of_blocklengths, const int *array_of_displacements, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_indexed(int count, const int *array_of_blocklengths, const int *array_of_displacements, MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_indexed");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_indexed(count, array_of_blocklengths, array_of_displacements, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_blocklengths);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_size(MPI_Datatype datatype, int *size)< */
int MPI_Type_size(MPI_Datatype datatype, int *size) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_size(datatype, size);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_lb(MPI_Datatype datatype, MPI_Aint *displacement)< */
int MPI_Type_lb(MPI_Datatype datatype, MPI_Aint *displacement) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_lb");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_lb(datatype, displacement);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displacement);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_ub(MPI_Datatype datatype, MPI_Aint *displacement)< */
int MPI_Type_ub(MPI_Datatype datatype, MPI_Aint *displacement) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_ub");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_ub(datatype, displacement);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displacement);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_commit(MPI_Datatype *datatype)< */
int MPI_Type_commit(MPI_Datatype *datatype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_commit");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_commit(datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_free(MPI_Datatype *datatype)< */
int MPI_Type_free(MPI_Datatype *datatype) { 
  int ret;

  if(!mpi_initialized) {
    // this is weird because boost::mpi seems to call MPI_Type_free
    // after MPI finalize when it is profiled ????
    return MPI_SUCCESS;
  }

  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_free(datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Get_elements(const MPI_Status *status, MPI_Datatype datatype, int *count)< */
int MPI_Get_elements(const MPI_Status *status, MPI_Datatype datatype, int *count) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_elements");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Get_elements(status, datatype, count);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)count);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Pack(const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, int outsize, int *position, MPI_Comm comm)< */
int MPI_Pack(const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, int outsize, int *position, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Pack");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Pack(inbuf, incount, datatype, outbuf, outsize, position, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)incount);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)outsize);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)position);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Unpack(const void *inbuf, int insize, int *position, void *outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm)< */
int MPI_Unpack(const void *inbuf, int insize, int *position, void *outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Unpack");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Unpack(inbuf, insize, position, outbuf, outcount, datatype, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)insize);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)position);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)outcount);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size)< */
int MPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Pack_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Pack_size(incount, datatype, comm, size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)incount);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Barrier(MPI_Comm comm)< */
int MPI_Barrier(MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Barrier");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Barrier(comm);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)< */
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Bcast");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Bcast(buffer, count, datatype, root, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buffer);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)< */
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Gather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm)< */
int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Gatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", recvcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", recvcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", displs[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", displs[i]);
  }
}

  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)< */
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Scatter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)< */
int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Scatterv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", sendcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", sendcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", displs[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", displs[i]);
  }
}

  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)< */
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Allgather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)< */
int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Allgatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", recvcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", recvcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", displs[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", displs[i]);
  }
}

  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)< */
int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Alltoall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Alltoallv(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)< */
int MPI_Alltoallv(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Alltoallv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", sendcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", sendcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", sdispls[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", sdispls[i]);
  }
}

  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", recvcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", recvcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", rdispls[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", rdispls[i]);
  }
}

  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)< */
int MPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Alltoallw");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendtypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvtypes);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
int MPI_Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Exscan");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)< */
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Reduce");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op)< */
int MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Op_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Op_create(user_fn, commute, op);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)user_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)commute);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)op);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Op_free(MPI_Op *op)< */
int MPI_Op_free(MPI_Op *op) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Op_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Op_free(op);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)op);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Allreduce");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Reduce_scatter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Scan");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_size(MPI_Group group, int *size)< */
int MPI_Group_size(MPI_Group group, int *size) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_size(group, size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_rank(MPI_Group group, int *rank)< */
int MPI_Group_rank(MPI_Group group, int *rank) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_rank");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_rank(group, rank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[], MPI_Group group2, int ranks2[])< */
int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[], MPI_Group group2, int ranks2[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_translate_ranks");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_translate_ranks(group1, n, ranks1, group2, ranks2);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranks1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranks2);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_compare(MPI_Group group1, MPI_Group group2, int *result)< */
int MPI_Group_compare(MPI_Group group1, MPI_Group group2, int *result) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_compare");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_compare(group1, group2, result);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_group(MPI_Comm comm, MPI_Group *group)< */
int MPI_Comm_group(MPI_Comm comm, MPI_Group *group) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_group");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_group(comm, group);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)< */
int MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_union");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_union(group1, group2, newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)< */
int MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_intersection");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_intersection(group1, group2, newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)< */
int MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_difference");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_difference(group1, group2, newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup)< */
int MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_incl");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_incl(group, n, ranks, newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranks);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_excl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup)< */
int MPI_Group_excl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_excl");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_excl(group, n, ranks, newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranks);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)< */
int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_range_incl");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_range_incl(group, n, ranges, newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)< */
int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_range_excl");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_range_excl(group, n, ranges, newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Group_free(MPI_Group *group)< */
int MPI_Group_free(MPI_Group *group) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Group_free(group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_size(MPI_Comm comm, int *size)< */
int MPI_Comm_size(MPI_Comm comm, int *size) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_size(comm, size);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_rank(MPI_Comm comm, int *rank)< */
int MPI_Comm_rank(MPI_Comm comm, int *rank) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_rank");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_rank(comm, rank);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result)< */
int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_compare");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_compare(comm1, comm2, result);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm1);
{
  int i;
  PMPI_Comm_rank(comm1, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm1, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm2);
{
  int i;
  PMPI_Comm_rank(comm2, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm2, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)< */
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_dup");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_dup(comm, newcomm);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm)< */
int MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_dup_with_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_dup_with_info(comm, info, newcomm);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)< */
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_create(comm, group, newcomm);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)< */
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_split");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_split(comm, color, key, newcomm);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)color);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_free(MPI_Comm *comm)< */
int MPI_Comm_free(MPI_Comm *comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_free(comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_test_inter(MPI_Comm comm, int *flag)< */
int MPI_Comm_test_inter(MPI_Comm comm, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_test_inter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_test_inter(comm, flag);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_remote_size(MPI_Comm comm, int *size)< */
int MPI_Comm_remote_size(MPI_Comm comm, int *size) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_remote_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_remote_size(comm, size);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group)< */
int MPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_remote_group");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_remote_group(comm, group);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *newintercomm)< */
int MPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *newintercomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Intercomm_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Intercomm_create(local_comm, local_leader, peer_comm, remote_leader, tag, newintercomm);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)local_comm);
{
  int i;
  PMPI_Comm_rank(local_comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(local_comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)local_leader);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)peer_comm);
{
  int i;
  PMPI_Comm_rank(peer_comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(peer_comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)remote_leader);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newintercomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm)< */
int MPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Intercomm_merge");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Intercomm_merge(intercomm, high, newintracomm);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)intercomm);
{
  int i;
  PMPI_Comm_rank(intercomm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(intercomm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)high);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newintracomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn, int *keyval, void *extra_state)< */
int MPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn, int *keyval, void *extra_state) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Keyval_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Keyval_create(copy_fn, delete_fn, keyval, extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)copy_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)delete_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Keyval_free(int *keyval)< */
int MPI_Keyval_free(int *keyval) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Keyval_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Keyval_free(keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val)< */
int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Attr_put");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Attr_put(comm, keyval, attribute_val);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag)< */
int MPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Attr_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Attr_get(comm, keyval, attribute_val, flag);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Attr_delete(MPI_Comm comm, int keyval)< */
int MPI_Attr_delete(MPI_Comm comm, int keyval) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Attr_delete");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Attr_delete(comm, keyval);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Topo_test(MPI_Comm comm, int *status)< */
int MPI_Topo_test(MPI_Comm comm, int *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Topo_test");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Topo_test(comm, status);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart)< */
int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Cart_create(comm_old, ndims, dims, periods, reorder, comm_cart);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_old);
{
  int i;
  PMPI_Comm_rank(comm_old, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm_old, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)dims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)periods);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)reorder);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_cart);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Dims_create(int nnodes, int ndims, int dims[])< */
int MPI_Dims_create(int nnodes, int ndims, int dims[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dims_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Dims_create(nnodes, ndims, dims);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)nnodes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)dims);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Graph_create(MPI_Comm comm_old, int nnodes, const int indx[], const int edges[], int reorder, MPI_Comm *comm_graph)< */
int MPI_Graph_create(MPI_Comm comm_old, int nnodes, const int indx[], const int edges[], int reorder, MPI_Comm *comm_graph) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Graph_create(comm_old, nnodes, indx, edges, reorder, comm_graph);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_old);
{
  int i;
  PMPI_Comm_rank(comm_old, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm_old, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)nnodes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)edges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)reorder);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_graph);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges)< */
int MPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graphdims_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Graphdims_get(comm, nnodes, nedges);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)nnodes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)nedges);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int indx[], int edges[])< */
int MPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int indx[], int edges[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Graph_get(comm, maxindex, maxedges, indx, edges);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)maxindex);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)maxedges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)edges);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Cartdim_get(MPI_Comm comm, int *ndims)< */
int MPI_Cartdim_get(MPI_Comm comm, int *ndims) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cartdim_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Cartdim_get(comm, ndims);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ndims);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[])< */
int MPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Cart_get(comm, maxdims, dims, periods, coords);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)maxdims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)dims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)periods);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)coords);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Cart_rank(MPI_Comm comm, const int coords[], int *rank)< */
int MPI_Cart_rank(MPI_Comm comm, const int coords[], int *rank) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_rank");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Cart_rank(comm, coords, rank);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)coords);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[])< */
int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_coords");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Cart_coords(comm, rank, maxdims, coords);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)maxdims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)coords);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors)< */
int MPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_neighbors_count");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Graph_neighbors_count(comm, rank, nneighbors);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)nneighbors);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int neighbors[])< */
int MPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int neighbors[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_neighbors");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Graph_neighbors(comm, rank, maxneighbors, neighbors);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)maxneighbors);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)neighbors);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest)< */
int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_shift");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Cart_shift(comm, direction, disp, rank_source, rank_dest);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)direction);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank_source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank_dest);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm)< */
int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_sub");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Cart_sub(comm, remain_dims, newcomm);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)remain_dims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Cart_map(MPI_Comm comm, int ndims, const int dims[], const int periods[], int *newrank)< */
int MPI_Cart_map(MPI_Comm comm, int ndims, const int dims[], const int periods[], int *newrank) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_map");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Cart_map(comm, ndims, dims, periods, newrank);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)dims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)periods);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newrank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Graph_map(MPI_Comm comm, int nnodes, const int indx[], const int edges[], int *newrank)< */
int MPI_Graph_map(MPI_Comm comm, int nnodes, const int indx[], const int edges[], int *newrank) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_map");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Graph_map(comm, nnodes, indx, edges, newrank);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)nnodes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)edges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newrank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Get_processor_name(char *name, int *resultlen)< */
int MPI_Get_processor_name(char *name, int *resultlen) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_processor_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Get_processor_name(name, resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Get_version(int *version, int *subversion)< */
int MPI_Get_version(int *version, int *subversion) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_version");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Get_version(version, subversion);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)version);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)subversion);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Get_library_version(char *version, int *resultlen)< */
int MPI_Get_library_version(char *version, int *resultlen) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_library_version");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Get_library_version(version, resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)version);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler)< */
int MPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Errhandler_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Errhandler_create(function, errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)function);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler)< */
int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Errhandler_set");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Errhandler_set(comm, errhandler);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler)< */
int MPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Errhandler_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Errhandler_get(comm, errhandler);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Errhandler_free(MPI_Errhandler *errhandler)< */
int MPI_Errhandler_free(MPI_Errhandler *errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Errhandler_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Errhandler_free(errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Error_string(int errorcode, char *string, int *resultlen)< */
int MPI_Error_string(int errorcode, char *string, int *resultlen) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Error_string");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Error_string(errorcode, string, resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)errorcode);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)string);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Error_class(int errorcode, int *errorclass)< */
int MPI_Error_class(int errorcode, int *errorclass) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Error_class");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Error_class(errorcode, errorclass);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)errorcode);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errorclass);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >double MPI_Wtime(void)< */
double MPI_Wtime(void) { 
  double ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Wtime");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Wtime();


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >double MPI_Wtick(void)< */
double MPI_Wtick(void) { 
  double ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Wtick");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Wtick();


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Init(int *argc, char ***argv)< */
int MPI_Init(int *argc, char ***argv) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-");

  ret = PMPI_Init(argc, argv);


  {
    if(!mpi_initialized) mpi_initialize();
 print_banner(world_rank, "C", "MPI_Init", world_size);
  }
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argc);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argv);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Finalize(void)< */
int MPI_Finalize(void) { 
  int ret;
  mpi_finalize();
  double end_time = PMPI_Wtime();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Finalize");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", end_time * 1e6);
  
  ret = PMPI_Finalize();

  
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-\n");
  printf("[DEBUG] Total runtime: %f\n", end_time - start_time);
  fputs((char*)curbuf, fp);; fclose(fp);  return ret;
}

/* parsing >int MPI_Initialized(int *flag)< */
int MPI_Initialized(int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Initialized");
  if (mpi_initialized)
    bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);
  else
    bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-");

  ret = PMPI_Initialized(flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  if (mpi_initialized)
    bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);
  else
    bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-\n");

  return ret;
}

/* parsing >int MPI_Abort(MPI_Comm comm, int errorcode)< */
int MPI_Abort(MPI_Comm comm, int errorcode) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Abort");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Abort(comm, errorcode);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_DUP_FN(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag)< */
int MPI_DUP_FN(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_DUP_FN");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_DUP_FN(oldcomm, keyval, extra_state, attribute_val_in, attribute_val_out, flag);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)oldcomm);
{
  int i;
  PMPI_Comm_rank(oldcomm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(oldcomm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val_in);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val_out);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Close_port(const char *port_name)< */
int MPI_Close_port(const char *port_name) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Close_port");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Close_port(port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_accept(const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm)< */
int MPI_Comm_accept(const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_accept");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_accept(port_name, info, root, comm, newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_connect(const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm)< */
int MPI_Comm_connect(const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_connect");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_connect(port_name, info, root, comm, newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_disconnect(MPI_Comm *comm)< */
int MPI_Comm_disconnect(MPI_Comm *comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_disconnect");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_disconnect(comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_get_parent(MPI_Comm *parent)< */
int MPI_Comm_get_parent(MPI_Comm *parent) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_parent");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_get_parent(parent);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)parent);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_join(int fd, MPI_Comm *intercomm)< */
int MPI_Comm_join(int fd, MPI_Comm *intercomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_join");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_join(fd, intercomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)fd);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)intercomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_spawn(const char *command, char *argv[], int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[])< */
int MPI_Comm_spawn(const char *command, char *argv[], int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_spawn");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_spawn(command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)command);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argv);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)maxprocs);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)intercomm);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_errcodes);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_spawn_multiple(int count, char *array_of_commands[], char **array_of_argv[], const int array_of_maxprocs[], const MPI_Info array_of_info[], int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[])< */
int MPI_Comm_spawn_multiple(int count, char *array_of_commands[], char **array_of_argv[], const int array_of_maxprocs[], const MPI_Info array_of_info[], int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_spawn_multiple");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_spawn_multiple(count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_commands);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_argv);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_maxprocs);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)intercomm);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_errcodes);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Lookup_name(const char *service_name, MPI_Info info, char *port_name)< */
int MPI_Lookup_name(const char *service_name, MPI_Info info, char *port_name) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Lookup_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Lookup_name(service_name, info, port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)service_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Open_port(MPI_Info info, char *port_name)< */
int MPI_Open_port(MPI_Info info, char *port_name) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Open_port");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Open_port(info, port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Publish_name(const char *service_name, MPI_Info info, const char *port_name)< */
int MPI_Publish_name(const char *service_name, MPI_Info info, const char *port_name) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Publish_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Publish_name(service_name, info, port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)service_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Unpublish_name(const char *service_name, MPI_Info info, const char *port_name)< */
int MPI_Unpublish_name(const char *service_name, MPI_Info info, const char *port_name) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Unpublish_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Unpublish_name(service_name, info, port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)service_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_set_info(MPI_Comm comm, MPI_Info info)< */
int MPI_Comm_set_info(MPI_Comm comm, MPI_Info info) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_set_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_set_info(comm, info);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_get_info(MPI_Comm comm, MPI_Info *info)< */
int MPI_Comm_get_info(MPI_Comm comm, MPI_Info *info) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_get_info(comm, info);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)< */
int MPI_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Accumulate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Accumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)origin_count);
  
bufptr += printdatatype(origin_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (origin_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(origin_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_count);
  
bufptr += printdatatype(target_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (target_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(target_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)< */
int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)origin_count);
  
bufptr += printdatatype(origin_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (origin_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(origin_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_count);
  
bufptr += printdatatype(target_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (target_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(target_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)< */
int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Put");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)origin_count);
  
bufptr += printdatatype(origin_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (origin_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(origin_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_count);
  
bufptr += printdatatype(target_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (target_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(target_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_complete(MPI_Win win)< */
int MPI_Win_complete(MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_complete");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_complete(win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win)< */
int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_create(base, size, disp_unit, info, comm, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)base);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)disp_unit);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_fence(int assert, MPI_Win win)< */
int MPI_Win_fence(int assert, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_fence");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_fence(assert, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_free(MPI_Win *win)< */
int MPI_Win_free(MPI_Win *win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_free(win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_get_group(MPI_Win win, MPI_Group *group)< */
int MPI_Win_get_group(MPI_Win win, MPI_Group *group) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_group");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_get_group(win, group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win)< */
int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_lock");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_lock(lock_type, rank, assert, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)lock_type);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_post(MPI_Group group, int assert, MPI_Win win)< */
int MPI_Win_post(MPI_Group group, int assert, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_post");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_post(group, assert, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_start(MPI_Group group, int assert, MPI_Win win)< */
int MPI_Win_start(MPI_Group group, int assert, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_start");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_start(group, assert, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_test(MPI_Win win, int *flag)< */
int MPI_Win_test(MPI_Win win, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_test");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_test(win, flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_unlock(int rank, MPI_Win win)< */
int MPI_Win_unlock(int rank, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_unlock");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_unlock(rank, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_wait(MPI_Win win)< */
int MPI_Win_wait(MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_wait");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_wait(win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)< */
int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_allocate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_allocate(size, disp_unit, info, comm, baseptr, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)disp_unit);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)baseptr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)< */
int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_allocate_shared");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_allocate_shared(size, disp_unit, info, comm, baseptr, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)disp_unit);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)baseptr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr)< */
int MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_shared_query");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_shared_query(win, rank, size, disp_unit, baseptr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)disp_unit);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)baseptr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win)< */
int MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_create_dynamic");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_create_dynamic(info, comm, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size)< */
int MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_attach");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_attach(win, base, size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)base);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_detach(MPI_Win win, const void *base)< */
int MPI_Win_detach(MPI_Win win, const void *base) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_detach");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_detach(win, base);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)base);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_get_info(MPI_Win win, MPI_Info *info_used)< */
int MPI_Win_get_info(MPI_Win win, MPI_Info *info_used) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_get_info(win, info_used);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info_used);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_set_info(MPI_Win win, MPI_Info info)< */
int MPI_Win_set_info(MPI_Win win, MPI_Info info) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_set_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_set_info(win, info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Get_accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, void *result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)< */
int MPI_Get_accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, void *result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_accumulate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Get_accumulate(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)origin_count);
  
bufptr += printdatatype(origin_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (origin_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(origin_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)result_count);
  
bufptr += printdatatype(result_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (result_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(result_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_count);
  
bufptr += printdatatype(target_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (target_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(target_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Fetch_and_op(const void *origin_addr, void *result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win)< */
int MPI_Fetch_and_op(const void *origin_addr, void *result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Fetch_and_op");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Fetch_and_op(origin_addr, result_addr, datatype, target_rank, target_disp, op, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result_addr);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Compare_and_swap(const void *origin_addr, const void *compare_addr, void *result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Win win)< */
int MPI_Compare_and_swap(const void *origin_addr, const void *compare_addr, void *result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Compare_and_swap");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)compare_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result_addr);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Rput(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request *request)< */
int MPI_Rput(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rput");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Rput(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)origin_count);
  
bufptr += printdatatype(origin_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (origin_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(origin_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_count);
  
bufptr += printdatatype(target_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (target_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(target_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Rget(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request *request)< */
int MPI_Rget(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rget");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Rget(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)origin_count);
  
bufptr += printdatatype(origin_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (origin_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(origin_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_count);
  
bufptr += printdatatype(target_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (target_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(target_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Raccumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request *request)< */
int MPI_Raccumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Raccumulate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Raccumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)origin_count);
  
bufptr += printdatatype(origin_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (origin_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(origin_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_count);
  
bufptr += printdatatype(target_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (target_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(target_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Rget_accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, void *result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request *request)< */
int MPI_Rget_accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, void *result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rget_accumulate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Rget_accumulate(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)origin_count);
  
bufptr += printdatatype(origin_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (origin_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(origin_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)result_count);
  
bufptr += printdatatype(result_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (result_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(result_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)target_count);
  
bufptr += printdatatype(target_datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (target_datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(target_datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_lock_all(int assert, MPI_Win win)< */
int MPI_Win_lock_all(int assert, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_lock_all");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_lock_all(assert, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_unlock_all(MPI_Win win)< */
int MPI_Win_unlock_all(MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_unlock_all");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_unlock_all(win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_flush(int rank, MPI_Win win)< */
int MPI_Win_flush(int rank, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_flush");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_flush(rank, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_flush_all(MPI_Win win)< */
int MPI_Win_flush_all(MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_flush_all");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_flush_all(win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_flush_local(int rank, MPI_Win win)< */
int MPI_Win_flush_local(int rank, MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_flush_local");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_flush_local(rank, win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_flush_local_all(MPI_Win win)< */
int MPI_Win_flush_local_all(MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_flush_local_all");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_flush_local_all(win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_sync(MPI_Win win)< */
int MPI_Win_sync(MPI_Win win) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_sync");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_sync(win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Add_error_class(int *errorclass)< */
int MPI_Add_error_class(int *errorclass) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Add_error_class");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Add_error_class(errorclass);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errorclass);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Add_error_code(int errorclass, int *errorcode)< */
int MPI_Add_error_code(int errorclass, int *errorcode) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Add_error_code");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Add_error_code(errorclass, errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)errorclass);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Add_error_string(int errorcode, const char *string)< */
int MPI_Add_error_string(int errorcode, const char *string) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Add_error_string");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Add_error_string(errorcode, string);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)errorcode);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)string);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_call_errhandler(MPI_Comm comm, int errorcode)< */
int MPI_Comm_call_errhandler(MPI_Comm comm, int errorcode) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_call_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_call_errhandler(comm, errorcode);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, int *comm_keyval, void *extra_state)< */
int MPI_Comm_create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, int *comm_keyval, void *extra_state) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_create_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_create_keyval(comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_copy_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_delete_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval)< */
int MPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_delete_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_delete_attr(comm, comm_keyval);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)comm_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_free_keyval(int *comm_keyval)< */
int MPI_Comm_free_keyval(int *comm_keyval) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_free_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_free_keyval(comm_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag)< */
int MPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_get_attr(comm, comm_keyval, attribute_val, flag);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)comm_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen)< */
int MPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_get_name(comm, comm_name, resultlen);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val)< */
int MPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_set_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_set_attr(comm, comm_keyval, attribute_val);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)comm_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_set_name(MPI_Comm comm, const char *comm_name)< */
int MPI_Comm_set_name(MPI_Comm comm, const char *comm_name) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_set_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_set_name(comm, comm_name);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_File_call_errhandler(MPI_File fh, int errorcode)< */
int MPI_File_call_errhandler(MPI_File fh, int errorcode) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_File_call_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_File_call_errhandler(fh, errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)fh);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Grequest_complete(MPI_Request request)< */
int MPI_Grequest_complete(MPI_Request request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Grequest_complete");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Grequest_complete(request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Grequest_start(MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn, void *extra_state, MPI_Request *request)< */
int MPI_Grequest_start(MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn, void *extra_state, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Grequest_start");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Grequest_start(query_fn, free_fn, cancel_fn, extra_state, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)query_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)free_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)cancel_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)< */
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Init_thread");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-");

  ret = PMPI_Init_thread(argc, argv, required, provided);


  {
    if(!mpi_initialized) mpi_initialize();
 print_banner(world_rank, "C", "MPI_Init_thread", world_size);
  }
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argc);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argv);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)required);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)provided);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Is_thread_main(int *flag)< */
int MPI_Is_thread_main(int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Is_thread_main");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Is_thread_main(flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Query_thread(int *provided)< */
int MPI_Query_thread(int *provided) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Query_thread");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Query_thread(provided);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)provided);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Status_set_cancelled(MPI_Status *status, int flag)< */
int MPI_Status_set_cancelled(MPI_Status *status, int flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Status_set_cancelled");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Status_set_cancelled(status, flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count)< */
int MPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Status_set_elements");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Status_set_elements(status, datatype, count);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, int *type_keyval, void *extra_state)< */
int MPI_Type_create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, int *type_keyval, void *extra_state) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_keyval(type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_copy_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_delete_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_delete_attr(MPI_Datatype datatype, int type_keyval)< */
int MPI_Type_delete_attr(MPI_Datatype datatype, int type_keyval) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_delete_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_delete_attr(datatype, type_keyval);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)type_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_dup(MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_dup(MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_dup");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_dup(oldtype, newtype);

  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_free_keyval(int *type_keyval)< */
int MPI_Type_free_keyval(int *type_keyval) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_free_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_free_keyval(type_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_get_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val, int *flag)< */
int MPI_Type_get_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_get_attr(datatype, type_keyval, attribute_val, flag);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)type_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[])< */
int MPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_contents");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_get_contents(datatype, max_integers, max_addresses, max_datatypes, array_of_integers, array_of_addresses, array_of_datatypes);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)max_integers);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)max_addresses);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)max_datatypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_integers);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_addresses);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_datatypes);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_get_envelope(MPI_Datatype datatype, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner)< */
int MPI_Type_get_envelope(MPI_Datatype datatype, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_envelope");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_get_envelope(datatype, num_integers, num_addresses, num_datatypes, combiner);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)num_integers);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)num_addresses);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)num_datatypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)combiner);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_get_name(MPI_Datatype datatype, char *type_name, int *resultlen)< */
int MPI_Type_get_name(MPI_Datatype datatype, char *type_name, int *resultlen) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_get_name(datatype, type_name, resultlen);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_set_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val)< */
int MPI_Type_set_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_set_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_set_attr(datatype, type_keyval, attribute_val);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)type_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_set_name(MPI_Datatype datatype, const char *type_name)< */
int MPI_Type_set_name(MPI_Datatype datatype, const char *type_name) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_set_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_set_name(datatype, type_name);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_match_size(int typeclass, int size, MPI_Datatype *datatype)< */
int MPI_Type_match_size(int typeclass, int size, MPI_Datatype *datatype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_match_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_match_size(typeclass, size, datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)typeclass);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_call_errhandler(MPI_Win win, int errorcode)< */
int MPI_Win_call_errhandler(MPI_Win win, int errorcode) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_call_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_call_errhandler(win, errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, int *win_keyval, void *extra_state)< */
int MPI_Win_create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, int *win_keyval, void *extra_state) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_create_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_create_keyval(win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_copy_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_delete_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_delete_attr(MPI_Win win, int win_keyval)< */
int MPI_Win_delete_attr(MPI_Win win, int win_keyval) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_delete_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_delete_attr(win, win_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)win_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_free_keyval(int *win_keyval)< */
int MPI_Win_free_keyval(int *win_keyval) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_free_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_free_keyval(win_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag)< */
int MPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_get_attr(win, win_keyval, attribute_val, flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)win_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_get_name(MPI_Win win, char *win_name, int *resultlen)< */
int MPI_Win_get_name(MPI_Win win, char *win_name, int *resultlen) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_get_name(win, win_name, resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val)< */
int MPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_set_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_set_attr(win, win_keyval, attribute_val);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)win_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_set_name(MPI_Win win, const char *win_name)< */
int MPI_Win_set_name(MPI_Win win, const char *win_name) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_set_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_set_name(win, win_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr)< */
int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Alloc_mem");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Alloc_mem(size, info, baseptr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)baseptr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_create_errhandler(MPI_Comm_errhandler_function *comm_errhandler_fn, MPI_Errhandler *errhandler)< */
int MPI_Comm_create_errhandler(MPI_Comm_errhandler_function *comm_errhandler_fn, MPI_Errhandler *errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_create_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_create_errhandler(comm_errhandler_fn, errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_errhandler_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler)< */
int MPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_get_errhandler(comm, errhandler);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler)< */
int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_set_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_set_errhandler(comm, errhandler);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_File_create_errhandler(MPI_File_errhandler_function *file_errhandler_fn, MPI_Errhandler *errhandler)< */
int MPI_File_create_errhandler(MPI_File_errhandler_function *file_errhandler_fn, MPI_Errhandler *errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_File_create_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_File_create_errhandler(file_errhandler_fn, errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)file_errhandler_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler)< */
int MPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_File_get_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_File_get_errhandler(file, errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)file);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler)< */
int MPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_File_set_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_File_set_errhandler(file, errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)file);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Finalized(int *flag)< */
int MPI_Finalized(int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Finalized");
  if (mpi_initialized)
    bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);
  else
    bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-");

  ret = PMPI_Finalized(flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  if (mpi_initialized)
    bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);
  else
    bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-\n");

  return ret;
}

/* parsing >int MPI_Free_mem(void *base)< */
int MPI_Free_mem(void *base) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Free_mem");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Free_mem(base);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)base);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Get_address(const void *location, MPI_Aint *address)< */
int MPI_Get_address(const void *location, MPI_Aint *address) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_address");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Get_address(location, address);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)location);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)address);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Info_create(MPI_Info *info)< */
int MPI_Info_create(MPI_Info *info) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Info_create(info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Info_delete(MPI_Info info, const char *key)< */
int MPI_Info_delete(MPI_Info info, const char *key) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_delete");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Info_delete(info, key);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Info_dup(MPI_Info info, MPI_Info *newinfo)< */
int MPI_Info_dup(MPI_Info info, MPI_Info *newinfo) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_dup");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Info_dup(info, newinfo);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newinfo);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Info_free(MPI_Info *info)< */
int MPI_Info_free(MPI_Info *info) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Info_free(info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag)< */
int MPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Info_get(info, key, valuelen, value, flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)valuelen);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)value);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Info_get_nkeys(MPI_Info info, int *nkeys)< */
int MPI_Info_get_nkeys(MPI_Info info, int *nkeys) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_get_nkeys");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Info_get_nkeys(info, nkeys);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)nkeys);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Info_get_nthkey(MPI_Info info, int n, char *key)< */
int MPI_Info_get_nthkey(MPI_Info info, int n, char *key) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_get_nthkey");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Info_get_nthkey(info, n, key);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag)< */
int MPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_get_valuelen");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Info_get_valuelen(info, key, valuelen, flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)valuelen);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Info_set(MPI_Info info, const char *key, const char *value)< */
int MPI_Info_set(MPI_Info info, const char *key, const char *value) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_set");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Info_set(info, key, value);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)value);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Pack_external(const char datarep[], const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position)< */
int MPI_Pack_external(const char datarep[], const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Pack_external");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Pack_external(datarep, inbuf, incount, datatype, outbuf, outsize, position);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datarep);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)incount);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)outsize);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)position);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Pack_external_size(const char datarep[], int incount, MPI_Datatype datatype, MPI_Aint *size)< */
int MPI_Pack_external_size(const char datarep[], int incount, MPI_Datatype datatype, MPI_Aint *size) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Pack_external_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Pack_external_size(datarep, incount, datatype, size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datarep);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)incount);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status)< */
int MPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Request_get_status");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Request_get_status(request, flag, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Status_c2f(const MPI_Status *c_status, MPI_Fint *f_status)< */
int MPI_Status_c2f(const MPI_Status *c_status, MPI_Fint *f_status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Status_c2f");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Status_c2f(c_status, f_status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)c_status);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)f_status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Status_f2c(const MPI_Fint *f_status, MPI_Status *c_status)< */
int MPI_Status_f2c(const MPI_Fint *f_status, MPI_Status *c_status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Status_f2c");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Status_f2c(f_status, c_status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)f_status);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)c_status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[], const int array_of_distribs[], const int array_of_dargs[], const int array_of_psizes[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[], const int array_of_distribs[], const int array_of_dargs[], const int array_of_psizes[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_darray");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_darray(size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs, array_of_psizes, order, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_gsizes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_distribs);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_dargs);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_psizes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)order);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_hindexed(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_create_hindexed(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_hindexed");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_hindexed(count, array_of_blocklengths, array_of_displacements, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_blocklengths);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_hvector");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_hvector(count, blocklength, stride, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)stride);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_indexed_block(int count, int blocklength, const int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_create_indexed_block(int count, int blocklength, const int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_indexed_block");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_indexed_block(count, blocklength, array_of_displacements, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_hindexed_block(int count, int blocklength, const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_create_hindexed_block(int count, int blocklength, const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_hindexed_block");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_hindexed_block(count, blocklength, array_of_displacements, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype)< */
int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_resized");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_resized(oldtype, lb, extent, newtype);

  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)lb);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)extent);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_struct(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], const MPI_Datatype array_of_types[], MPI_Datatype *newtype)< */
int MPI_Type_create_struct(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], const MPI_Datatype array_of_types[], MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_struct");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements, array_of_types, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_blocklengths);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_types);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
int MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_subarray");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, array_of_starts, order, oldtype, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_sizes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_subsizes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_starts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)order);
  
bufptr += printdatatype(oldtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (oldtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(oldtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent)< */
int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_extent");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_get_extent(datatype, lb, extent);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)lb);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extent);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent)< */
int MPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_true_extent");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_get_true_extent(datatype, true_lb, true_extent);

  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)true_lb);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)true_extent);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Unpack_external(const char datarep[], const void *inbuf, MPI_Aint insize, MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype)< */
int MPI_Unpack_external(const char datarep[], const void *inbuf, MPI_Aint insize, MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Unpack_external");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Unpack_external(datarep, inbuf, insize, position, outbuf, outcount, datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datarep);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)insize);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)position);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)outcount);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_create_errhandler(MPI_Win_errhandler_function *win_errhandler_fn, MPI_Errhandler *errhandler)< */
int MPI_Win_create_errhandler(MPI_Win_errhandler_function *win_errhandler_fn, MPI_Errhandler *errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_create_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_create_errhandler(win_errhandler_fn, errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_errhandler_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler)< */
int MPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_get_errhandler(win, errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler)< */
int MPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_set_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Win_set_errhandler(win, errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_f90_integer(int range, MPI_Datatype *newtype)< */
int MPI_Type_create_f90_integer(int range, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_f90_integer");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_f90_integer(range, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)range);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_f90_real(int precision, int range, MPI_Datatype *newtype)< */
int MPI_Type_create_f90_real(int precision, int range, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_f90_real");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_f90_real(precision, range, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)precision);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)range);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Type_create_f90_complex(int precision, int range, MPI_Datatype *newtype)< */
int MPI_Type_create_f90_complex(int precision, int range, MPI_Datatype *newtype) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_f90_complex");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Type_create_f90_complex(precision, range, newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)precision);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)range);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Reduce_local(const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op op)< */
int MPI_Reduce_local(const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op op) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Reduce_local");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Reduce_local(inbuf, inoutbuf, count, datatype, op);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inoutbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Op_commutative(MPI_Op op, int *commute)< */
int MPI_Op_commutative(MPI_Op op, int *commute) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Op_commutative");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Op_commutative(op, commute);

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)commute);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
int MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Reduce_scatter_block");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, const int sources[], const int sourceweights[], int outdegree, const int destinations[], const int destweights[], MPI_Info info, int reorder, MPI_Comm *comm_dist_graph)< */
int MPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, const int sources[], const int sourceweights[], int outdegree, const int destinations[], const int destweights[], MPI_Info info, int reorder, MPI_Comm *comm_dist_graph) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dist_graph_create_adjacent");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Dist_graph_create_adjacent(comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_old);
{
  int i;
  PMPI_Comm_rank(comm_old, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm_old, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)indegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sources);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sourceweights);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)outdegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destinations);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destweights);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)reorder);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_dist_graph);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Dist_graph_create(MPI_Comm comm_old, int n, const int sources[], const int degrees[], const int destinations[], const int weights[], MPI_Info info, int reorder, MPI_Comm *comm_dist_graph)< */
int MPI_Dist_graph_create(MPI_Comm comm_old, int n, const int sources[], const int degrees[], const int destinations[], const int weights[], MPI_Info info, int reorder, MPI_Comm *comm_dist_graph) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dist_graph_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Dist_graph_create(comm_old, n, sources, degrees, destinations, weights, info, reorder, comm_dist_graph);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_old);
{
  int i;
  PMPI_Comm_rank(comm_old, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm_old, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sources);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)degrees);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destinations);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)weights);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)reorder);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_dist_graph);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Dist_graph_neighbors_count(MPI_Comm comm, int *indegree, int *outdegree, int *weighted)< */
int MPI_Dist_graph_neighbors_count(MPI_Comm comm, int *indegree, int *outdegree, int *weighted) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dist_graph_neighbors_count");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Dist_graph_neighbors_count(comm, indegree, outdegree, weighted);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outdegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)weighted);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[], int maxoutdegree, int destinations[], int destweights[])< */
int MPI_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[], int maxoutdegree, int destinations[], int destweights[]) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dist_graph_neighbors");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Dist_graph_neighbors(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations, destweights);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)maxindegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sources);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sourceweights);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)maxoutdegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destinations);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destweights);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Improbe(int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, MPI_Status *status)< */
int MPI_Improbe(int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Improbe");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Improbe(source, tag, comm, flag, message, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)message);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Request *request)< */
int MPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Imrecv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Imrecv(buf, count, datatype, message, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)message);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status)< */
int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Mprobe");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Mprobe(source, tag, comm, message, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)message);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Mrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Status *status)< */
int MPI_Mrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Status *status) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Mrecv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Mrecv(buf, count, datatype, message, status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)message);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_idup(MPI_Comm comm, MPI_Comm *newcomm, MPI_Request *request)< */
int MPI_Comm_idup(MPI_Comm comm, MPI_Comm *newcomm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_idup");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_idup(comm, newcomm, request);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ibarrier(MPI_Comm comm, MPI_Request *request)< */
int MPI_Ibarrier(MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ibarrier");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ibarrier(comm, request);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ibcast");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ibcast(buffer, count, datatype, root, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buffer);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request)< */
int MPI_Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Igather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request)< */
int MPI_Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Igatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Iscatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request)< */
int MPI_Iscatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iscatter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Iscatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Iscatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request)< */
int MPI_Iscatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iscatterv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
int MPI_Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iallgather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
int MPI_Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iallgatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ialltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ialltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ialltoall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ialltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ialltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ialltoallv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ialltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request *request)< */
int MPI_Ialltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ialltoallw");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendtypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvtypes);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ireduce");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iallreduce");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ireduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ireduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ireduce_scatter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ireduce_scatter_block");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ireduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Iscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
int MPI_Iscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iscan");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Iscan(sendbuf, recvbuf, count, datatype, op, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Iexscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
int MPI_Iexscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iexscan");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Iexscan(sendbuf, recvbuf, count, datatype, op, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)count);
  
bufptr += printdatatype(datatype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (datatype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(datatype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(op, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ineighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ineighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_allgather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ineighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ineighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ineighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_allgatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ineighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ineighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ineighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_alltoall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ineighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ineighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
int MPI_Ineighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_alltoallv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Ineighbor_alltoallw(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request *request)< */
int MPI_Ineighbor_alltoallw(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request *request) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_alltoallw");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Ineighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendtypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvtypes);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)< */
int MPI_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_allgather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Neighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Neighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm)< */
int MPI_Neighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_allgatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Neighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Neighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)< */
int MPI_Neighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_alltoall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Neighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)sendcount);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)recvcount);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)< */
int MPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_alltoallv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Neighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  
bufptr += printdatatype(sendtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (sendtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(sendtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  
bufptr += printdatatype(recvtype, (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size (recvtype, &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent(recvtype, &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Neighbor_alltoallw(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)< */
int MPI_Neighbor_alltoallw(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_alltoallw");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendtypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvtypes);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}

/* parsing >int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm)< */
int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm) { 
  int ret;
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_split_type");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  ret = PMPI_Comm_split_type(comm, split_type, key, info, newcomm);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);
{
  int i;
  PMPI_Comm_rank(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size(comm, &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)split_type);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);

  return ret;
}
#ifdef __cplusplus
}
#endif

