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
  /* pprint(numbuf, 1024, world_rank, world_size); */
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






/* parsing >int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)< */
void F77_FUNC(pmpi_send,PMPI_SEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *ierr); 
void F77_FUNC(mpi_send,MPI_SEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Send");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_send,PMPI_SEND)(buf, count, datatype, dest, tag, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)< */
void F77_FUNC(pmpi_recv,PMPI_RECV)(int *buf, int *count, int *datatype, int *source, int *tag, int *comm, int *status, int *ierr); 
void F77_FUNC(mpi_recv,MPI_RECV)(int *buf, int *count, int *datatype, int *source, int *tag, int *comm, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Recv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_recv,PMPI_RECV)(buf, count, datatype, source, tag, comm, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count)< */
void F77_FUNC(pmpi_get_count,PMPI_GET_COUNT)(int *status, int *datatype, int *count, int *ierr); 
void F77_FUNC(mpi_get_count,MPI_GET_COUNT)(int *status, int *datatype, int *count, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_count");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_get_count,PMPI_GET_COUNT)(status, datatype, count, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)count);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)< */
void F77_FUNC(pmpi_bsend,PMPI_BSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *ierr); 
void F77_FUNC(mpi_bsend,MPI_BSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Bsend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_bsend,PMPI_BSEND)(buf, count, datatype, dest, tag, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)< */
void F77_FUNC(pmpi_ssend,PMPI_SSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *ierr); 
void F77_FUNC(mpi_ssend,MPI_SSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ssend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ssend,PMPI_SSEND)(buf, count, datatype, dest, tag, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)< */
void F77_FUNC(pmpi_rsend,PMPI_RSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *ierr); 
void F77_FUNC(mpi_rsend,MPI_RSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rsend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_rsend,PMPI_RSEND)(buf, count, datatype, dest, tag, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Buffer_attach(void *buffer, int size)< */
void F77_FUNC(pmpi_buffer_attach,PMPI_BUFFER_ATTACH)(int *buffer, int *size, int *ierr); 
void F77_FUNC(mpi_buffer_attach,MPI_BUFFER_ATTACH)(int *buffer, int *size, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Buffer_attach");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_buffer_attach,PMPI_BUFFER_ATTACH)(buffer, size, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buffer);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Buffer_detach(void *buffer_addr, int *size)< */
void F77_FUNC(pmpi_buffer_detach,PMPI_BUFFER_DETACH)(int *buffer_addr, int *size, int *ierr); 
void F77_FUNC(mpi_buffer_detach,MPI_BUFFER_DETACH)(int *buffer_addr, int *size, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Buffer_detach");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_buffer_detach,PMPI_BUFFER_DETACH)(buffer_addr, size, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buffer_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_isend,PMPI_ISEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_isend,MPI_ISEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Isend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_isend,PMPI_ISEND)(buf, count, datatype, dest, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ibsend,PMPI_IBSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ibsend,MPI_IBSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ibsend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ibsend,PMPI_IBSEND)(buf, count, datatype, dest, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_issend,PMPI_ISSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_issend,MPI_ISSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Issend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_issend,PMPI_ISSEND)(buf, count, datatype, dest, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_irsend,PMPI_IRSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_irsend,MPI_IRSEND)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Irsend");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_irsend,PMPI_IRSEND)(buf, count, datatype, dest, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_irecv,PMPI_IRECV)(int *buf, int *count, int *datatype, int *source, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_irecv,MPI_IRECV)(int *buf, int *count, int *datatype, int *source, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Irecv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_irecv,PMPI_IRECV)(buf, count, datatype, source, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Wait(MPI_Request *request, MPI_Status *status)< */
void F77_FUNC(pmpi_wait,PMPI_WAIT)(int *request, int *status, int *ierr); 
void F77_FUNC(mpi_wait,MPI_WAIT)(int *request, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Wait");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_wait,PMPI_WAIT)(request, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)< */
void F77_FUNC(pmpi_test,PMPI_TEST)(int *request, int *flag, int *status, int *ierr); 
void F77_FUNC(mpi_test,MPI_TEST)(int *request, int *flag, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Test");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_test,PMPI_TEST)(request, flag, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Request_free(MPI_Request *request)< */
void F77_FUNC(pmpi_request_free,PMPI_REQUEST_FREE)(int *request, int *ierr); 
void F77_FUNC(mpi_request_free,MPI_REQUEST_FREE)(int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Request_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_request_free,PMPI_REQUEST_FREE)(request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Waitany(int count, MPI_Request array_of_requests[], int *indx, MPI_Status *status)< */
void F77_FUNC(pmpi_waitany,PMPI_WAITANY)(int *count, int *array_of_requests[], int *indx, int *status, int *ierr); 
void F77_FUNC(mpi_waitany,MPI_WAITANY)(int *count, int *array_of_requests[], int *indx, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Waitany");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_waitany,PMPI_WAITANY)(count, array_of_requests, indx, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Testany(int count, MPI_Request array_of_requests[], int *indx, int *flag, MPI_Status *status)< */
void F77_FUNC(pmpi_testany,PMPI_TESTANY)(int *count, int *array_of_requests[], int *indx, int *flag, int *status, int *ierr); 
void F77_FUNC(mpi_testany,MPI_TESTANY)(int *count, int *array_of_requests[], int *indx, int *flag, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Testany");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_testany,PMPI_TESTANY)(count, array_of_requests, indx, flag, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])< */
void F77_FUNC(pmpi_waitall,PMPI_WAITALL)(int *count, int *array_of_requests[], int *array_of_statuses[], int *ierr); 
void F77_FUNC(mpi_waitall,MPI_WAITALL)(int *count, int *array_of_requests[], int *array_of_statuses[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Waitall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_waitall,PMPI_WAITALL)(count, array_of_requests, array_of_statuses, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[])< */
void F77_FUNC(pmpi_testall,PMPI_TESTALL)(int *count, int *array_of_requests[], int *flag, int *array_of_statuses[], int *ierr); 
void F77_FUNC(mpi_testall,MPI_TESTALL)(int *count, int *array_of_requests[], int *flag, int *array_of_statuses[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Testall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_testall,PMPI_TESTALL)(count, array_of_requests, flag, array_of_statuses, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])< */
void F77_FUNC(pmpi_waitsome,PMPI_WAITSOME)(int *incount, int *array_of_requests[], int *outcount, int *array_of_indices[], int *array_of_statuses[], int *ierr); 
void F77_FUNC(mpi_waitsome,MPI_WAITSOME)(int *incount, int *array_of_requests[], int *outcount, int *array_of_indices[], int *array_of_statuses[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Waitsome");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_waitsome,PMPI_WAITSOME)(incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*incount);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outcount);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_indices);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])< */
void F77_FUNC(pmpi_testsome,PMPI_TESTSOME)(int *incount, int *array_of_requests[], int *outcount, int *array_of_indices[], int *array_of_statuses[], int *ierr); 
void F77_FUNC(mpi_testsome,MPI_TESTSOME)(int *incount, int *array_of_requests[], int *outcount, int *array_of_indices[], int *array_of_statuses[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Testsome");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_testsome,PMPI_TESTSOME)(incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*incount);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outcount);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_indices);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_statuses);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status)< */
void F77_FUNC(pmpi_iprobe,PMPI_IPROBE)(int *source, int *tag, int *comm, int *flag, int *status, int *ierr); 
void F77_FUNC(mpi_iprobe,MPI_IPROBE)(int *source, int *tag, int *comm, int *flag, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iprobe");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_iprobe,PMPI_IPROBE)(source, tag, comm, flag, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status)< */
void F77_FUNC(pmpi_probe,PMPI_PROBE)(int *source, int *tag, int *comm, int *status, int *ierr); 
void F77_FUNC(mpi_probe,MPI_PROBE)(int *source, int *tag, int *comm, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Probe");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_probe,PMPI_PROBE)(source, tag, comm, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Cancel(MPI_Request *request)< */
void F77_FUNC(pmpi_cancel,PMPI_CANCEL)(int *request, int *ierr); 
void F77_FUNC(mpi_cancel,MPI_CANCEL)(int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cancel");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_cancel,PMPI_CANCEL)(request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Test_cancelled(const MPI_Status *status, int *flag)< */
void F77_FUNC(pmpi_test_cancelled,PMPI_TEST_CANCELLED)(int *status, int *flag, int *ierr); 
void F77_FUNC(mpi_test_cancelled,MPI_TEST_CANCELLED)(int *status, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Test_cancelled");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_test_cancelled,PMPI_TEST_CANCELLED)(status, flag, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_send_init,PMPI_SEND_INIT)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_send_init,MPI_SEND_INIT)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Send_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_send_init,PMPI_SEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_bsend_init,PMPI_BSEND_INIT)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_bsend_init,MPI_BSEND_INIT)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Bsend_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_bsend_init,PMPI_BSEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ssend_init,PMPI_SSEND_INIT)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ssend_init,MPI_SSEND_INIT)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ssend_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ssend_init,PMPI_SSEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Rsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_rsend_init,PMPI_RSEND_INIT)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_rsend_init,MPI_RSEND_INIT)(int *buf, int *count, int *datatype, int *dest, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rsend_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_rsend_init,PMPI_RSEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_recv_init,PMPI_RECV_INIT)(int *buf, int *count, int *datatype, int *source, int *tag, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_recv_init,MPI_RECV_INIT)(int *buf, int *count, int *datatype, int *source, int *tag, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Recv_init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_recv_init,PMPI_RECV_INIT)(buf, count, datatype, source, tag, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Start(MPI_Request *request)< */
void F77_FUNC(pmpi_start,PMPI_START)(int *request, int *ierr); 
void F77_FUNC(mpi_start,MPI_START)(int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Start");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_start,PMPI_START)(request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Startall(int count, MPI_Request array_of_requests[])< */
void F77_FUNC(pmpi_startall,PMPI_STARTALL)(int *count, int *array_of_requests[], int *ierr); 
void F77_FUNC(mpi_startall,MPI_STARTALL)(int *count, int *array_of_requests[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Startall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_startall,PMPI_STARTALL)(count, array_of_requests, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_requests);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)< */
void F77_FUNC(pmpi_sendrecv,PMPI_SENDRECV)(int *sendbuf, int *sendcount, int *sendtype, int *dest, int *sendtag, int *recvbuf, int *recvcount, int *recvtype, int *source, int *recvtag, int *comm, int *status, int *ierr); 
void F77_FUNC(mpi_sendrecv,MPI_SENDRECV)(int *sendbuf, int *sendcount, int *sendtype, int *dest, int *sendtag, int *recvbuf, int *recvcount, int *recvtype, int *source, int *recvtag, int *comm, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Sendrecv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_sendrecv,PMPI_SENDRECV)(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendtag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvtag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status)< */
void F77_FUNC(pmpi_sendrecv_replace,PMPI_SENDRECV_REPLACE)(int *buf, int *count, int *datatype, int *dest, int *sendtag, int *source, int *recvtag, int *comm, int *status, int *ierr); 
void F77_FUNC(mpi_sendrecv_replace,MPI_SENDRECV_REPLACE)(int *buf, int *count, int *datatype, int *dest, int *sendtag, int *source, int *recvtag, int *comm, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Sendrecv_replace");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_sendrecv_replace,PMPI_SENDRECV_REPLACE)(buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*dest);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendtag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvtag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_contiguous,PMPI_TYPE_CONTIGUOUS)(int *count, int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_contiguous,MPI_TYPE_CONTIGUOUS)(int *count, int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_contiguous");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_contiguous,PMPI_TYPE_CONTIGUOUS)(count, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_vector,PMPI_TYPE_VECTOR)(int *count, int *blocklength, int *stride, int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_vector,MPI_TYPE_VECTOR)(int *count, int *blocklength, int *stride, int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_vector");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_vector,PMPI_TYPE_VECTOR)(count, blocklength, stride, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*stride);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_hvector,PMPI_TYPE_HVECTOR)(int *count, int *blocklength, MPI_Aint *stride, int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_hvector,MPI_TYPE_HVECTOR)(int *count, int *blocklength, MPI_Aint *stride, int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_hvector");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_hvector,PMPI_TYPE_HVECTOR)(count, blocklength, stride, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*stride);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_indexed(int count, const int *array_of_blocklengths, const int *array_of_displacements, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_indexed,PMPI_TYPE_INDEXED)(int *count, int *array_of_blocklengths, int *array_of_displacements, int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_indexed,MPI_TYPE_INDEXED)(int *count, int *array_of_blocklengths, int *array_of_displacements, int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_indexed");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_indexed,PMPI_TYPE_INDEXED)(count, array_of_blocklengths, array_of_displacements, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_blocklengths);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_size(MPI_Datatype datatype, int *size)< */
void F77_FUNC(mpi_type_size,MPI_TYPE_SIZE)(int *datatype, int *size, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_size,PMPI_TYPE_SIZE)(datatype, size, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_lb(MPI_Datatype datatype, MPI_Aint *displacement)< */
void F77_FUNC(pmpi_type_lb,PMPI_TYPE_LB)(int *datatype, MPI_Aint *displacement, int *ierr); 
void F77_FUNC(mpi_type_lb,MPI_TYPE_LB)(int *datatype, MPI_Aint *displacement, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_lb");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_lb,PMPI_TYPE_LB)(datatype, displacement, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displacement);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_ub(MPI_Datatype datatype, MPI_Aint *displacement)< */
void F77_FUNC(pmpi_type_ub,PMPI_TYPE_UB)(int *datatype, MPI_Aint *displacement, int *ierr); 
void F77_FUNC(mpi_type_ub,MPI_TYPE_UB)(int *datatype, MPI_Aint *displacement, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_ub");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_ub,PMPI_TYPE_UB)(datatype, displacement, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displacement);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_commit(MPI_Datatype *datatype)< */
void F77_FUNC(pmpi_type_commit,PMPI_TYPE_COMMIT)(int *datatype, int *ierr); 
void F77_FUNC(mpi_type_commit,MPI_TYPE_COMMIT)(int *datatype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_commit");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_commit,PMPI_TYPE_COMMIT)(datatype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_free(MPI_Datatype *datatype)< */
void F77_FUNC(pmpi_type_free,PMPI_TYPE_FREE)(int *datatype, int *ierr); 
void F77_FUNC(mpi_type_free,MPI_TYPE_FREE)(int *datatype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_free,PMPI_TYPE_FREE)(datatype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Get_elements(const MPI_Status *status, MPI_Datatype datatype, int *count)< */
void F77_FUNC(pmpi_get_elements,PMPI_GET_ELEMENTS)(int *status, int *datatype, int *count, int *ierr); 
void F77_FUNC(mpi_get_elements,MPI_GET_ELEMENTS)(int *status, int *datatype, int *count, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_elements");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_get_elements,PMPI_GET_ELEMENTS)(status, datatype, count, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)count);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Pack(const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, int outsize, int *position, MPI_Comm comm)< */
void F77_FUNC(pmpi_pack,PMPI_PACK)(int *inbuf, int *incount, int *datatype, int *outbuf, int *outsize, int *position, int *comm, int *ierr); 
void F77_FUNC(mpi_pack,MPI_PACK)(int *inbuf, int *incount, int *datatype, int *outbuf, int *outsize, int *position, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Pack");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_pack,PMPI_PACK)(inbuf, incount, datatype, outbuf, outsize, position, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*incount);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*outsize);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)position);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Unpack(const void *inbuf, int insize, int *position, void *outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm)< */
void F77_FUNC(pmpi_unpack,PMPI_UNPACK)(int *inbuf, int *insize, int *position, int *outbuf, int *outcount, int *datatype, int *comm, int *ierr); 
void F77_FUNC(mpi_unpack,MPI_UNPACK)(int *inbuf, int *insize, int *position, int *outbuf, int *outcount, int *datatype, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Unpack");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_unpack,PMPI_UNPACK)(inbuf, insize, position, outbuf, outcount, datatype, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*insize);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)position);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*outcount);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size)< */
void F77_FUNC(pmpi_pack_size,PMPI_PACK_SIZE)(int *incount, int *datatype, int *comm, int *size, int *ierr); 
void F77_FUNC(mpi_pack_size,MPI_PACK_SIZE)(int *incount, int *datatype, int *comm, int *size, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Pack_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_pack_size,PMPI_PACK_SIZE)(incount, datatype, comm, size, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*incount);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Barrier(MPI_Comm comm)< */
void F77_FUNC(pmpi_barrier,PMPI_BARRIER)(int *comm, int *ierr); 
void F77_FUNC(mpi_barrier,MPI_BARRIER)(int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Barrier");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_barrier,PMPI_BARRIER)(comm, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)< */
void F77_FUNC(pmpi_bcast,PMPI_BCAST)(int *buffer, int *count, int *datatype, int *root, int *comm, int *ierr); 
void F77_FUNC(mpi_bcast,MPI_BCAST)(int *buffer, int *count, int *datatype, int *root, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Bcast");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_bcast,PMPI_BCAST)(buffer, count, datatype, root, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buffer);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)< */
void F77_FUNC(pmpi_gather,PMPI_GATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *ierr); 
void F77_FUNC(mpi_gather,MPI_GATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Gather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_gather,PMPI_GATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm)< */
void F77_FUNC(pmpi_gatherv,PMPI_GATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts, int *displs, int *recvtype, int *root, int *comm, int *ierr); 
void F77_FUNC(mpi_gatherv,MPI_GATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts, int *displs, int *recvtype, int *root, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Gatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_gatherv,PMPI_GATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", recvcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", recvcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", displs[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", displs[i]);
  }
}

  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)< */
void F77_FUNC(pmpi_scatter,PMPI_SCATTER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *ierr); 
void F77_FUNC(mpi_scatter,MPI_SCATTER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Scatter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_scatter,PMPI_SCATTER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)< */
void F77_FUNC(pmpi_scatterv,PMPI_SCATTERV)(int *sendbuf, int *sendcounts, int *displs, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *ierr); 
void F77_FUNC(mpi_scatterv,MPI_SCATTERV)(int *sendbuf, int *sendcounts, int *displs, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Scatterv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_scatterv,PMPI_SCATTERV)(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", sendcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", sendcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", displs[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", displs[i]);
  }
}

  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)< */
void F77_FUNC(pmpi_allgather,PMPI_ALLGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *ierr); 
void F77_FUNC(mpi_allgather,MPI_ALLGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Allgather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_allgather,PMPI_ALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)< */
void F77_FUNC(pmpi_allgatherv,PMPI_ALLGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts, int *displs, int *recvtype, int *comm, int *ierr); 
void F77_FUNC(mpi_allgatherv,MPI_ALLGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts, int *displs, int *recvtype, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Allgatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_allgatherv,PMPI_ALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", recvcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", recvcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", displs[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", displs[i]);
  }
}

  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)< */
void F77_FUNC(pmpi_alltoall,PMPI_ALLTOALL)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *ierr); 
void F77_FUNC(mpi_alltoall,MPI_ALLTOALL)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Alltoall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_alltoall,PMPI_ALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Alltoallv(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)< */
void F77_FUNC(pmpi_alltoallv,PMPI_ALLTOALLV)(int *sendbuf, int *sendcounts, int *sdispls, int *sendtype, int *recvbuf, int *recvcounts, int *rdispls, int *recvtype, int *comm, int *ierr); 
void F77_FUNC(mpi_alltoallv,MPI_ALLTOALLV)(int *sendbuf, int *sendcounts, int *sdispls, int *sendtype, int *recvbuf, int *recvcounts, int *rdispls, int *recvtype, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Alltoallv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_alltoallv,PMPI_ALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", sendcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", sendcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", sdispls[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", sdispls[i]);
  }
}

  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", recvcounts[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", recvcounts[i]);
  }
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", rdispls[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", rdispls[i]);
  }
}

  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)< */
void F77_FUNC(pmpi_alltoallw,PMPI_ALLTOALLW)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtypes[], int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtypes[], int *comm, int *ierr); 
void F77_FUNC(mpi_alltoallw,MPI_ALLTOALLW)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtypes[], int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtypes[], int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Alltoallw");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_alltoallw,PMPI_ALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendtypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvtypes);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
void F77_FUNC(pmpi_exscan,PMPI_EXSCAN)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *ierr); 
void F77_FUNC(mpi_exscan,MPI_EXSCAN)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Exscan");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_exscan,PMPI_EXSCAN)(sendbuf, recvbuf, count, datatype, op, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)< */
void F77_FUNC(pmpi_reduce,PMPI_REDUCE)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *root, int *comm, int *ierr); 
void F77_FUNC(mpi_reduce,MPI_REDUCE)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *root, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Reduce");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_reduce,PMPI_REDUCE)(sendbuf, recvbuf, count, datatype, op, root, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op)< */
void F77_FUNC(pmpi_op_create,PMPI_OP_CREATE)(int *user_fn, int *commute, int *op, int *ierr); 
void F77_FUNC(mpi_op_create,MPI_OP_CREATE)(int *user_fn, int *commute, int *op, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Op_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_op_create,PMPI_OP_CREATE)(user_fn, commute, op, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)user_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*commute);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)op);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Op_free(MPI_Op *op)< */
void F77_FUNC(pmpi_op_free,PMPI_OP_FREE)(int *op, int *ierr); 
void F77_FUNC(mpi_op_free,MPI_OP_FREE)(int *op, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Op_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_op_free,PMPI_OP_FREE)(op, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)op);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
void F77_FUNC(pmpi_allreduce,PMPI_ALLREDUCE)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *ierr); 
void F77_FUNC(mpi_allreduce,MPI_ALLREDUCE)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Allreduce");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_allreduce,PMPI_ALLREDUCE)(sendbuf, recvbuf, count, datatype, op, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
void F77_FUNC(pmpi_reduce_scatter,PMPI_REDUCE_SCATTER)(int *sendbuf, int *recvbuf, int *recvcounts[], int *datatype, int *op, int *comm, int *ierr); 
void F77_FUNC(mpi_reduce_scatter,MPI_REDUCE_SCATTER)(int *sendbuf, int *recvbuf, int *recvcounts[], int *datatype, int *op, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Reduce_scatter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_reduce_scatter,PMPI_REDUCE_SCATTER)(sendbuf, recvbuf, recvcounts, datatype, op, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
void F77_FUNC(pmpi_scan,PMPI_SCAN)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *ierr); 
void F77_FUNC(mpi_scan,MPI_SCAN)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Scan");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_scan,PMPI_SCAN)(sendbuf, recvbuf, count, datatype, op, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_size(MPI_Group group, int *size)< */
void F77_FUNC(pmpi_group_size,PMPI_GROUP_SIZE)(int *group, int *size, int *ierr); 
void F77_FUNC(mpi_group_size,MPI_GROUP_SIZE)(int *group, int *size, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_size,PMPI_GROUP_SIZE)(group, size, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_rank(MPI_Group group, int *rank)< */
void F77_FUNC(pmpi_group_rank,PMPI_GROUP_RANK)(int *group, int *rank, int *ierr); 
void F77_FUNC(mpi_group_rank,MPI_GROUP_RANK)(int *group, int *rank, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_rank");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_rank,PMPI_GROUP_RANK)(group, rank, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[], MPI_Group group2, int ranks2[])< */
void F77_FUNC(pmpi_group_translate_ranks,PMPI_GROUP_TRANSLATE_RANKS)(int *group1, int *n, int *ranks1[], int *group2, int *ranks2[], int *ierr); 
void F77_FUNC(mpi_group_translate_ranks,MPI_GROUP_TRANSLATE_RANKS)(int *group1, int *n, int *ranks1[], int *group2, int *ranks2[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_translate_ranks");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_translate_ranks,PMPI_GROUP_TRANSLATE_RANKS)(group1, n, ranks1, group2, ranks2, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranks1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranks2);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_compare(MPI_Group group1, MPI_Group group2, int *result)< */
void F77_FUNC(pmpi_group_compare,PMPI_GROUP_COMPARE)(int *group1, int *group2, int *result, int *ierr); 
void F77_FUNC(mpi_group_compare,MPI_GROUP_COMPARE)(int *group1, int *group2, int *result, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_compare");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_compare,PMPI_GROUP_COMPARE)(group1, group2, result, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_group(MPI_Comm comm, MPI_Group *group)< */
void F77_FUNC(pmpi_comm_group,PMPI_COMM_GROUP)(int *comm, int *group, int *ierr); 
void F77_FUNC(mpi_comm_group,MPI_COMM_GROUP)(int *comm, int *group, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_group");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_group,PMPI_COMM_GROUP)(comm, group, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)< */
void F77_FUNC(pmpi_group_union,PMPI_GROUP_UNION)(int *group1, int *group2, int *newgroup, int *ierr); 
void F77_FUNC(mpi_group_union,MPI_GROUP_UNION)(int *group1, int *group2, int *newgroup, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_union");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_union,PMPI_GROUP_UNION)(group1, group2, newgroup, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)< */
void F77_FUNC(pmpi_group_intersection,PMPI_GROUP_INTERSECTION)(int *group1, int *group2, int *newgroup, int *ierr); 
void F77_FUNC(mpi_group_intersection,MPI_GROUP_INTERSECTION)(int *group1, int *group2, int *newgroup, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_intersection");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_intersection,PMPI_GROUP_INTERSECTION)(group1, group2, newgroup, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)< */
void F77_FUNC(pmpi_group_difference,PMPI_GROUP_DIFFERENCE)(int *group1, int *group2, int *newgroup, int *ierr); 
void F77_FUNC(mpi_group_difference,MPI_GROUP_DIFFERENCE)(int *group1, int *group2, int *newgroup, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_difference");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_difference,PMPI_GROUP_DIFFERENCE)(group1, group2, newgroup, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group1);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group2);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup)< */
void F77_FUNC(pmpi_group_incl,PMPI_GROUP_INCL)(int *group, int *n, int *ranks[], int *newgroup, int *ierr); 
void F77_FUNC(mpi_group_incl,MPI_GROUP_INCL)(int *group, int *n, int *ranks[], int *newgroup, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_incl");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_incl,PMPI_GROUP_INCL)(group, n, ranks, newgroup, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranks);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_excl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup)< */
void F77_FUNC(pmpi_group_excl,PMPI_GROUP_EXCL)(int *group, int *n, int *ranks[], int *newgroup, int *ierr); 
void F77_FUNC(mpi_group_excl,MPI_GROUP_EXCL)(int *group, int *n, int *ranks[], int *newgroup, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_excl");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_excl,PMPI_GROUP_EXCL)(group, n, ranks, newgroup, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranks);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)< */
void F77_FUNC(pmpi_group_range_incl,PMPI_GROUP_RANGE_INCL)(int *group, int *n, int *ranges[][3], int *newgroup, int *ierr); 
void F77_FUNC(mpi_group_range_incl,MPI_GROUP_RANGE_INCL)(int *group, int *n, int *ranges[][3], int *newgroup, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_range_incl");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_range_incl,PMPI_GROUP_RANGE_INCL)(group, n, ranges, newgroup, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)< */
void F77_FUNC(pmpi_group_range_excl,PMPI_GROUP_RANGE_EXCL)(int *group, int *n, int *ranges[][3], int *newgroup, int *ierr); 
void F77_FUNC(mpi_group_range_excl,MPI_GROUP_RANGE_EXCL)(int *group, int *n, int *ranges[][3], int *newgroup, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_range_excl");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_range_excl,PMPI_GROUP_RANGE_EXCL)(group, n, ranges, newgroup, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ranges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newgroup);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Group_free(MPI_Group *group)< */
void F77_FUNC(pmpi_group_free,PMPI_GROUP_FREE)(int *group, int *ierr); 
void F77_FUNC(mpi_group_free,MPI_GROUP_FREE)(int *group, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Group_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_group_free,PMPI_GROUP_FREE)(group, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_size(MPI_Comm comm, int *size)< */
void F77_FUNC(mpi_comm_size,MPI_COMM_SIZE)(int *comm, int *size, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_size,PMPI_COMM_SIZE)(comm, size, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_rank(MPI_Comm comm, int *rank)< */
void F77_FUNC(mpi_comm_rank,MPI_COMM_RANK)(int *comm, int *rank, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_rank");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_rank,PMPI_COMM_RANK)(comm, rank, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result)< */
void F77_FUNC(pmpi_comm_compare,PMPI_COMM_COMPARE)(int *comm1, int *comm2, int *result, int *ierr); 
void F77_FUNC(mpi_comm_compare,MPI_COMM_COMPARE)(int *comm1, int *comm2, int *result, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_compare");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_compare,PMPI_COMM_COMPARE)(comm1, comm2, result, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm1);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm1, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm1, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm2);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm2, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm2, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)< */
void F77_FUNC(pmpi_comm_dup,PMPI_COMM_DUP)(int *comm, int *newcomm, int *ierr); 
void F77_FUNC(mpi_comm_dup,MPI_COMM_DUP)(int *comm, int *newcomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_dup");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_dup,PMPI_COMM_DUP)(comm, newcomm, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm)< */
void F77_FUNC(pmpi_comm_dup_with_info,PMPI_COMM_DUP_WITH_INFO)(int *comm, int *info, int *newcomm, int *ierr); 
void F77_FUNC(mpi_comm_dup_with_info,MPI_COMM_DUP_WITH_INFO)(int *comm, int *info, int *newcomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_dup_with_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_dup_with_info,PMPI_COMM_DUP_WITH_INFO)(comm, info, newcomm, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)< */
void F77_FUNC(pmpi_comm_create,PMPI_COMM_CREATE)(int *comm, int *group, int *newcomm, int *ierr); 
void F77_FUNC(mpi_comm_create,MPI_COMM_CREATE)(int *comm, int *group, int *newcomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_create,PMPI_COMM_CREATE)(comm, group, newcomm, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)< */
void F77_FUNC(pmpi_comm_split,PMPI_COMM_SPLIT)(int *comm, int *color, int *key, int *newcomm, int *ierr); 
void F77_FUNC(mpi_comm_split,MPI_COMM_SPLIT)(int *comm, int *color, int *key, int *newcomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_split");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_split,PMPI_COMM_SPLIT)(comm, color, key, newcomm, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*color);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_free(MPI_Comm *comm)< */
void F77_FUNC(pmpi_comm_free,PMPI_COMM_FREE)(int *comm, int *ierr); 
void F77_FUNC(mpi_comm_free,MPI_COMM_FREE)(int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_free,PMPI_COMM_FREE)(comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_test_inter(MPI_Comm comm, int *flag)< */
void F77_FUNC(pmpi_comm_test_inter,PMPI_COMM_TEST_INTER)(int *comm, int *flag, int *ierr); 
void F77_FUNC(mpi_comm_test_inter,MPI_COMM_TEST_INTER)(int *comm, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_test_inter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_test_inter,PMPI_COMM_TEST_INTER)(comm, flag, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_remote_size(MPI_Comm comm, int *size)< */
void F77_FUNC(pmpi_comm_remote_size,PMPI_COMM_REMOTE_SIZE)(int *comm, int *size, int *ierr); 
void F77_FUNC(mpi_comm_remote_size,MPI_COMM_REMOTE_SIZE)(int *comm, int *size, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_remote_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_remote_size,PMPI_COMM_REMOTE_SIZE)(comm, size, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group)< */
void F77_FUNC(pmpi_comm_remote_group,PMPI_COMM_REMOTE_GROUP)(int *comm, int *group, int *ierr); 
void F77_FUNC(mpi_comm_remote_group,MPI_COMM_REMOTE_GROUP)(int *comm, int *group, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_remote_group");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_remote_group,PMPI_COMM_REMOTE_GROUP)(comm, group, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *newintercomm)< */
void F77_FUNC(pmpi_intercomm_create,PMPI_INTERCOMM_CREATE)(int *local_comm, int *local_leader, int *peer_comm, int *remote_leader, int *tag, int *newintercomm, int *ierr); 
void F77_FUNC(mpi_intercomm_create,MPI_INTERCOMM_CREATE)(int *local_comm, int *local_leader, int *peer_comm, int *remote_leader, int *tag, int *newintercomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Intercomm_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_intercomm_create,PMPI_INTERCOMM_CREATE)(local_comm, local_leader, peer_comm, remote_leader, tag, newintercomm, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *local_comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(local_comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(local_comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*local_leader);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *peer_comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(peer_comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(peer_comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*remote_leader);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newintercomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm)< */
void F77_FUNC(pmpi_intercomm_merge,PMPI_INTERCOMM_MERGE)(int *intercomm, int *high, int *newintracomm, int *ierr); 
void F77_FUNC(mpi_intercomm_merge,MPI_INTERCOMM_MERGE)(int *intercomm, int *high, int *newintracomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Intercomm_merge");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_intercomm_merge,PMPI_INTERCOMM_MERGE)(intercomm, high, newintracomm, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *intercomm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(intercomm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(intercomm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*high);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newintracomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn, int *keyval, void *extra_state)< */
void F77_FUNC(pmpi_keyval_create,PMPI_KEYVAL_CREATE)(int *copy_fn, int *delete_fn, int *keyval, int *extra_state, int *ierr); 
void F77_FUNC(mpi_keyval_create,MPI_KEYVAL_CREATE)(int *copy_fn, int *delete_fn, int *keyval, int *extra_state, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Keyval_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_keyval_create,PMPI_KEYVAL_CREATE)(copy_fn, delete_fn, keyval, extra_state, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)copy_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)delete_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Keyval_free(int *keyval)< */
void F77_FUNC(pmpi_keyval_free,PMPI_KEYVAL_FREE)(int *keyval, int *ierr); 
void F77_FUNC(mpi_keyval_free,MPI_KEYVAL_FREE)(int *keyval, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Keyval_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_keyval_free,PMPI_KEYVAL_FREE)(keyval, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val)< */
void F77_FUNC(pmpi_attr_put,PMPI_ATTR_PUT)(int *comm, int *keyval, int *attribute_val, int *ierr); 
void F77_FUNC(mpi_attr_put,MPI_ATTR_PUT)(int *comm, int *keyval, int *attribute_val, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Attr_put");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_attr_put,PMPI_ATTR_PUT)(comm, keyval, attribute_val, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag)< */
void F77_FUNC(pmpi_attr_get,PMPI_ATTR_GET)(int *comm, int *keyval, int *attribute_val, int *flag, int *ierr); 
void F77_FUNC(mpi_attr_get,MPI_ATTR_GET)(int *comm, int *keyval, int *attribute_val, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Attr_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_attr_get,PMPI_ATTR_GET)(comm, keyval, attribute_val, flag, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Attr_delete(MPI_Comm comm, int keyval)< */
void F77_FUNC(pmpi_attr_delete,PMPI_ATTR_DELETE)(int *comm, int *keyval, int *ierr); 
void F77_FUNC(mpi_attr_delete,MPI_ATTR_DELETE)(int *comm, int *keyval, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Attr_delete");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_attr_delete,PMPI_ATTR_DELETE)(comm, keyval, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Topo_test(MPI_Comm comm, int *status)< */
void F77_FUNC(pmpi_topo_test,PMPI_TOPO_TEST)(int *comm, int *status, int *ierr); 
void F77_FUNC(mpi_topo_test,MPI_TOPO_TEST)(int *comm, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Topo_test");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_topo_test,PMPI_TOPO_TEST)(comm, status, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart)< */
void F77_FUNC(pmpi_cart_create,PMPI_CART_CREATE)(int *comm_old, int *ndims, int *dims[], int *periods[], int *reorder, int *comm_cart, int *ierr); 
void F77_FUNC(mpi_cart_create,MPI_CART_CREATE)(int *comm_old, int *ndims, int *dims[], int *periods[], int *reorder, int *comm_cart, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_cart_create,PMPI_CART_CREATE)(comm_old, ndims, dims, periods, reorder, comm_cart, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm_old);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm_old, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm_old, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)dims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)periods);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*reorder);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_cart);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Dims_create(int nnodes, int ndims, int dims[])< */
void F77_FUNC(pmpi_dims_create,PMPI_DIMS_CREATE)(int *nnodes, int *ndims, int *dims[], int *ierr); 
void F77_FUNC(mpi_dims_create,MPI_DIMS_CREATE)(int *nnodes, int *ndims, int *dims[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dims_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_dims_create,PMPI_DIMS_CREATE)(nnodes, ndims, dims, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*nnodes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)dims);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Graph_create(MPI_Comm comm_old, int nnodes, const int indx[], const int edges[], int reorder, MPI_Comm *comm_graph)< */
void F77_FUNC(pmpi_graph_create,PMPI_GRAPH_CREATE)(int *comm_old, int *nnodes, int *indx[], int *edges[], int *reorder, int *comm_graph, int *ierr); 
void F77_FUNC(mpi_graph_create,MPI_GRAPH_CREATE)(int *comm_old, int *nnodes, int *indx[], int *edges[], int *reorder, int *comm_graph, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_graph_create,PMPI_GRAPH_CREATE)(comm_old, nnodes, indx, edges, reorder, comm_graph, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm_old);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm_old, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm_old, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*nnodes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)edges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*reorder);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_graph);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges)< */
void F77_FUNC(pmpi_graphdims_get,PMPI_GRAPHDIMS_GET)(int *comm, int *nnodes, int *nedges, int *ierr); 
void F77_FUNC(mpi_graphdims_get,MPI_GRAPHDIMS_GET)(int *comm, int *nnodes, int *nedges, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graphdims_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_graphdims_get,PMPI_GRAPHDIMS_GET)(comm, nnodes, nedges, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)nnodes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)nedges);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int indx[], int edges[])< */
void F77_FUNC(pmpi_graph_get,PMPI_GRAPH_GET)(int *comm, int *maxindex, int *maxedges, int *indx[], int *edges[], int *ierr); 
void F77_FUNC(mpi_graph_get,MPI_GRAPH_GET)(int *comm, int *maxindex, int *maxedges, int *indx[], int *edges[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_graph_get,PMPI_GRAPH_GET)(comm, maxindex, maxedges, indx, edges, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*maxindex);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*maxedges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)edges);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Cartdim_get(MPI_Comm comm, int *ndims)< */
void F77_FUNC(pmpi_cartdim_get,PMPI_CARTDIM_GET)(int *comm, int *ndims, int *ierr); 
void F77_FUNC(mpi_cartdim_get,MPI_CARTDIM_GET)(int *comm, int *ndims, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cartdim_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_cartdim_get,PMPI_CARTDIM_GET)(comm, ndims, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)ndims);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[])< */
void F77_FUNC(pmpi_cart_get,PMPI_CART_GET)(int *comm, int *maxdims, int *dims[], int *periods[], int *coords[], int *ierr); 
void F77_FUNC(mpi_cart_get,MPI_CART_GET)(int *comm, int *maxdims, int *dims[], int *periods[], int *coords[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_cart_get,PMPI_CART_GET)(comm, maxdims, dims, periods, coords, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*maxdims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)dims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)periods);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)coords);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Cart_rank(MPI_Comm comm, const int coords[], int *rank)< */
void F77_FUNC(pmpi_cart_rank,PMPI_CART_RANK)(int *comm, int *coords[], int *rank, int *ierr); 
void F77_FUNC(mpi_cart_rank,MPI_CART_RANK)(int *comm, int *coords[], int *rank, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_rank");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_cart_rank,PMPI_CART_RANK)(comm, coords, rank, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)coords);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[])< */
void F77_FUNC(pmpi_cart_coords,PMPI_CART_COORDS)(int *comm, int *rank, int *maxdims, int *coords[], int *ierr); 
void F77_FUNC(mpi_cart_coords,MPI_CART_COORDS)(int *comm, int *rank, int *maxdims, int *coords[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_coords");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_cart_coords,PMPI_CART_COORDS)(comm, rank, maxdims, coords, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*maxdims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)coords);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors)< */
void F77_FUNC(pmpi_graph_neighbors_count,PMPI_GRAPH_NEIGHBORS_COUNT)(int *comm, int *rank, int *nneighbors, int *ierr); 
void F77_FUNC(mpi_graph_neighbors_count,MPI_GRAPH_NEIGHBORS_COUNT)(int *comm, int *rank, int *nneighbors, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_neighbors_count");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_graph_neighbors_count,PMPI_GRAPH_NEIGHBORS_COUNT)(comm, rank, nneighbors, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)nneighbors);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int neighbors[])< */
void F77_FUNC(pmpi_graph_neighbors,PMPI_GRAPH_NEIGHBORS)(int *comm, int *rank, int *maxneighbors, int *neighbors[], int *ierr); 
void F77_FUNC(mpi_graph_neighbors,MPI_GRAPH_NEIGHBORS)(int *comm, int *rank, int *maxneighbors, int *neighbors[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_neighbors");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_graph_neighbors,PMPI_GRAPH_NEIGHBORS)(comm, rank, maxneighbors, neighbors, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*maxneighbors);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)neighbors);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest)< */
void F77_FUNC(pmpi_cart_shift,PMPI_CART_SHIFT)(int *comm, int *direction, int *disp, int *rank_source, int *rank_dest, int *ierr); 
void F77_FUNC(mpi_cart_shift,MPI_CART_SHIFT)(int *comm, int *direction, int *disp, int *rank_source, int *rank_dest, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_shift");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_cart_shift,PMPI_CART_SHIFT)(comm, direction, disp, rank_source, rank_dest, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*direction);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank_source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rank_dest);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm)< */
void F77_FUNC(pmpi_cart_sub,PMPI_CART_SUB)(int *comm, int *remain_dims[], int *newcomm, int *ierr); 
void F77_FUNC(mpi_cart_sub,MPI_CART_SUB)(int *comm, int *remain_dims[], int *newcomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_sub");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_cart_sub,PMPI_CART_SUB)(comm, remain_dims, newcomm, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)remain_dims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Cart_map(MPI_Comm comm, int ndims, const int dims[], const int periods[], int *newrank)< */
void F77_FUNC(pmpi_cart_map,PMPI_CART_MAP)(int *comm, int *ndims, int *dims[], int *periods[], int *newrank, int *ierr); 
void F77_FUNC(mpi_cart_map,MPI_CART_MAP)(int *comm, int *ndims, int *dims[], int *periods[], int *newrank, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Cart_map");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_cart_map,PMPI_CART_MAP)(comm, ndims, dims, periods, newrank, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)dims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)periods);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newrank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Graph_map(MPI_Comm comm, int nnodes, const int indx[], const int edges[], int *newrank)< */
void F77_FUNC(pmpi_graph_map,PMPI_GRAPH_MAP)(int *comm, int *nnodes, int *indx[], int *edges[], int *newrank, int *ierr); 
void F77_FUNC(mpi_graph_map,MPI_GRAPH_MAP)(int *comm, int *nnodes, int *indx[], int *edges[], int *newrank, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Graph_map");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_graph_map,PMPI_GRAPH_MAP)(comm, nnodes, indx, edges, newrank, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*nnodes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indx);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)edges);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newrank);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Get_processor_name(char *name, int *resultlen)< */
void F77_FUNC(pmpi_get_processor_name,PMPI_GET_PROCESSOR_NAME)(int *name, int *resultlen, int *ierr); 
void F77_FUNC(mpi_get_processor_name,MPI_GET_PROCESSOR_NAME)(int *name, int *resultlen, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_processor_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_get_processor_name,PMPI_GET_PROCESSOR_NAME)(name, resultlen, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Get_version(int *version, int *subversion)< */
void F77_FUNC(pmpi_get_version,PMPI_GET_VERSION)(int *version, int *subversion, int *ierr); 
void F77_FUNC(mpi_get_version,MPI_GET_VERSION)(int *version, int *subversion, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_version");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_get_version,PMPI_GET_VERSION)(version, subversion, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)version);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)subversion);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Get_library_version(char *version, int *resultlen)< */
void F77_FUNC(pmpi_get_library_version,PMPI_GET_LIBRARY_VERSION)(int *version, int *resultlen, int *ierr); 
void F77_FUNC(mpi_get_library_version,MPI_GET_LIBRARY_VERSION)(int *version, int *resultlen, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_library_version");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_get_library_version,PMPI_GET_LIBRARY_VERSION)(version, resultlen, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)version);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler)< */
void F77_FUNC(pmpi_errhandler_create,PMPI_ERRHANDLER_CREATE)(int *function, int *errhandler, int *ierr); 
void F77_FUNC(mpi_errhandler_create,MPI_ERRHANDLER_CREATE)(int *function, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Errhandler_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_errhandler_create,PMPI_ERRHANDLER_CREATE)(function, errhandler, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)function);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler)< */
void F77_FUNC(pmpi_errhandler_set,PMPI_ERRHANDLER_SET)(int *comm, int *errhandler, int *ierr); 
void F77_FUNC(mpi_errhandler_set,MPI_ERRHANDLER_SET)(int *comm, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Errhandler_set");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_errhandler_set,PMPI_ERRHANDLER_SET)(comm, errhandler, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler)< */
void F77_FUNC(pmpi_errhandler_get,PMPI_ERRHANDLER_GET)(int *comm, int *errhandler, int *ierr); 
void F77_FUNC(mpi_errhandler_get,MPI_ERRHANDLER_GET)(int *comm, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Errhandler_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_errhandler_get,PMPI_ERRHANDLER_GET)(comm, errhandler, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Errhandler_free(MPI_Errhandler *errhandler)< */
void F77_FUNC(pmpi_errhandler_free,PMPI_ERRHANDLER_FREE)(int *errhandler, int *ierr); 
void F77_FUNC(mpi_errhandler_free,MPI_ERRHANDLER_FREE)(int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Errhandler_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_errhandler_free,PMPI_ERRHANDLER_FREE)(errhandler, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Error_string(int errorcode, char *string, int *resultlen)< */
void F77_FUNC(pmpi_error_string,PMPI_ERROR_STRING)(int *errorcode, int *string, int *resultlen, int *ierr); 
void F77_FUNC(mpi_error_string,MPI_ERROR_STRING)(int *errorcode, int *string, int *resultlen, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Error_string");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_error_string,PMPI_ERROR_STRING)(errorcode, string, resultlen, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*errorcode);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)string);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Error_class(int errorcode, int *errorclass)< */
void F77_FUNC(pmpi_error_class,PMPI_ERROR_CLASS)(int *errorcode, int *errorclass, int *ierr); 
void F77_FUNC(mpi_error_class,MPI_ERROR_CLASS)(int *errorcode, int *errorclass, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Error_class");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_error_class,PMPI_ERROR_CLASS)(errorcode, errorclass, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*errorcode);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errorclass);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >double MPI_Wtime(void)< */
void F77_FUNC(pmpi_wtime,PMPI_WTIME)(int *ierr); 
void F77_FUNC(mpi_wtime,MPI_WTIME)(int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Wtime");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_wtime,PMPI_WTIME)(ierr);


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >double MPI_Wtick(void)< */
void F77_FUNC(pmpi_wtick,PMPI_WTICK)(int *ierr); 
void F77_FUNC(mpi_wtick,MPI_WTICK)(int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Wtick");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_wtick,PMPI_WTICK)(ierr);


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Init(int *argc, char ***argv)< */
void F77_FUNC(pmpi_init,PMPI_INIT)(int *argc, int *argv, int *ierr); 
void F77_FUNC(mpi_init,MPI_INIT)(int *argc, int *argv, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Init");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-");

  F77_FUNC(pmpi_init,PMPI_INIT)(argc, argv, ierr);


  {
    if(!mpi_initialized) mpi_initialize();
 print_banner(world_rank, "F77", "MPI_Init", world_size);
  }
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argc);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argv);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Finalize(void)< */
void F77_FUNC(pmpi_finalize,PMPI_FINALIZE)(int *ierr); 
void F77_FUNC(mpi_finalize,MPI_FINALIZE)(int *ierr) { 
  mpi_finalize();  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Finalize");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_finalize,PMPI_FINALIZE)(ierr);


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-\n");

  fputs((char*)curbuf, fp);; fclose(fp);
}

/* parsing >int MPI_Initialized(int *flag)< */
void F77_FUNC(pmpi_initialized,PMPI_INITIALIZED)(int *flag, int *ierr); 
void F77_FUNC(mpi_initialized,MPI_INITIALIZED)(int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Initialized");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_initialized,PMPI_INITIALIZED)(flag, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Abort(MPI_Comm comm, int errorcode)< */
void F77_FUNC(pmpi_abort,PMPI_ABORT)(int *comm, int *errorcode, int *ierr); 
void F77_FUNC(mpi_abort,MPI_ABORT)(int *comm, int *errorcode, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Abort");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_abort,PMPI_ABORT)(comm, errorcode, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_DUP_FN(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag)< */
void F77_FUNC(pmpi_dup_fn,PMPI_DUP_FN)(int *oldcomm, int *keyval, int *extra_state, int *attribute_val_in, int *attribute_val_out, int *flag, int *ierr); 
void F77_FUNC(mpi_dup_fn,MPI_DUP_FN)(int *oldcomm, int *keyval, int *extra_state, int *attribute_val_in, int *attribute_val_out, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_DUP_FN");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_dup_fn,PMPI_DUP_FN)(oldcomm, keyval, extra_state, attribute_val_in, attribute_val_out, flag, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *oldcomm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(oldcomm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(oldcomm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val_in);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val_out);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Close_port(const char *port_name)< */
void F77_FUNC(pmpi_close_port,PMPI_CLOSE_PORT)(int *port_name, int *ierr); 
void F77_FUNC(mpi_close_port,MPI_CLOSE_PORT)(int *port_name, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Close_port");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_close_port,PMPI_CLOSE_PORT)(port_name, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_accept(const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm)< */
void F77_FUNC(pmpi_comm_accept,PMPI_COMM_ACCEPT)(int *port_name, int *info, int *root, int *comm, int *newcomm, int *ierr); 
void F77_FUNC(mpi_comm_accept,MPI_COMM_ACCEPT)(int *port_name, int *info, int *root, int *comm, int *newcomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_accept");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_accept,PMPI_COMM_ACCEPT)(port_name, info, root, comm, newcomm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_connect(const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm)< */
void F77_FUNC(pmpi_comm_connect,PMPI_COMM_CONNECT)(int *port_name, int *info, int *root, int *comm, int *newcomm, int *ierr); 
void F77_FUNC(mpi_comm_connect,MPI_COMM_CONNECT)(int *port_name, int *info, int *root, int *comm, int *newcomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_connect");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_connect,PMPI_COMM_CONNECT)(port_name, info, root, comm, newcomm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_disconnect(MPI_Comm *comm)< */
void F77_FUNC(pmpi_comm_disconnect,PMPI_COMM_DISCONNECT)(int *comm, int *ierr); 
void F77_FUNC(mpi_comm_disconnect,MPI_COMM_DISCONNECT)(int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_disconnect");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_disconnect,PMPI_COMM_DISCONNECT)(comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_get_parent(MPI_Comm *parent)< */
void F77_FUNC(pmpi_comm_get_parent,PMPI_COMM_GET_PARENT)(int *parent, int *ierr); 
void F77_FUNC(mpi_comm_get_parent,MPI_COMM_GET_PARENT)(int *parent, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_parent");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_get_parent,PMPI_COMM_GET_PARENT)(parent, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)parent);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_join(int fd, MPI_Comm *intercomm)< */
void F77_FUNC(pmpi_comm_join,PMPI_COMM_JOIN)(int *fd, int *intercomm, int *ierr); 
void F77_FUNC(mpi_comm_join,MPI_COMM_JOIN)(int *fd, int *intercomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_join");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_join,PMPI_COMM_JOIN)(fd, intercomm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*fd);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)intercomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_spawn(const char *command, char *argv[], int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[])< */
void F77_FUNC(pmpi_comm_spawn,PMPI_COMM_SPAWN)(int *command, int *argv[], int *maxprocs, int *info, int *root, int *comm, int *intercomm, int *array_of_errcodes[], int *ierr); 
void F77_FUNC(mpi_comm_spawn,MPI_COMM_SPAWN)(int *command, int *argv[], int *maxprocs, int *info, int *root, int *comm, int *intercomm, int *array_of_errcodes[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_spawn");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_spawn,PMPI_COMM_SPAWN)(command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)command);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argv);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*maxprocs);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)intercomm);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_errcodes);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_spawn_multiple(int count, char *array_of_commands[], char **array_of_argv[], const int array_of_maxprocs[], const MPI_Info array_of_info[], int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[])< */
void F77_FUNC(pmpi_comm_spawn_multiple,PMPI_COMM_SPAWN_MULTIPLE)(int *count, int *array_of_commands[], int *array_of_argv[], int *array_of_maxprocs[], int *array_of_info[], int *root, int *comm, int *intercomm, int *array_of_errcodes[], int *ierr); 
void F77_FUNC(mpi_comm_spawn_multiple,MPI_COMM_SPAWN_MULTIPLE)(int *count, int *array_of_commands[], int *array_of_argv[], int *array_of_maxprocs[], int *array_of_info[], int *root, int *comm, int *intercomm, int *array_of_errcodes[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_spawn_multiple");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_spawn_multiple,PMPI_COMM_SPAWN_MULTIPLE)(count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_commands);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_argv);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_maxprocs);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)intercomm);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_errcodes);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Lookup_name(const char *service_name, MPI_Info info, char *port_name)< */
void F77_FUNC(pmpi_lookup_name,PMPI_LOOKUP_NAME)(int *service_name, int *info, int *port_name, int *ierr); 
void F77_FUNC(mpi_lookup_name,MPI_LOOKUP_NAME)(int *service_name, int *info, int *port_name, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Lookup_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_lookup_name,PMPI_LOOKUP_NAME)(service_name, info, port_name, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)service_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Open_port(MPI_Info info, char *port_name)< */
void F77_FUNC(pmpi_open_port,PMPI_OPEN_PORT)(int *info, int *port_name, int *ierr); 
void F77_FUNC(mpi_open_port,MPI_OPEN_PORT)(int *info, int *port_name, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Open_port");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_open_port,PMPI_OPEN_PORT)(info, port_name, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Publish_name(const char *service_name, MPI_Info info, const char *port_name)< */
void F77_FUNC(pmpi_publish_name,PMPI_PUBLISH_NAME)(int *service_name, int *info, int *port_name, int *ierr); 
void F77_FUNC(mpi_publish_name,MPI_PUBLISH_NAME)(int *service_name, int *info, int *port_name, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Publish_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_publish_name,PMPI_PUBLISH_NAME)(service_name, info, port_name, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)service_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Unpublish_name(const char *service_name, MPI_Info info, const char *port_name)< */
void F77_FUNC(pmpi_unpublish_name,PMPI_UNPUBLISH_NAME)(int *service_name, int *info, int *port_name, int *ierr); 
void F77_FUNC(mpi_unpublish_name,MPI_UNPUBLISH_NAME)(int *service_name, int *info, int *port_name, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Unpublish_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_unpublish_name,PMPI_UNPUBLISH_NAME)(service_name, info, port_name, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)service_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)port_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_set_info(MPI_Comm comm, MPI_Info info)< */
void F77_FUNC(pmpi_comm_set_info,PMPI_COMM_SET_INFO)(int *comm, int *info, int *ierr); 
void F77_FUNC(mpi_comm_set_info,MPI_COMM_SET_INFO)(int *comm, int *info, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_set_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_set_info,PMPI_COMM_SET_INFO)(comm, info, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_get_info(MPI_Comm comm, MPI_Info *info)< */
void F77_FUNC(pmpi_comm_get_info,PMPI_COMM_GET_INFO)(int *comm, int *info, int *ierr); 
void F77_FUNC(mpi_comm_get_info,MPI_COMM_GET_INFO)(int *comm, int *info, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_get_info,PMPI_COMM_GET_INFO)(comm, info, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)< */
void F77_FUNC(pmpi_accumulate,PMPI_ACCUMULATE)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *op, int *win, int *ierr); 
void F77_FUNC(mpi_accumulate,MPI_ACCUMULATE)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *op, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Accumulate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_accumulate,PMPI_ACCUMULATE)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*origin_count);
  
bufptr += printdatatype(MPI_Type_f2c(*origin_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (origin_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(origin_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_count);
  
bufptr += printdatatype(MPI_Type_f2c(*target_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (target_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(target_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)< */
void F77_FUNC(pmpi_get,PMPI_GET)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *win, int *ierr); 
void F77_FUNC(mpi_get,MPI_GET)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_get,PMPI_GET)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*origin_count);
  
bufptr += printdatatype(MPI_Type_f2c(*origin_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (origin_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(origin_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_count);
  
bufptr += printdatatype(MPI_Type_f2c(*target_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (target_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(target_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)< */
void F77_FUNC(pmpi_put,PMPI_PUT)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *win, int *ierr); 
void F77_FUNC(mpi_put,MPI_PUT)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Put");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_put,PMPI_PUT)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*origin_count);
  
bufptr += printdatatype(MPI_Type_f2c(*origin_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (origin_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(origin_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_count);
  
bufptr += printdatatype(MPI_Type_f2c(*target_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (target_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(target_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_complete(MPI_Win win)< */
void F77_FUNC(pmpi_win_complete,PMPI_WIN_COMPLETE)(int *win, int *ierr); 
void F77_FUNC(mpi_win_complete,MPI_WIN_COMPLETE)(int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_complete");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_complete,PMPI_WIN_COMPLETE)(win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win)< */
void F77_FUNC(pmpi_win_create,PMPI_WIN_CREATE)(int *base, MPI_Aint *size, int *disp_unit, int *info, int *comm, int *win, int *ierr); 
void F77_FUNC(mpi_win_create,MPI_WIN_CREATE)(int *base, MPI_Aint *size, int *disp_unit, int *info, int *comm, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_create,PMPI_WIN_CREATE)(base, size, disp_unit, info, comm, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)base);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*disp_unit);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_fence(int assert, MPI_Win win)< */
void F77_FUNC(pmpi_win_fence,PMPI_WIN_FENCE)(int *assert, int *win, int *ierr); 
void F77_FUNC(mpi_win_fence,MPI_WIN_FENCE)(int *assert, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_fence");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_fence,PMPI_WIN_FENCE)(assert, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_free(MPI_Win *win)< */
void F77_FUNC(pmpi_win_free,PMPI_WIN_FREE)(int *win, int *ierr); 
void F77_FUNC(mpi_win_free,MPI_WIN_FREE)(int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_free,PMPI_WIN_FREE)(win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_get_group(MPI_Win win, MPI_Group *group)< */
void F77_FUNC(pmpi_win_get_group,PMPI_WIN_GET_GROUP)(int *win, int *group, int *ierr); 
void F77_FUNC(mpi_win_get_group,MPI_WIN_GET_GROUP)(int *win, int *group, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_group");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_get_group,PMPI_WIN_GET_GROUP)(win, group, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)group);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win)< */
void F77_FUNC(pmpi_win_lock,PMPI_WIN_LOCK)(int *lock_type, int *rank, int *assert, int *win, int *ierr); 
void F77_FUNC(mpi_win_lock,MPI_WIN_LOCK)(int *lock_type, int *rank, int *assert, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_lock");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_lock,PMPI_WIN_LOCK)(lock_type, rank, assert, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*lock_type);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_post(MPI_Group group, int assert, MPI_Win win)< */
void F77_FUNC(pmpi_win_post,PMPI_WIN_POST)(int *group, int *assert, int *win, int *ierr); 
void F77_FUNC(mpi_win_post,MPI_WIN_POST)(int *group, int *assert, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_post");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_post,PMPI_WIN_POST)(group, assert, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_start(MPI_Group group, int assert, MPI_Win win)< */
void F77_FUNC(pmpi_win_start,PMPI_WIN_START)(int *group, int *assert, int *win, int *ierr); 
void F77_FUNC(mpi_win_start,MPI_WIN_START)(int *group, int *assert, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_start");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_start,PMPI_WIN_START)(group, assert, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *group);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_test(MPI_Win win, int *flag)< */
void F77_FUNC(pmpi_win_test,PMPI_WIN_TEST)(int *win, int *flag, int *ierr); 
void F77_FUNC(mpi_win_test,MPI_WIN_TEST)(int *win, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_test");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_test,PMPI_WIN_TEST)(win, flag, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_unlock(int rank, MPI_Win win)< */
void F77_FUNC(pmpi_win_unlock,PMPI_WIN_UNLOCK)(int *rank, int *win, int *ierr); 
void F77_FUNC(mpi_win_unlock,MPI_WIN_UNLOCK)(int *rank, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_unlock");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_unlock,PMPI_WIN_UNLOCK)(rank, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_wait(MPI_Win win)< */
void F77_FUNC(pmpi_win_wait,PMPI_WIN_WAIT)(int *win, int *ierr); 
void F77_FUNC(mpi_win_wait,MPI_WIN_WAIT)(int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_wait");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_wait,PMPI_WIN_WAIT)(win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)< */
void F77_FUNC(pmpi_win_allocate,PMPI_WIN_ALLOCATE)(MPI_Aint *size, int *disp_unit, int *info, int *comm, int *baseptr, int *win, int *ierr); 
void F77_FUNC(mpi_win_allocate,MPI_WIN_ALLOCATE)(MPI_Aint *size, int *disp_unit, int *info, int *comm, int *baseptr, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_allocate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_allocate,PMPI_WIN_ALLOCATE)(size, disp_unit, info, comm, baseptr, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*disp_unit);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)baseptr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)< */
void F77_FUNC(pmpi_win_allocate_shared,PMPI_WIN_ALLOCATE_SHARED)(MPI_Aint *size, int *disp_unit, int *info, int *comm, int *baseptr, int *win, int *ierr); 
void F77_FUNC(mpi_win_allocate_shared,MPI_WIN_ALLOCATE_SHARED)(MPI_Aint *size, int *disp_unit, int *info, int *comm, int *baseptr, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_allocate_shared");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_allocate_shared,PMPI_WIN_ALLOCATE_SHARED)(size, disp_unit, info, comm, baseptr, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*disp_unit);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)baseptr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr)< */
void F77_FUNC(pmpi_win_shared_query,PMPI_WIN_SHARED_QUERY)(int *win, int *rank, MPI_Aint *size, int *disp_unit, int *baseptr, int *ierr); 
void F77_FUNC(mpi_win_shared_query,MPI_WIN_SHARED_QUERY)(int *win, int *rank, MPI_Aint *size, int *disp_unit, int *baseptr, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_shared_query");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_shared_query,PMPI_WIN_SHARED_QUERY)(win, rank, size, disp_unit, baseptr, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)disp_unit);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)baseptr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win)< */
void F77_FUNC(pmpi_win_create_dynamic,PMPI_WIN_CREATE_DYNAMIC)(int *info, int *comm, int *win, int *ierr); 
void F77_FUNC(mpi_win_create_dynamic,MPI_WIN_CREATE_DYNAMIC)(int *info, int *comm, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_create_dynamic");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_create_dynamic,PMPI_WIN_CREATE_DYNAMIC)(info, comm, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size)< */
void F77_FUNC(pmpi_win_attach,PMPI_WIN_ATTACH)(int *win, int *base, MPI_Aint *size, int *ierr); 
void F77_FUNC(mpi_win_attach,MPI_WIN_ATTACH)(int *win, int *base, MPI_Aint *size, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_attach");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_attach,PMPI_WIN_ATTACH)(win, base, size, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)base);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_detach(MPI_Win win, const void *base)< */
void F77_FUNC(pmpi_win_detach,PMPI_WIN_DETACH)(int *win, int *base, int *ierr); 
void F77_FUNC(mpi_win_detach,MPI_WIN_DETACH)(int *win, int *base, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_detach");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_detach,PMPI_WIN_DETACH)(win, base, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)base);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_get_info(MPI_Win win, MPI_Info *info_used)< */
void F77_FUNC(pmpi_win_get_info,PMPI_WIN_GET_INFO)(int *win, int *info_used, int *ierr); 
void F77_FUNC(mpi_win_get_info,MPI_WIN_GET_INFO)(int *win, int *info_used, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_get_info,PMPI_WIN_GET_INFO)(win, info_used, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info_used);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_set_info(MPI_Win win, MPI_Info info)< */
void F77_FUNC(pmpi_win_set_info,PMPI_WIN_SET_INFO)(int *win, int *info, int *ierr); 
void F77_FUNC(mpi_win_set_info,MPI_WIN_SET_INFO)(int *win, int *info, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_set_info");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_set_info,PMPI_WIN_SET_INFO)(win, info, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Get_accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, void *result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)< */
void F77_FUNC(pmpi_get_accumulate,PMPI_GET_ACCUMULATE)(int *origin_addr, int *origin_count, int *origin_datatype, int *result_addr, int *result_count, int *result_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *op, int *win, int *ierr); 
void F77_FUNC(mpi_get_accumulate,MPI_GET_ACCUMULATE)(int *origin_addr, int *origin_count, int *origin_datatype, int *result_addr, int *result_count, int *result_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *op, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_accumulate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_get_accumulate,PMPI_GET_ACCUMULATE)(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*origin_count);
  
bufptr += printdatatype(MPI_Type_f2c(*origin_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (origin_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(origin_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*result_count);
  
bufptr += printdatatype(MPI_Type_f2c(*result_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (result_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(result_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_count);
  
bufptr += printdatatype(MPI_Type_f2c(*target_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (target_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(target_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Fetch_and_op(const void *origin_addr, void *result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win)< */
void F77_FUNC(pmpi_fetch_and_op,PMPI_FETCH_AND_OP)(int *origin_addr, int *result_addr, int *datatype, int *target_rank, MPI_Aint *target_disp, int *op, int *win, int *ierr); 
void F77_FUNC(mpi_fetch_and_op,MPI_FETCH_AND_OP)(int *origin_addr, int *result_addr, int *datatype, int *target_rank, MPI_Aint *target_disp, int *op, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Fetch_and_op");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_fetch_and_op,PMPI_FETCH_AND_OP)(origin_addr, result_addr, datatype, target_rank, target_disp, op, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result_addr);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Compare_and_swap(const void *origin_addr, const void *compare_addr, void *result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Win win)< */
void F77_FUNC(pmpi_compare_and_swap,PMPI_COMPARE_AND_SWAP)(int *origin_addr, int *compare_addr, int *result_addr, int *datatype, int *target_rank, MPI_Aint *target_disp, int *win, int *ierr); 
void F77_FUNC(mpi_compare_and_swap,MPI_COMPARE_AND_SWAP)(int *origin_addr, int *compare_addr, int *result_addr, int *datatype, int *target_rank, MPI_Aint *target_disp, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Compare_and_swap");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_compare_and_swap,PMPI_COMPARE_AND_SWAP)(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)compare_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result_addr);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Rput(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request *request)< */
void F77_FUNC(pmpi_rput,PMPI_RPUT)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *win, int *request, int *ierr); 
void F77_FUNC(mpi_rput,MPI_RPUT)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *win, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rput");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_rput,PMPI_RPUT)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*origin_count);
  
bufptr += printdatatype(MPI_Type_f2c(*origin_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (origin_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(origin_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_count);
  
bufptr += printdatatype(MPI_Type_f2c(*target_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (target_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(target_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Rget(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request *request)< */
void F77_FUNC(pmpi_rget,PMPI_RGET)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *win, int *request, int *ierr); 
void F77_FUNC(mpi_rget,MPI_RGET)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *win, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rget");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_rget,PMPI_RGET)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*origin_count);
  
bufptr += printdatatype(MPI_Type_f2c(*origin_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (origin_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(origin_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_count);
  
bufptr += printdatatype(MPI_Type_f2c(*target_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (target_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(target_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Raccumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request *request)< */
void F77_FUNC(pmpi_raccumulate,PMPI_RACCUMULATE)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *op, int *win, int *request, int *ierr); 
void F77_FUNC(mpi_raccumulate,MPI_RACCUMULATE)(int *origin_addr, int *origin_count, int *origin_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *op, int *win, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Raccumulate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_raccumulate,PMPI_RACCUMULATE)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*origin_count);
  
bufptr += printdatatype(MPI_Type_f2c(*origin_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (origin_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(origin_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_count);
  
bufptr += printdatatype(MPI_Type_f2c(*target_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (target_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(target_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Rget_accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, void *result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request *request)< */
void F77_FUNC(pmpi_rget_accumulate,PMPI_RGET_ACCUMULATE)(int *origin_addr, int *origin_count, int *origin_datatype, int *result_addr, int *result_count, int *result_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *op, int *win, int *request, int *ierr); 
void F77_FUNC(mpi_rget_accumulate,MPI_RGET_ACCUMULATE)(int *origin_addr, int *origin_count, int *origin_datatype, int *result_addr, int *result_count, int *result_datatype, int *target_rank, MPI_Aint *target_disp, int *target_count, int *target_datatype, int *op, int *win, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Rget_accumulate");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_rget_accumulate,PMPI_RGET_ACCUMULATE)(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)origin_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*origin_count);
  
bufptr += printdatatype(MPI_Type_f2c(*origin_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (origin_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(origin_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)result_addr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*result_count);
  
bufptr += printdatatype(MPI_Type_f2c(*result_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (result_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(result_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_disp);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*target_count);
  
bufptr += printdatatype(MPI_Type_f2c(*target_datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (target_datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(target_datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_lock_all(int assert, MPI_Win win)< */
void F77_FUNC(pmpi_win_lock_all,PMPI_WIN_LOCK_ALL)(int *assert, int *win, int *ierr); 
void F77_FUNC(mpi_win_lock_all,MPI_WIN_LOCK_ALL)(int *assert, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_lock_all");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_lock_all,PMPI_WIN_LOCK_ALL)(assert, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*assert);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_unlock_all(MPI_Win win)< */
void F77_FUNC(pmpi_win_unlock_all,PMPI_WIN_UNLOCK_ALL)(int *win, int *ierr); 
void F77_FUNC(mpi_win_unlock_all,MPI_WIN_UNLOCK_ALL)(int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_unlock_all");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_unlock_all,PMPI_WIN_UNLOCK_ALL)(win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_flush(int rank, MPI_Win win)< */
void F77_FUNC(pmpi_win_flush,PMPI_WIN_FLUSH)(int *rank, int *win, int *ierr); 
void F77_FUNC(mpi_win_flush,MPI_WIN_FLUSH)(int *rank, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_flush");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_flush,PMPI_WIN_FLUSH)(rank, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_flush_all(MPI_Win win)< */
void F77_FUNC(pmpi_win_flush_all,PMPI_WIN_FLUSH_ALL)(int *win, int *ierr); 
void F77_FUNC(mpi_win_flush_all,MPI_WIN_FLUSH_ALL)(int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_flush_all");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_flush_all,PMPI_WIN_FLUSH_ALL)(win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_flush_local(int rank, MPI_Win win)< */
void F77_FUNC(pmpi_win_flush_local,PMPI_WIN_FLUSH_LOCAL)(int *rank, int *win, int *ierr); 
void F77_FUNC(mpi_win_flush_local,MPI_WIN_FLUSH_LOCAL)(int *rank, int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_flush_local");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_flush_local,PMPI_WIN_FLUSH_LOCAL)(rank, win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_flush_local_all(MPI_Win win)< */
void F77_FUNC(pmpi_win_flush_local_all,PMPI_WIN_FLUSH_LOCAL_ALL)(int *win, int *ierr); 
void F77_FUNC(mpi_win_flush_local_all,MPI_WIN_FLUSH_LOCAL_ALL)(int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_flush_local_all");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_flush_local_all,PMPI_WIN_FLUSH_LOCAL_ALL)(win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_sync(MPI_Win win)< */
void F77_FUNC(pmpi_win_sync,PMPI_WIN_SYNC)(int *win, int *ierr); 
void F77_FUNC(mpi_win_sync,MPI_WIN_SYNC)(int *win, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_sync");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_sync,PMPI_WIN_SYNC)(win, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Add_error_class(int *errorclass)< */
void F77_FUNC(pmpi_add_error_class,PMPI_ADD_ERROR_CLASS)(int *errorclass, int *ierr); 
void F77_FUNC(mpi_add_error_class,MPI_ADD_ERROR_CLASS)(int *errorclass, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Add_error_class");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_add_error_class,PMPI_ADD_ERROR_CLASS)(errorclass, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errorclass);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Add_error_code(int errorclass, int *errorcode)< */
void F77_FUNC(pmpi_add_error_code,PMPI_ADD_ERROR_CODE)(int *errorclass, int *errorcode, int *ierr); 
void F77_FUNC(mpi_add_error_code,MPI_ADD_ERROR_CODE)(int *errorclass, int *errorcode, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Add_error_code");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_add_error_code,PMPI_ADD_ERROR_CODE)(errorclass, errorcode, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*errorclass);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Add_error_string(int errorcode, const char *string)< */
void F77_FUNC(pmpi_add_error_string,PMPI_ADD_ERROR_STRING)(int *errorcode, int *string, int *ierr); 
void F77_FUNC(mpi_add_error_string,MPI_ADD_ERROR_STRING)(int *errorcode, int *string, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Add_error_string");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_add_error_string,PMPI_ADD_ERROR_STRING)(errorcode, string, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*errorcode);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)string);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_call_errhandler(MPI_Comm comm, int errorcode)< */
void F77_FUNC(pmpi_comm_call_errhandler,PMPI_COMM_CALL_ERRHANDLER)(int *comm, int *errorcode, int *ierr); 
void F77_FUNC(mpi_comm_call_errhandler,MPI_COMM_CALL_ERRHANDLER)(int *comm, int *errorcode, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_call_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_call_errhandler,PMPI_COMM_CALL_ERRHANDLER)(comm, errorcode, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, int *comm_keyval, void *extra_state)< */
void F77_FUNC(pmpi_comm_create_keyval,PMPI_COMM_CREATE_KEYVAL)(int *comm_copy_attr_fn, int *comm_delete_attr_fn, int *comm_keyval, int *extra_state, int *ierr); 
void F77_FUNC(mpi_comm_create_keyval,MPI_COMM_CREATE_KEYVAL)(int *comm_copy_attr_fn, int *comm_delete_attr_fn, int *comm_keyval, int *extra_state, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_create_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_create_keyval,PMPI_COMM_CREATE_KEYVAL)(comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_copy_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_delete_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval)< */
void F77_FUNC(pmpi_comm_delete_attr,PMPI_COMM_DELETE_ATTR)(int *comm, int *comm_keyval, int *ierr); 
void F77_FUNC(mpi_comm_delete_attr,MPI_COMM_DELETE_ATTR)(int *comm, int *comm_keyval, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_delete_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_delete_attr,PMPI_COMM_DELETE_ATTR)(comm, comm_keyval, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*comm_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_free_keyval(int *comm_keyval)< */
void F77_FUNC(pmpi_comm_free_keyval,PMPI_COMM_FREE_KEYVAL)(int *comm_keyval, int *ierr); 
void F77_FUNC(mpi_comm_free_keyval,MPI_COMM_FREE_KEYVAL)(int *comm_keyval, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_free_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_free_keyval,PMPI_COMM_FREE_KEYVAL)(comm_keyval, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag)< */
void F77_FUNC(pmpi_comm_get_attr,PMPI_COMM_GET_ATTR)(int *comm, int *comm_keyval, int *attribute_val, int *flag, int *ierr); 
void F77_FUNC(mpi_comm_get_attr,MPI_COMM_GET_ATTR)(int *comm, int *comm_keyval, int *attribute_val, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_get_attr,PMPI_COMM_GET_ATTR)(comm, comm_keyval, attribute_val, flag, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*comm_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen)< */
void F77_FUNC(pmpi_comm_get_name,PMPI_COMM_GET_NAME)(int *comm, int *comm_name, int *resultlen, int *ierr); 
void F77_FUNC(mpi_comm_get_name,MPI_COMM_GET_NAME)(int *comm, int *comm_name, int *resultlen, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_get_name,PMPI_COMM_GET_NAME)(comm, comm_name, resultlen, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val)< */
void F77_FUNC(pmpi_comm_set_attr,PMPI_COMM_SET_ATTR)(int *comm, int *comm_keyval, int *attribute_val, int *ierr); 
void F77_FUNC(mpi_comm_set_attr,MPI_COMM_SET_ATTR)(int *comm, int *comm_keyval, int *attribute_val, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_set_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_set_attr,PMPI_COMM_SET_ATTR)(comm, comm_keyval, attribute_val, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*comm_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_set_name(MPI_Comm comm, const char *comm_name)< */
void F77_FUNC(pmpi_comm_set_name,PMPI_COMM_SET_NAME)(int *comm, int *comm_name, int *ierr); 
void F77_FUNC(mpi_comm_set_name,MPI_COMM_SET_NAME)(int *comm, int *comm_name, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_set_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_set_name,PMPI_COMM_SET_NAME)(comm, comm_name, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_File_call_errhandler(MPI_File fh, int errorcode)< */
void F77_FUNC(pmpi_file_call_errhandler,PMPI_FILE_CALL_ERRHANDLER)(int *fh, int *errorcode, int *ierr); 
void F77_FUNC(mpi_file_call_errhandler,MPI_FILE_CALL_ERRHANDLER)(int *fh, int *errorcode, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_File_call_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_file_call_errhandler,PMPI_FILE_CALL_ERRHANDLER)(fh, errorcode, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *fh);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Grequest_complete(MPI_Request request)< */
void F77_FUNC(pmpi_grequest_complete,PMPI_GREQUEST_COMPLETE)(int *request, int *ierr); 
void F77_FUNC(mpi_grequest_complete,MPI_GREQUEST_COMPLETE)(int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Grequest_complete");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_grequest_complete,PMPI_GREQUEST_COMPLETE)(request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Grequest_start(MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn, void *extra_state, MPI_Request *request)< */
void F77_FUNC(pmpi_grequest_start,PMPI_GREQUEST_START)(int *query_fn, int *free_fn, int *cancel_fn, int *extra_state, int *request, int *ierr); 
void F77_FUNC(mpi_grequest_start,MPI_GREQUEST_START)(int *query_fn, int *free_fn, int *cancel_fn, int *extra_state, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Grequest_start");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_grequest_start,PMPI_GREQUEST_START)(query_fn, free_fn, cancel_fn, extra_state, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)query_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)free_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)cancel_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)< */
void F77_FUNC(pmpi_init_thread,PMPI_INIT_THREAD)(int *argc, int *argv, int *required, int *provided, int *ierr); 
void F77_FUNC(mpi_init_thread,MPI_INIT_THREAD)(int *argc, int *argv, int *required, int *provided, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Init_thread");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-");

  F77_FUNC(pmpi_init_thread,PMPI_INIT_THREAD)(argc, argv, required, provided, ierr);


  {
    if(!mpi_initialized) mpi_initialize();
 print_banner(world_rank, "F77", "MPI_Init_thread", world_size);
  }
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argc);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)argv);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*required);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)provided);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Is_thread_main(int *flag)< */
void F77_FUNC(pmpi_is_thread_main,PMPI_IS_THREAD_MAIN)(int *flag, int *ierr); 
void F77_FUNC(mpi_is_thread_main,MPI_IS_THREAD_MAIN)(int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Is_thread_main");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_is_thread_main,PMPI_IS_THREAD_MAIN)(flag, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Query_thread(int *provided)< */
void F77_FUNC(pmpi_query_thread,PMPI_QUERY_THREAD)(int *provided, int *ierr); 
void F77_FUNC(mpi_query_thread,MPI_QUERY_THREAD)(int *provided, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Query_thread");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_query_thread,PMPI_QUERY_THREAD)(provided, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)provided);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Status_set_cancelled(MPI_Status *status, int flag)< */
void F77_FUNC(pmpi_status_set_cancelled,PMPI_STATUS_SET_CANCELLED)(int *status, int *flag, int *ierr); 
void F77_FUNC(mpi_status_set_cancelled,MPI_STATUS_SET_CANCELLED)(int *status, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Status_set_cancelled");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_status_set_cancelled,PMPI_STATUS_SET_CANCELLED)(status, flag, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count)< */
void F77_FUNC(pmpi_status_set_elements,PMPI_STATUS_SET_ELEMENTS)(int *status, int *datatype, int *count, int *ierr); 
void F77_FUNC(mpi_status_set_elements,MPI_STATUS_SET_ELEMENTS)(int *status, int *datatype, int *count, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Status_set_elements");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_status_set_elements,PMPI_STATUS_SET_ELEMENTS)(status, datatype, count, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, int *type_keyval, void *extra_state)< */
void F77_FUNC(pmpi_type_create_keyval,PMPI_TYPE_CREATE_KEYVAL)(int *type_copy_attr_fn, int *type_delete_attr_fn, int *type_keyval, int *extra_state, int *ierr); 
void F77_FUNC(mpi_type_create_keyval,MPI_TYPE_CREATE_KEYVAL)(int *type_copy_attr_fn, int *type_delete_attr_fn, int *type_keyval, int *extra_state, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_keyval,PMPI_TYPE_CREATE_KEYVAL)(type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_copy_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_delete_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_delete_attr(MPI_Datatype datatype, int type_keyval)< */
void F77_FUNC(pmpi_type_delete_attr,PMPI_TYPE_DELETE_ATTR)(int *datatype, int *type_keyval, int *ierr); 
void F77_FUNC(mpi_type_delete_attr,MPI_TYPE_DELETE_ATTR)(int *datatype, int *type_keyval, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_delete_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_delete_attr,PMPI_TYPE_DELETE_ATTR)(datatype, type_keyval, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*type_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_dup(MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_dup,PMPI_TYPE_DUP)(int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_dup,MPI_TYPE_DUP)(int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_dup");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_dup,PMPI_TYPE_DUP)(oldtype, newtype, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_free_keyval(int *type_keyval)< */
void F77_FUNC(pmpi_type_free_keyval,PMPI_TYPE_FREE_KEYVAL)(int *type_keyval, int *ierr); 
void F77_FUNC(mpi_type_free_keyval,MPI_TYPE_FREE_KEYVAL)(int *type_keyval, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_free_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_free_keyval,PMPI_TYPE_FREE_KEYVAL)(type_keyval, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_get_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val, int *flag)< */
void F77_FUNC(pmpi_type_get_attr,PMPI_TYPE_GET_ATTR)(int *datatype, int *type_keyval, int *attribute_val, int *flag, int *ierr); 
void F77_FUNC(mpi_type_get_attr,MPI_TYPE_GET_ATTR)(int *datatype, int *type_keyval, int *attribute_val, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_get_attr,PMPI_TYPE_GET_ATTR)(datatype, type_keyval, attribute_val, flag, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*type_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[])< */
void F77_FUNC(pmpi_type_get_contents,PMPI_TYPE_GET_CONTENTS)(int *datatype, int *max_integers, int *max_addresses, int *max_datatypes, int *array_of_integers[], MPI_Aint *array_of_addresses[], int *array_of_datatypes[], int *ierr); 
void F77_FUNC(mpi_type_get_contents,MPI_TYPE_GET_CONTENTS)(int *datatype, int *max_integers, int *max_addresses, int *max_datatypes, int *array_of_integers[], MPI_Aint *array_of_addresses[], int *array_of_datatypes[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_contents");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_get_contents,PMPI_TYPE_GET_CONTENTS)(datatype, max_integers, max_addresses, max_datatypes, array_of_integers, array_of_addresses, array_of_datatypes, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*max_integers);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*max_addresses);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*max_datatypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_integers);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_addresses);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_datatypes);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_get_envelope(MPI_Datatype datatype, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner)< */
void F77_FUNC(pmpi_type_get_envelope,PMPI_TYPE_GET_ENVELOPE)(int *datatype, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner, int *ierr); 
void F77_FUNC(mpi_type_get_envelope,MPI_TYPE_GET_ENVELOPE)(int *datatype, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_envelope");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_get_envelope,PMPI_TYPE_GET_ENVELOPE)(datatype, num_integers, num_addresses, num_datatypes, combiner, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)num_integers);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)num_addresses);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)num_datatypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)combiner);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_get_name(MPI_Datatype datatype, char *type_name, int *resultlen)< */
void F77_FUNC(pmpi_type_get_name,PMPI_TYPE_GET_NAME)(int *datatype, int *type_name, int *resultlen, int *ierr); 
void F77_FUNC(mpi_type_get_name,MPI_TYPE_GET_NAME)(int *datatype, int *type_name, int *resultlen, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_get_name,PMPI_TYPE_GET_NAME)(datatype, type_name, resultlen, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_set_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val)< */
void F77_FUNC(pmpi_type_set_attr,PMPI_TYPE_SET_ATTR)(int *datatype, int *type_keyval, int *attribute_val, int *ierr); 
void F77_FUNC(mpi_type_set_attr,MPI_TYPE_SET_ATTR)(int *datatype, int *type_keyval, int *attribute_val, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_set_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_set_attr,PMPI_TYPE_SET_ATTR)(datatype, type_keyval, attribute_val, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*type_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_set_name(MPI_Datatype datatype, const char *type_name)< */
void F77_FUNC(pmpi_type_set_name,PMPI_TYPE_SET_NAME)(int *datatype, int *type_name, int *ierr); 
void F77_FUNC(mpi_type_set_name,MPI_TYPE_SET_NAME)(int *datatype, int *type_name, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_set_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_set_name,PMPI_TYPE_SET_NAME)(datatype, type_name, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)type_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_match_size(int typeclass, int size, MPI_Datatype *datatype)< */
void F77_FUNC(pmpi_type_match_size,PMPI_TYPE_MATCH_SIZE)(int *typeclass, int *size, int *datatype, int *ierr); 
void F77_FUNC(mpi_type_match_size,MPI_TYPE_MATCH_SIZE)(int *typeclass, int *size, int *datatype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_match_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_match_size,PMPI_TYPE_MATCH_SIZE)(typeclass, size, datatype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*typeclass);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datatype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_call_errhandler(MPI_Win win, int errorcode)< */
void F77_FUNC(pmpi_win_call_errhandler,PMPI_WIN_CALL_ERRHANDLER)(int *win, int *errorcode, int *ierr); 
void F77_FUNC(mpi_win_call_errhandler,MPI_WIN_CALL_ERRHANDLER)(int *win, int *errorcode, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_call_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_call_errhandler,PMPI_WIN_CALL_ERRHANDLER)(win, errorcode, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*errorcode);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, int *win_keyval, void *extra_state)< */
void F77_FUNC(pmpi_win_create_keyval,PMPI_WIN_CREATE_KEYVAL)(int *win_copy_attr_fn, int *win_delete_attr_fn, int *win_keyval, int *extra_state, int *ierr); 
void F77_FUNC(mpi_win_create_keyval,MPI_WIN_CREATE_KEYVAL)(int *win_copy_attr_fn, int *win_delete_attr_fn, int *win_keyval, int *extra_state, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_create_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_create_keyval,PMPI_WIN_CREATE_KEYVAL)(win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_copy_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_delete_attr_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extra_state);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_delete_attr(MPI_Win win, int win_keyval)< */
void F77_FUNC(pmpi_win_delete_attr,PMPI_WIN_DELETE_ATTR)(int *win, int *win_keyval, int *ierr); 
void F77_FUNC(mpi_win_delete_attr,MPI_WIN_DELETE_ATTR)(int *win, int *win_keyval, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_delete_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_delete_attr,PMPI_WIN_DELETE_ATTR)(win, win_keyval, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*win_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_free_keyval(int *win_keyval)< */
void F77_FUNC(pmpi_win_free_keyval,PMPI_WIN_FREE_KEYVAL)(int *win_keyval, int *ierr); 
void F77_FUNC(mpi_win_free_keyval,MPI_WIN_FREE_KEYVAL)(int *win_keyval, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_free_keyval");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_free_keyval,PMPI_WIN_FREE_KEYVAL)(win_keyval, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_keyval);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag)< */
void F77_FUNC(pmpi_win_get_attr,PMPI_WIN_GET_ATTR)(int *win, int *win_keyval, int *attribute_val, int *flag, int *ierr); 
void F77_FUNC(mpi_win_get_attr,MPI_WIN_GET_ATTR)(int *win, int *win_keyval, int *attribute_val, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_get_attr,PMPI_WIN_GET_ATTR)(win, win_keyval, attribute_val, flag, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*win_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_get_name(MPI_Win win, char *win_name, int *resultlen)< */
void F77_FUNC(pmpi_win_get_name,PMPI_WIN_GET_NAME)(int *win, int *win_name, int *resultlen, int *ierr); 
void F77_FUNC(mpi_win_get_name,MPI_WIN_GET_NAME)(int *win, int *win_name, int *resultlen, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_get_name,PMPI_WIN_GET_NAME)(win, win_name, resultlen, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_name);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)resultlen);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val)< */
void F77_FUNC(pmpi_win_set_attr,PMPI_WIN_SET_ATTR)(int *win, int *win_keyval, int *attribute_val, int *ierr); 
void F77_FUNC(mpi_win_set_attr,MPI_WIN_SET_ATTR)(int *win, int *win_keyval, int *attribute_val, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_set_attr");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_set_attr,PMPI_WIN_SET_ATTR)(win, win_keyval, attribute_val, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*win_keyval);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)attribute_val);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_set_name(MPI_Win win, const char *win_name)< */
void F77_FUNC(pmpi_win_set_name,PMPI_WIN_SET_NAME)(int *win, int *win_name, int *ierr); 
void F77_FUNC(mpi_win_set_name,MPI_WIN_SET_NAME)(int *win, int *win_name, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_set_name");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_set_name,PMPI_WIN_SET_NAME)(win, win_name, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_name);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr)< */
void F77_FUNC(pmpi_alloc_mem,PMPI_ALLOC_MEM)(MPI_Aint *size, int *info, int *baseptr, int *ierr); 
void F77_FUNC(mpi_alloc_mem,MPI_ALLOC_MEM)(MPI_Aint *size, int *info, int *baseptr, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Alloc_mem");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_alloc_mem,PMPI_ALLOC_MEM)(size, info, baseptr, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)baseptr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_create_errhandler(MPI_Comm_errhandler_function *comm_errhandler_fn, MPI_Errhandler *errhandler)< */
void F77_FUNC(pmpi_comm_create_errhandler,PMPI_COMM_CREATE_ERRHANDLER)(int *comm_errhandler_fn, int *errhandler, int *ierr); 
void F77_FUNC(mpi_comm_create_errhandler,MPI_COMM_CREATE_ERRHANDLER)(int *comm_errhandler_fn, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_create_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_create_errhandler,PMPI_COMM_CREATE_ERRHANDLER)(comm_errhandler_fn, errhandler, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_errhandler_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler)< */
void F77_FUNC(pmpi_comm_get_errhandler,PMPI_COMM_GET_ERRHANDLER)(int *comm, int *errhandler, int *ierr); 
void F77_FUNC(mpi_comm_get_errhandler,MPI_COMM_GET_ERRHANDLER)(int *comm, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_get_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_get_errhandler,PMPI_COMM_GET_ERRHANDLER)(comm, errhandler, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler)< */
void F77_FUNC(pmpi_comm_set_errhandler,PMPI_COMM_SET_ERRHANDLER)(int *comm, int *errhandler, int *ierr); 
void F77_FUNC(mpi_comm_set_errhandler,MPI_COMM_SET_ERRHANDLER)(int *comm, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_set_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_set_errhandler,PMPI_COMM_SET_ERRHANDLER)(comm, errhandler, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_File_create_errhandler(MPI_File_errhandler_function *file_errhandler_fn, MPI_Errhandler *errhandler)< */
void F77_FUNC(pmpi_file_create_errhandler,PMPI_FILE_CREATE_ERRHANDLER)(int *file_errhandler_fn, int *errhandler, int *ierr); 
void F77_FUNC(mpi_file_create_errhandler,MPI_FILE_CREATE_ERRHANDLER)(int *file_errhandler_fn, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_File_create_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_file_create_errhandler,PMPI_FILE_CREATE_ERRHANDLER)(file_errhandler_fn, errhandler, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)file_errhandler_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler)< */
void F77_FUNC(pmpi_file_get_errhandler,PMPI_FILE_GET_ERRHANDLER)(int *file, int *errhandler, int *ierr); 
void F77_FUNC(mpi_file_get_errhandler,MPI_FILE_GET_ERRHANDLER)(int *file, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_File_get_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_file_get_errhandler,PMPI_FILE_GET_ERRHANDLER)(file, errhandler, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *file);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler)< */
void F77_FUNC(pmpi_file_set_errhandler,PMPI_FILE_SET_ERRHANDLER)(int *file, int *errhandler, int *ierr); 
void F77_FUNC(mpi_file_set_errhandler,MPI_FILE_SET_ERRHANDLER)(int *file, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_File_set_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_file_set_errhandler,PMPI_FILE_SET_ERRHANDLER)(file, errhandler, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *file);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Finalized(int *flag)< */
void F77_FUNC(pmpi_finalized,PMPI_FINALIZED)(int *flag, int *ierr); 
void F77_FUNC(mpi_finalized,MPI_FINALIZED)(int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Finalized");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_finalized,PMPI_FINALIZED)(flag, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Free_mem(void *base)< */
void F77_FUNC(pmpi_free_mem,PMPI_FREE_MEM)(int *base, int *ierr); 
void F77_FUNC(mpi_free_mem,MPI_FREE_MEM)(int *base, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Free_mem");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_free_mem,PMPI_FREE_MEM)(base, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)base);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Get_address(const void *location, MPI_Aint *address)< */
void F77_FUNC(pmpi_get_address,PMPI_GET_ADDRESS)(int *location, MPI_Aint *address, int *ierr); 
void F77_FUNC(mpi_get_address,MPI_GET_ADDRESS)(int *location, MPI_Aint *address, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Get_address");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_get_address,PMPI_GET_ADDRESS)(location, address, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)location);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)address);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Info_create(MPI_Info *info)< */
void F77_FUNC(pmpi_info_create,PMPI_INFO_CREATE)(int *info, int *ierr); 
void F77_FUNC(mpi_info_create,MPI_INFO_CREATE)(int *info, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_info_create,PMPI_INFO_CREATE)(info, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Info_delete(MPI_Info info, const char *key)< */
void F77_FUNC(pmpi_info_delete,PMPI_INFO_DELETE)(int *info, int *key, int *ierr); 
void F77_FUNC(mpi_info_delete,MPI_INFO_DELETE)(int *info, int *key, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_delete");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_info_delete,PMPI_INFO_DELETE)(info, key, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Info_dup(MPI_Info info, MPI_Info *newinfo)< */
void F77_FUNC(pmpi_info_dup,PMPI_INFO_DUP)(int *info, int *newinfo, int *ierr); 
void F77_FUNC(mpi_info_dup,MPI_INFO_DUP)(int *info, int *newinfo, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_dup");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_info_dup,PMPI_INFO_DUP)(info, newinfo, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newinfo);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Info_free(MPI_Info *info)< */
void F77_FUNC(pmpi_info_free,PMPI_INFO_FREE)(int *info, int *ierr); 
void F77_FUNC(mpi_info_free,MPI_INFO_FREE)(int *info, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_free");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_info_free,PMPI_INFO_FREE)(info, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)info);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag)< */
void F77_FUNC(pmpi_info_get,PMPI_INFO_GET)(int *info, int *key, int *valuelen, int *value, int *flag, int *ierr); 
void F77_FUNC(mpi_info_get,MPI_INFO_GET)(int *info, int *key, int *valuelen, int *value, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_get");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_info_get,PMPI_INFO_GET)(info, key, valuelen, value, flag, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*valuelen);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)value);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Info_get_nkeys(MPI_Info info, int *nkeys)< */
void F77_FUNC(pmpi_info_get_nkeys,PMPI_INFO_GET_NKEYS)(int *info, int *nkeys, int *ierr); 
void F77_FUNC(mpi_info_get_nkeys,MPI_INFO_GET_NKEYS)(int *info, int *nkeys, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_get_nkeys");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_info_get_nkeys,PMPI_INFO_GET_NKEYS)(info, nkeys, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)nkeys);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Info_get_nthkey(MPI_Info info, int n, char *key)< */
void F77_FUNC(pmpi_info_get_nthkey,PMPI_INFO_GET_NTHKEY)(int *info, int *n, int *key, int *ierr); 
void F77_FUNC(mpi_info_get_nthkey,MPI_INFO_GET_NTHKEY)(int *info, int *n, int *key, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_get_nthkey");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_info_get_nthkey,PMPI_INFO_GET_NTHKEY)(info, n, key, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag)< */
void F77_FUNC(pmpi_info_get_valuelen,PMPI_INFO_GET_VALUELEN)(int *info, int *key, int *valuelen, int *flag, int *ierr); 
void F77_FUNC(mpi_info_get_valuelen,MPI_INFO_GET_VALUELEN)(int *info, int *key, int *valuelen, int *flag, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_get_valuelen");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_info_get_valuelen,PMPI_INFO_GET_VALUELEN)(info, key, valuelen, flag, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)valuelen);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Info_set(MPI_Info info, const char *key, const char *value)< */
void F77_FUNC(pmpi_info_set,PMPI_INFO_SET)(int *info, int *key, int *value, int *ierr); 
void F77_FUNC(mpi_info_set,MPI_INFO_SET)(int *info, int *key, int *value, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Info_set");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_info_set,PMPI_INFO_SET)(info, key, value, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)value);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Pack_external(const char datarep[], const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position)< */
void F77_FUNC(pmpi_pack_external,PMPI_PACK_EXTERNAL)(int *datarep[], int *inbuf, int *incount, int *datatype, int *outbuf, MPI_Aint *outsize, MPI_Aint *position, int *ierr); 
void F77_FUNC(mpi_pack_external,MPI_PACK_EXTERNAL)(int *datarep[], int *inbuf, int *incount, int *datatype, int *outbuf, MPI_Aint *outsize, MPI_Aint *position, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Pack_external");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_pack_external,PMPI_PACK_EXTERNAL)(datarep, inbuf, incount, datatype, outbuf, outsize, position, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datarep);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*incount);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*outsize);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)position);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Pack_external_size(const char datarep[], int incount, MPI_Datatype datatype, MPI_Aint *size)< */
void F77_FUNC(pmpi_pack_external_size,PMPI_PACK_EXTERNAL_SIZE)(int *datarep[], int *incount, int *datatype, MPI_Aint *size, int *ierr); 
void F77_FUNC(mpi_pack_external_size,MPI_PACK_EXTERNAL_SIZE)(int *datarep[], int *incount, int *datatype, MPI_Aint *size, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Pack_external_size");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_pack_external_size,PMPI_PACK_EXTERNAL_SIZE)(datarep, incount, datatype, size, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datarep);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*incount);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)size);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status)< */
void F77_FUNC(pmpi_request_get_status,PMPI_REQUEST_GET_STATUS)(int *request, int *flag, int *status, int *ierr); 
void F77_FUNC(mpi_request_get_status,MPI_REQUEST_GET_STATUS)(int *request, int *flag, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Request_get_status");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_request_get_status,PMPI_REQUEST_GET_STATUS)(request, flag, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *request);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Status_c2f(const MPI_Status *c_status, MPI_Fint *f_status)< */

/* parsing >int MPI_Status_f2c(const MPI_Fint *f_status, MPI_Status *c_status)< */

/* parsing >int MPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[], const int array_of_distribs[], const int array_of_dargs[], const int array_of_psizes[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_darray,PMPI_TYPE_CREATE_DARRAY)(int *size, int *rank, int *ndims, int *array_of_gsizes[], int *array_of_distribs[], int *array_of_dargs[], int *array_of_psizes[], int *order, int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_darray,MPI_TYPE_CREATE_DARRAY)(int *size, int *rank, int *ndims, int *array_of_gsizes[], int *array_of_distribs[], int *array_of_dargs[], int *array_of_psizes[], int *order, int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_darray");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_darray,PMPI_TYPE_CREATE_DARRAY)(size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs, array_of_psizes, order, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*size);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*rank);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_gsizes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_distribs);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_dargs);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_psizes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*order);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_hindexed(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_hindexed,PMPI_TYPE_CREATE_HINDEXED)(int *count, int *array_of_blocklengths[], MPI_Aint *array_of_displacements[], int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_hindexed,MPI_TYPE_CREATE_HINDEXED)(int *count, int *array_of_blocklengths[], MPI_Aint *array_of_displacements[], int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_hindexed");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_hindexed,PMPI_TYPE_CREATE_HINDEXED)(count, array_of_blocklengths, array_of_displacements, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_blocklengths);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_hvector,PMPI_TYPE_CREATE_HVECTOR)(int *count, int *blocklength, MPI_Aint *stride, int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_hvector,MPI_TYPE_CREATE_HVECTOR)(int *count, int *blocklength, MPI_Aint *stride, int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_hvector");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_hvector,PMPI_TYPE_CREATE_HVECTOR)(count, blocklength, stride, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*stride);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_indexed_block(int count, int blocklength, const int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_indexed_block,PMPI_TYPE_CREATE_INDEXED_BLOCK)(int *count, int *blocklength, int *array_of_displacements[], int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_indexed_block,MPI_TYPE_CREATE_INDEXED_BLOCK)(int *count, int *blocklength, int *array_of_displacements[], int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_indexed_block");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_indexed_block,PMPI_TYPE_CREATE_INDEXED_BLOCK)(count, blocklength, array_of_displacements, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_hindexed_block(int count, int blocklength, const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_hindexed_block,PMPI_TYPE_CREATE_HINDEXED_BLOCK)(int *count, int *blocklength, MPI_Aint *array_of_displacements[], int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_hindexed_block,MPI_TYPE_CREATE_HINDEXED_BLOCK)(int *count, int *blocklength, MPI_Aint *array_of_displacements[], int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_hindexed_block");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_hindexed_block,PMPI_TYPE_CREATE_HINDEXED_BLOCK)(count, blocklength, array_of_displacements, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*blocklength);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_resized,PMPI_TYPE_CREATE_RESIZED)(int *oldtype, MPI_Aint *lb, MPI_Aint *extent, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_resized,MPI_TYPE_CREATE_RESIZED)(int *oldtype, MPI_Aint *lb, MPI_Aint *extent, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_resized");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_resized,PMPI_TYPE_CREATE_RESIZED)(oldtype, lb, extent, newtype, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*lb);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*extent);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_struct(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], const MPI_Datatype array_of_types[], MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_struct,PMPI_TYPE_CREATE_STRUCT)(int *count, int *array_of_blocklengths[], MPI_Aint *array_of_displacements[], int *array_of_types[], int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_struct,MPI_TYPE_CREATE_STRUCT)(int *count, int *array_of_blocklengths[], MPI_Aint *array_of_displacements[], int *array_of_types[], int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_struct");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_struct,PMPI_TYPE_CREATE_STRUCT)(count, array_of_blocklengths, array_of_displacements, array_of_types, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_blocklengths);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_displacements);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_types);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_subarray,PMPI_TYPE_CREATE_SUBARRAY)(int *ndims, int *array_of_sizes[], int *array_of_subsizes[], int *array_of_starts[], int *order, int *oldtype, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_subarray,MPI_TYPE_CREATE_SUBARRAY)(int *ndims, int *array_of_sizes[], int *array_of_subsizes[], int *array_of_starts[], int *order, int *oldtype, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_subarray");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_subarray,PMPI_TYPE_CREATE_SUBARRAY)(ndims, array_of_sizes, array_of_subsizes, array_of_starts, order, oldtype, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*ndims);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_sizes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_subsizes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)array_of_starts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*order);
  
bufptr += printdatatype(MPI_Type_f2c(*oldtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (oldtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(oldtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent)< */
void F77_FUNC(pmpi_type_get_extent,PMPI_TYPE_GET_EXTENT)(int *datatype, MPI_Aint *lb, MPI_Aint *extent, int *ierr); 
void F77_FUNC(mpi_type_get_extent,MPI_TYPE_GET_EXTENT)(int *datatype, MPI_Aint *lb, MPI_Aint *extent, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_extent");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_get_extent,PMPI_TYPE_GET_EXTENT)(datatype, lb, extent, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)lb);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)extent);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent)< */
void F77_FUNC(pmpi_type_get_true_extent,PMPI_TYPE_GET_TRUE_EXTENT)(int *datatype, MPI_Aint *true_lb, MPI_Aint *true_extent, int *ierr); 
void F77_FUNC(mpi_type_get_true_extent,MPI_TYPE_GET_TRUE_EXTENT)(int *datatype, MPI_Aint *true_lb, MPI_Aint *true_extent, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_get_true_extent");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_get_true_extent,PMPI_TYPE_GET_TRUE_EXTENT)(datatype, true_lb, true_extent, ierr);

  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)true_lb);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)true_extent);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Unpack_external(const char datarep[], const void *inbuf, MPI_Aint insize, MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype)< */
void F77_FUNC(pmpi_unpack_external,PMPI_UNPACK_EXTERNAL)(int *datarep[], int *inbuf, MPI_Aint *insize, MPI_Aint *position, int *outbuf, int *outcount, int *datatype, int *ierr); 
void F77_FUNC(mpi_unpack_external,MPI_UNPACK_EXTERNAL)(int *datarep[], int *inbuf, MPI_Aint *insize, MPI_Aint *position, int *outbuf, int *outcount, int *datatype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Unpack_external");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_unpack_external,PMPI_UNPACK_EXTERNAL)(datarep, inbuf, insize, position, outbuf, outcount, datatype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)datarep);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*insize);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)position);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*outcount);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_create_errhandler(MPI_Win_errhandler_function *win_errhandler_fn, MPI_Errhandler *errhandler)< */
void F77_FUNC(pmpi_win_create_errhandler,PMPI_WIN_CREATE_ERRHANDLER)(int *win_errhandler_fn, int *errhandler, int *ierr); 
void F77_FUNC(mpi_win_create_errhandler,MPI_WIN_CREATE_ERRHANDLER)(int *win_errhandler_fn, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_create_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_create_errhandler,PMPI_WIN_CREATE_ERRHANDLER)(win_errhandler_fn, errhandler, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)win_errhandler_fn);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler)< */
void F77_FUNC(pmpi_win_get_errhandler,PMPI_WIN_GET_ERRHANDLER)(int *win, int *errhandler, int *ierr); 
void F77_FUNC(mpi_win_get_errhandler,MPI_WIN_GET_ERRHANDLER)(int *win, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_get_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_get_errhandler,PMPI_WIN_GET_ERRHANDLER)(win, errhandler, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler)< */
void F77_FUNC(pmpi_win_set_errhandler,PMPI_WIN_SET_ERRHANDLER)(int *win, int *errhandler, int *ierr); 
void F77_FUNC(mpi_win_set_errhandler,MPI_WIN_SET_ERRHANDLER)(int *win, int *errhandler, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Win_set_errhandler");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_win_set_errhandler,PMPI_WIN_SET_ERRHANDLER)(win, errhandler, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *win);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *errhandler);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_f90_integer(int range, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_f90_integer,PMPI_TYPE_CREATE_F90_INTEGER)(int *range, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_f90_integer,MPI_TYPE_CREATE_F90_INTEGER)(int *range, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_f90_integer");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_f90_integer,PMPI_TYPE_CREATE_F90_INTEGER)(range, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*range);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_f90_real(int precision, int range, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_f90_real,PMPI_TYPE_CREATE_F90_REAL)(int *precision, int *range, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_f90_real,MPI_TYPE_CREATE_F90_REAL)(int *precision, int *range, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_f90_real");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_f90_real,PMPI_TYPE_CREATE_F90_REAL)(precision, range, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*precision);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*range);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Type_create_f90_complex(int precision, int range, MPI_Datatype *newtype)< */
void F77_FUNC(pmpi_type_create_f90_complex,PMPI_TYPE_CREATE_F90_COMPLEX)(int *precision, int *range, int *newtype, int *ierr); 
void F77_FUNC(mpi_type_create_f90_complex,MPI_TYPE_CREATE_F90_COMPLEX)(int *precision, int *range, int *newtype, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Type_create_f90_complex");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_type_create_f90_complex,PMPI_TYPE_CREATE_F90_COMPLEX)(precision, range, newtype, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*precision);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*range);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newtype);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Reduce_local(const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op op)< */
void F77_FUNC(pmpi_reduce_local,PMPI_REDUCE_LOCAL)(int *inbuf, int *inoutbuf, int *count, int *datatype, int *op, int *ierr); 
void F77_FUNC(mpi_reduce_local,MPI_REDUCE_LOCAL)(int *inbuf, int *inoutbuf, int *count, int *datatype, int *op, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Reduce_local");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_reduce_local,PMPI_REDUCE_LOCAL)(inbuf, inoutbuf, count, datatype, op, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)inoutbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Op_commutative(MPI_Op op, int *commute)< */
void F77_FUNC(pmpi_op_commutative,PMPI_OP_COMMUTATIVE)(int *op, int *commute, int *ierr); 
void F77_FUNC(mpi_op_commutative,MPI_OP_COMMUTATIVE)(int *op, int *commute, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Op_commutative");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_op_commutative,PMPI_OP_COMMUTATIVE)(op, commute, ierr);

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)commute);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)< */
void F77_FUNC(pmpi_reduce_scatter_block,PMPI_REDUCE_SCATTER_BLOCK)(int *sendbuf, int *recvbuf, int *recvcount, int *datatype, int *op, int *comm, int *ierr); 
void F77_FUNC(mpi_reduce_scatter_block,MPI_REDUCE_SCATTER_BLOCK)(int *sendbuf, int *recvbuf, int *recvcount, int *datatype, int *op, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Reduce_scatter_block");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_reduce_scatter_block,PMPI_REDUCE_SCATTER_BLOCK)(sendbuf, recvbuf, recvcount, datatype, op, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, const int sources[], const int sourceweights[], int outdegree, const int destinations[], const int destweights[], MPI_Info info, int reorder, MPI_Comm *comm_dist_graph)< */
void F77_FUNC(pmpi_dist_graph_create_adjacent,PMPI_DIST_GRAPH_CREATE_ADJACENT)(int *comm_old, int *indegree, int *sources[], int *sourceweights[], int *outdegree, int *destinations[], int *destweights[], int *info, int *reorder, int *comm_dist_graph, int *ierr); 
void F77_FUNC(mpi_dist_graph_create_adjacent,MPI_DIST_GRAPH_CREATE_ADJACENT)(int *comm_old, int *indegree, int *sources[], int *sourceweights[], int *outdegree, int *destinations[], int *destweights[], int *info, int *reorder, int *comm_dist_graph, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dist_graph_create_adjacent");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_dist_graph_create_adjacent,PMPI_DIST_GRAPH_CREATE_ADJACENT)(comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm_old);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm_old, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm_old, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*indegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sources);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sourceweights);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*outdegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destinations);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destweights);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*reorder);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_dist_graph);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Dist_graph_create(MPI_Comm comm_old, int n, const int sources[], const int degrees[], const int destinations[], const int weights[], MPI_Info info, int reorder, MPI_Comm *comm_dist_graph)< */
void F77_FUNC(pmpi_dist_graph_create,PMPI_DIST_GRAPH_CREATE)(int *comm_old, int *n, int *sources[], int *degrees[], int *destinations[], int *weights[], int *info, int *reorder, int *comm_dist_graph, int *ierr); 
void F77_FUNC(mpi_dist_graph_create,MPI_DIST_GRAPH_CREATE)(int *comm_old, int *n, int *sources[], int *degrees[], int *destinations[], int *weights[], int *info, int *reorder, int *comm_dist_graph, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dist_graph_create");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_dist_graph_create,PMPI_DIST_GRAPH_CREATE)(comm_old, n, sources, degrees, destinations, weights, info, reorder, comm_dist_graph, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm_old);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm_old, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm_old, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*n);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sources);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)degrees);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destinations);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)weights);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*reorder);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)comm_dist_graph);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Dist_graph_neighbors_count(MPI_Comm comm, int *indegree, int *outdegree, int *weighted)< */
void F77_FUNC(pmpi_dist_graph_neighbors_count,PMPI_DIST_GRAPH_NEIGHBORS_COUNT)(int *comm, int *indegree, int *outdegree, int *weighted, int *ierr); 
void F77_FUNC(mpi_dist_graph_neighbors_count,MPI_DIST_GRAPH_NEIGHBORS_COUNT)(int *comm, int *indegree, int *outdegree, int *weighted, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dist_graph_neighbors_count");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_dist_graph_neighbors_count,PMPI_DIST_GRAPH_NEIGHBORS_COUNT)(comm, indegree, outdegree, weighted, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)indegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)outdegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)weighted);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[], int maxoutdegree, int destinations[], int destweights[])< */
void F77_FUNC(pmpi_dist_graph_neighbors,PMPI_DIST_GRAPH_NEIGHBORS)(int *comm, int *maxindegree, int *sources[], int *sourceweights[], int *maxoutdegree, int *destinations[], int *destweights[], int *ierr); 
void F77_FUNC(mpi_dist_graph_neighbors,MPI_DIST_GRAPH_NEIGHBORS)(int *comm, int *maxindegree, int *sources[], int *sourceweights[], int *maxoutdegree, int *destinations[], int *destweights[], int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Dist_graph_neighbors");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_dist_graph_neighbors,PMPI_DIST_GRAPH_NEIGHBORS)(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations, destweights, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*maxindegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sources);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sourceweights);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*maxoutdegree);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destinations);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)destweights);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Improbe(int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, MPI_Status *status)< */
void F77_FUNC(pmpi_improbe,PMPI_IMPROBE)(int *source, int *tag, int *comm, int *flag, int *message, int *status, int *ierr); 
void F77_FUNC(mpi_improbe,MPI_IMPROBE)(int *source, int *tag, int *comm, int *flag, int *message, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Improbe");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_improbe,PMPI_IMPROBE)(source, tag, comm, flag, message, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)flag);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)message);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Request *request)< */
void F77_FUNC(pmpi_imrecv,PMPI_IMRECV)(int *buf, int *count, int *datatype, int *message, int *request, int *ierr); 
void F77_FUNC(mpi_imrecv,MPI_IMRECV)(int *buf, int *count, int *datatype, int *message, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Imrecv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_imrecv,PMPI_IMRECV)(buf, count, datatype, message, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)message);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status)< */
void F77_FUNC(pmpi_mprobe,PMPI_MPROBE)(int *source, int *tag, int *comm, int *message, int *status, int *ierr); 
void F77_FUNC(mpi_mprobe,MPI_MPROBE)(int *source, int *tag, int *comm, int *message, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Mprobe");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_mprobe,PMPI_MPROBE)(source, tag, comm, message, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*source);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*tag);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)message);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Mrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Status *status)< */
void F77_FUNC(pmpi_mrecv,PMPI_MRECV)(int *buf, int *count, int *datatype, int *message, int *status, int *ierr); 
void F77_FUNC(mpi_mrecv,MPI_MRECV)(int *buf, int *count, int *datatype, int *message, int *status, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Mrecv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_mrecv,PMPI_MRECV)(buf, count, datatype, message, status, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)message);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)status);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_idup(MPI_Comm comm, MPI_Comm *newcomm, MPI_Request *request)< */
void F77_FUNC(pmpi_comm_idup,PMPI_COMM_IDUP)(int *comm, int *newcomm, int *request, int *ierr); 
void F77_FUNC(mpi_comm_idup,MPI_COMM_IDUP)(int *comm, int *newcomm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_idup");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_idup,PMPI_COMM_IDUP)(comm, newcomm, request, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ibarrier(MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ibarrier,PMPI_IBARRIER)(int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ibarrier,MPI_IBARRIER)(int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ibarrier");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ibarrier,PMPI_IBARRIER)(comm, request, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ibcast,PMPI_IBCAST)(int *buffer, int *count, int *datatype, int *root, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ibcast,MPI_IBCAST)(int *buffer, int *count, int *datatype, int *root, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ibcast");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ibcast,PMPI_IBCAST)(buffer, count, datatype, root, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)buffer);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_igather,PMPI_IGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_igather,MPI_IGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Igather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_igather,PMPI_IGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_igatherv,PMPI_IGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts[], int *displs[], int *recvtype, int *root, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_igatherv,MPI_IGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts[], int *displs[], int *recvtype, int *root, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Igatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_igatherv,PMPI_IGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Iscatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_iscatter,PMPI_ISCATTER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_iscatter,MPI_ISCATTER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iscatter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_iscatter,PMPI_ISCATTER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Iscatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_iscatterv,PMPI_ISCATTERV)(int *sendbuf, int *sendcounts[], int *displs[], int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_iscatterv,MPI_ISCATTERV)(int *sendbuf, int *sendcounts[], int *displs[], int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iscatterv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_iscatterv,PMPI_ISCATTERV)(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_iallgather,PMPI_IALLGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_iallgather,MPI_IALLGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iallgather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_iallgather,PMPI_IALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_iallgatherv,PMPI_IALLGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts[], int *displs[], int *recvtype, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_iallgatherv,MPI_IALLGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts[], int *displs[], int *recvtype, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iallgatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_iallgatherv,PMPI_IALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ialltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ialltoall,PMPI_IALLTOALL)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ialltoall,MPI_IALLTOALL)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ialltoall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ialltoall,PMPI_IALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ialltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ialltoallv,PMPI_IALLTOALLV)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtype, int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtype, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ialltoallv,MPI_IALLTOALLV)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtype, int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtype, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ialltoallv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ialltoallv,PMPI_IALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ialltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ialltoallw,PMPI_IALLTOALLW)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtypes[], int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtypes[], int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ialltoallw,MPI_IALLTOALLW)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtypes[], int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtypes[], int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ialltoallw");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ialltoallw,PMPI_IALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendtypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvtypes);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ireduce,PMPI_IREDUCE)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *root, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ireduce,MPI_IREDUCE)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *root, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ireduce");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ireduce,PMPI_IREDUCE)(sendbuf, recvbuf, count, datatype, op, root, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*root);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_iallreduce,PMPI_IALLREDUCE)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_iallreduce,MPI_IALLREDUCE)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iallreduce");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_iallreduce,PMPI_IALLREDUCE)(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ireduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ireduce_scatter,PMPI_IREDUCE_SCATTER)(int *sendbuf, int *recvbuf, int *recvcounts[], int *datatype, int *op, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ireduce_scatter,MPI_IREDUCE_SCATTER)(int *sendbuf, int *recvbuf, int *recvcounts[], int *datatype, int *op, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ireduce_scatter");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ireduce_scatter,PMPI_IREDUCE_SCATTER)(sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ireduce_scatter_block,PMPI_IREDUCE_SCATTER_BLOCK)(int *sendbuf, int *recvbuf, int *recvcount, int *datatype, int *op, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ireduce_scatter_block,MPI_IREDUCE_SCATTER_BLOCK)(int *sendbuf, int *recvbuf, int *recvcount, int *datatype, int *op, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ireduce_scatter_block");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ireduce_scatter_block,PMPI_IREDUCE_SCATTER_BLOCK)(sendbuf, recvbuf, recvcount, datatype, op, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Iscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_iscan,PMPI_ISCAN)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_iscan,MPI_ISCAN)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iscan");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_iscan,PMPI_ISCAN)(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Iexscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_iexscan,PMPI_IEXSCAN)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_iexscan,MPI_IEXSCAN)(int *sendbuf, int *recvbuf, int *count, int *datatype, int *op, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Iexscan");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_iexscan,PMPI_IEXSCAN)(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*count);
  
bufptr += printdatatype(MPI_Type_f2c(*datatype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (datatype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(datatype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += printop(MPI_Op_f2c(*op), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ineighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ineighbor_allgather,PMPI_INEIGHBOR_ALLGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ineighbor_allgather,MPI_INEIGHBOR_ALLGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_allgather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ineighbor_allgather,PMPI_INEIGHBOR_ALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ineighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ineighbor_allgatherv,PMPI_INEIGHBOR_ALLGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts[], int *displs[], int *recvtype, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ineighbor_allgatherv,MPI_INEIGHBOR_ALLGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts[], int *displs[], int *recvtype, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_allgatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ineighbor_allgatherv,PMPI_INEIGHBOR_ALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ineighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ineighbor_alltoall,PMPI_INEIGHBOR_ALLTOALL)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ineighbor_alltoall,MPI_INEIGHBOR_ALLTOALL)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_alltoall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ineighbor_alltoall,PMPI_INEIGHBOR_ALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ineighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ineighbor_alltoallv,PMPI_INEIGHBOR_ALLTOALLV)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtype, int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtype, int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ineighbor_alltoallv,MPI_INEIGHBOR_ALLTOALLV)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtype, int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtype, int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_alltoallv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ineighbor_alltoallv,PMPI_INEIGHBOR_ALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Ineighbor_alltoallw(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request *request)< */
void F77_FUNC(pmpi_ineighbor_alltoallw,PMPI_INEIGHBOR_ALLTOALLW)(int *sendbuf, int *sendcounts[], MPI_Aint *sdispls[], int *sendtypes[], int *recvbuf, int *recvcounts[], MPI_Aint *rdispls[], int *recvtypes[], int *comm, int *request, int *ierr); 
void F77_FUNC(mpi_ineighbor_alltoallw,MPI_INEIGHBOR_ALLTOALLW)(int *sendbuf, int *sendcounts[], MPI_Aint *sdispls[], int *sendtypes[], int *recvbuf, int *recvcounts[], MPI_Aint *rdispls[], int *recvtypes[], int *comm, int *request, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Ineighbor_alltoallw");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_ineighbor_alltoallw,PMPI_INEIGHBOR_ALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendtypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvtypes);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)request);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)< */
void F77_FUNC(pmpi_neighbor_allgather,PMPI_NEIGHBOR_ALLGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *ierr); 
void F77_FUNC(mpi_neighbor_allgather,MPI_NEIGHBOR_ALLGATHER)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_allgather");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_neighbor_allgather,PMPI_NEIGHBOR_ALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Neighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm)< */
void F77_FUNC(pmpi_neighbor_allgatherv,PMPI_NEIGHBOR_ALLGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts[], int *displs[], int *recvtype, int *comm, int *ierr); 
void F77_FUNC(mpi_neighbor_allgatherv,MPI_NEIGHBOR_ALLGATHERV)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcounts[], int *displs[], int *recvtype, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_allgatherv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_neighbor_allgatherv,PMPI_NEIGHBOR_ALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)displs);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Neighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)< */
void F77_FUNC(pmpi_neighbor_alltoall,PMPI_NEIGHBOR_ALLTOALL)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *ierr); 
void F77_FUNC(mpi_neighbor_alltoall,MPI_NEIGHBOR_ALLTOALL)(int *sendbuf, int *sendcount, int *sendtype, int *recvbuf, int *recvcount, int *recvtype, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_alltoall");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_neighbor_alltoall,PMPI_NEIGHBOR_ALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*sendcount);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*recvcount);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)< */
void F77_FUNC(pmpi_neighbor_alltoallv,PMPI_NEIGHBOR_ALLTOALLV)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtype, int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtype, int *comm, int *ierr); 
void F77_FUNC(mpi_neighbor_alltoallv,MPI_NEIGHBOR_ALLTOALLV)(int *sendbuf, int *sendcounts[], int *sdispls[], int *sendtype, int *recvbuf, int *recvcounts[], int *rdispls[], int *recvtype, int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_alltoallv");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_neighbor_alltoallv,PMPI_NEIGHBOR_ALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  
bufptr += printdatatype(MPI_Type_f2c(*sendtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (sendtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(sendtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  
bufptr += printdatatype(MPI_Type_f2c(*recvtype), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) (recvtype, &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)(recvtype, &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Neighbor_alltoallw(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)< */
void F77_FUNC(pmpi_neighbor_alltoallw,PMPI_NEIGHBOR_ALLTOALLW)(int *sendbuf, int *sendcounts[], MPI_Aint *sdispls[], int *sendtypes[], int *recvbuf, int *recvcounts[], MPI_Aint *rdispls[], int *recvtypes[], int *comm, int *ierr); 
void F77_FUNC(mpi_neighbor_alltoallw,MPI_NEIGHBOR_ALLTOALLW)(int *sendbuf, int *sendcounts[], MPI_Aint *sdispls[], int *sendtypes[], int *recvbuf, int *recvcounts[], MPI_Aint *rdispls[], int *recvtypes[], int *comm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Neighbor_alltoallw");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_neighbor_alltoallw,PMPI_NEIGHBOR_ALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)sendtypes);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvbuf);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvcounts);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)rdispls);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)recvtypes);
  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}


  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}

/* parsing >int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm)< */
void F77_FUNC(pmpi_comm_split_type,PMPI_COMM_SPLIT_TYPE)(int *comm, int *split_type, int *key, int *info, int *newcomm, int *ierr); 
void F77_FUNC(mpi_comm_split_type,MPI_COMM_SPLIT_TYPE)(int *comm, int *split_type, int *key, int *info, int *newcomm, int *ierr) { 
  check();
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "MPI_Comm_split_type");
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);

  F77_FUNC(pmpi_comm_split_type,PMPI_COMM_SPLIT_TYPE)(comm, split_type, key, info, newcomm, ierr);

  
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *comm);
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)(comm, &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*split_type);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*key);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *info);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)newcomm);

  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\n", PMPI_Wtime()*1e6);


}
#ifdef __cplusplus
}
#endif

