#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#define UNW_LOCAL_ONLY //we do not need to unwind frames in another process
#include <libunwind.h>
#include <linux/limits.h>

#define LAP2_TRANSFER_BUFFER_SIZE  1024
#define LAP2_BACKTRACE_BUF_SIZE    4096
#define WRITE_TRACE(fmt, args...) fprintf(lap_fptr, fmt, args)

FILE* lap_fptr = NULL;
char* lap_backtrace_buf = NULL;
int lap_initialized = 0;
int lap_mpi_initialized = 0;

int lap_tracing_enabled = 1;
int lap_backtrace_enabled = 1;
int lap_elem_tracing_enabled = 0;


static void init_back_trace(void) {

}

static void lap_get_full_backtrace(char* buf, size_t len) {
  size_t written = 0;
  unw_cursor_t cursor;
  unw_context_t context;

  // Initialize cursor to current frame for local unwinding.
  unw_getcontext(&context);
  unw_init_local(&cursor, &context);

  // Unwind frames one by one, going up the frame stack.
  while (unw_step(&cursor) > 0) {
    unw_word_t offset, pc;
    unw_get_reg(&cursor, UNW_REG_IP, &pc);
    if (pc == 0) {
      break;
    }
    written += snprintf(&buf[written], len-written, "0x%lx:", pc);

    char sym[256];
    if (unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0) {
      written += snprintf(&buf[written], len-written, " (%s+0x%lx) <- ", sym, offset);
    } else {
      written += snprintf(&buf[written], len-written, "NO_SYMBOL    ");
    }
  }
  if (written>0) written -= 4;
  buf[written] = '\0';
}

static void lap_check(void) {
  if (lap_mpi_initialized == 0) PMPI_Initialized(&lap_mpi_initialized);
  if (lap_initialized) return;
  lap_fptr = tmpfile(); //write to a tmpfile, we don't know our rank yet, until MPI is initialized
  lap_backtrace_buf = malloc(LAP2_BACKTRACE_BUF_SIZE);
  assert(lap_backtrace_buf);
  assert(lap_fptr);
  init_back_trace();
  lap_initialized = 1;
}


static void lap_collect_traces(void) {
    int comm_rank, comm_size;
    PMPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    int trace_size = ftell(lap_fptr);
    fseek(lap_fptr, 0, SEEK_SET);

    
    char trace_fname[PATH_MAX];
    char *env = getenv("HTOR_PMPI_FILE_PREFIX");
    if (env == NULL)
      snprintf(trace_fname, PATH_MAX, "/scratch/sshen/traces/lap2-trace-rank-%i.txt", comm_rank);
    else
    {
      snprintf(trace_fname, PATH_MAX, "%s%i.txt", env, comm_rank);
    }
    FILE* trace_fh = fopen(trace_fname, "w");
    void *buf = malloc(sizeof(char) * trace_size);
    assert (buf != NULL);

    // Reads from the temporary file
    size_t bytes_read = fread(buf, 1, trace_size, lap_fptr);
    assert (bytes_read == trace_size);
    fwrite(buf, 1, trace_size, trace_fh);
    fclose(trace_fh);
    free(buf);
    printf("*** lap2: rank %i of %i trace is %i bytes long ***\n", comm_rank, comm_size, trace_size);
    /* int* trace_sizes = malloc(comm_size * sizeof(int)); */
    /* if (trace_sizes == NULL) { */
    /*   fprintf(stderr, "lap2 ran out of memory when collecting traces :(\n"); */
    /*   return; */
    /* } */
    /* void* chunkbuf = malloc(LAP2_TRANSFER_BUFFER_SIZE); */
    /* if (chunkbuf == NULL) { */
    /*   fprintf(stderr, "lap2 ran out of memory when collecting traces, decrease LAP2_TRANSFER_BUFFER_SIZE=%i :(\n", LAP2_TRANSFER_BUFFER_SIZE); */
    /*   return; */
    /* } */
    /* PMPI_Gather(&trace_size, 1, MPI_INT, trace_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD); */
    /* if (comm_rank == 0) { */
    /*     for (int r=0; r<comm_size; r++) { */
    /*         printf("*** lap2: rank %i of %i trace is %i bytes long ***\n", r, comm_size, trace_sizes[r]); */
    /*         char trace_fname[FILENAME_MAX]; */
    /*         snprintf(trace_fname, FILENAME_MAX, "lap2-trace-rank-%i-of-%i.txt", r, comm_size); */
    /*         FILE* trace_fh = fopen(trace_fname, "w"); */
    /*         int num_chunks = trace_sizes[r] / LAP2_TRANSFER_BUFFER_SIZE; */
    /*         if (num_chunks * LAP2_TRANSFER_BUFFER_SIZE < trace_sizes[r]) num_chunks += 1; */
    /*         for (int chunk=0; chunk<num_chunks; chunk++) { */
    /*             int bytes_received = 0; */
    /*             if (r != comm_rank) { */
    /*                 MPI_Status recv_status; */
    /*                 PMPI_Recv(chunkbuf, LAP2_TRANSFER_BUFFER_SIZE, MPI_BYTE, r, chunk, MPI_COMM_WORLD, &recv_status); */
    /*                 PMPI_Get_count(&recv_status, MPI_BYTE, &bytes_received); */
    /*             } */
    /*             else { */
    /*                 bytes_received = fread(chunkbuf, 1, LAP2_TRANSFER_BUFFER_SIZE, lap_fptr); */
    /*             } */
    /*             fwrite(chunkbuf, 1, bytes_received, trace_fh); */
    /*         } */
    /*         fclose(trace_fh); */
    /*     } */
    /* } */
    /* else { */
    /*     int num_chunks = trace_size / LAP2_TRANSFER_BUFFER_SIZE; */
    /*     if (num_chunks * LAP2_TRANSFER_BUFFER_SIZE < trace_size) num_chunks += 1; */
    /*     for (int chunk=0; chunk<num_chunks; chunk++) { */
    /*         size_t bytes_read = fread(chunkbuf, 1, LAP2_TRANSFER_BUFFER_SIZE, lap_fptr); */
    /*         PMPI_Send(chunkbuf, bytes_read, MPI_BYTE, 0, chunk, MPI_COMM_WORLD); */
    /*     } */
    /* } */
    /* free(trace_sizes); */
    /* free(chunkbuf); */
}

int MPI_Abort (MPI_Comm comm, int errorcode) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Abort(comm, errorcode);
  lap_mpi_initialized = 0;
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Abort:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Abort(comm, errorcode);
  lap_mpi_initialized = 0;
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (errorcode));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Accumulate (const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Accumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Accumulate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Accumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win);
  WRITE_TRACE("%p:", origin_addr);
    WRITE_TRACE("%lli:", (long long int) (origin_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(origin_datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) (target_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(target_datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Add_error_class (int * errorclass) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Add_error_class(errorclass);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Add_error_class:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Add_error_class(errorclass);
    WRITE_TRACE("%lli:", (long long int) *(errorclass));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Add_error_code (int errorclass, int * errorcode) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Add_error_code(errorclass, errorcode);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Add_error_code:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Add_error_code(errorclass, errorcode);
    WRITE_TRACE("%lli:", (long long int) (errorclass));
    WRITE_TRACE("%lli:", (long long int) *(errorcode));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Add_error_string (int errorcode, const char * string) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Add_error_string(errorcode, string);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Add_error_string:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Add_error_string(errorcode, string);
    WRITE_TRACE("%lli:", (long long int) (errorcode));
  WRITE_TRACE("%p:", string);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Allgather (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Allgather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Allgatherv (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Allgatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (displs[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Alloc_mem (MPI_Aint size, MPI_Info info, void * baseptr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Alloc_mem(size, info, baseptr);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Alloc_mem:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Alloc_mem(size, info, baseptr);
    WRITE_TRACE("%lli:", (long long int) (size));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%p:", baseptr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Allreduce (const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Allreduce:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Alltoall (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Alltoall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Alltoallv (const void * sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Alltoallv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (rdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Alltoallw (const void * sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void * recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Alltoallw:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sendtypes, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(sendtypes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (rdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) recvtypes, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(recvtypes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

/* int MPI_Attr_delete (MPI_Comm comm, int keyval) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Attr_delete(comm, keyval);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Attr_delete:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Attr_delete(comm, keyval);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (keyval));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
} */

/* int MPI_Attr_get (MPI_Comm comm, int keyval, void * attribute_val, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Attr_get(comm, keyval, attribute_val, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Attr_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Attr_get(comm, keyval, attribute_val, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (keyval));
  WRITE_TRACE("%p:", attribute_val);
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
} */

/* int MPI_Attr_put (MPI_Comm comm, int keyval, void * attribute_val) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Attr_put(comm, keyval, attribute_val);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Attr_put:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Attr_put(comm, keyval, attribute_val);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (keyval));
  WRITE_TRACE("%p:", attribute_val);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
} */

int MPI_Barrier (MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Barrier(comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Barrier:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Barrier(comm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Bcast (void * buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Bcast(buffer, count, datatype, root, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Bcast:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Bcast(buffer, count, datatype, root, comm);
  WRITE_TRACE("%p:", buffer);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Bsend (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Bsend(buf, count, datatype, dest, tag, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Bsend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Bsend(buf, count, datatype, dest, tag, comm);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Bsend_init (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Bsend_init(buf, count, datatype, dest, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Bsend_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Bsend_init(buf, count, datatype, dest, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Buffer_attach (void * buffer, int size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Buffer_attach(buffer, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Buffer_attach:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Buffer_attach(buffer, size);
  WRITE_TRACE("%p:", buffer);
    WRITE_TRACE("%lli:", (long long int) (size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Buffer_detach (void * buffer, int * size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Buffer_detach(buffer, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Buffer_detach:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Buffer_detach(buffer, size);
  WRITE_TRACE("%p:", buffer);
    WRITE_TRACE("%lli:", (long long int) *(size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Cancel (MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Cancel(request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Cancel:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Cancel(request);
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Cart_coords (MPI_Comm comm, int rank, int maxdims, int coords[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Cart_coords(comm, rank, maxdims, coords);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_coords:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Cart_coords(comm, rank, maxdims, coords);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (rank));
    WRITE_TRACE("%lli:", (long long int) (maxdims));
  WRITE_TRACE("%p,%i[", (void*) coords, (int) maxdims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxdims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (coords[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Cart_create (MPI_Comm old_comm, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm * comm_cart) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Cart_create(old_comm, ndims, dims, periods, reorder, comm_cart);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Cart_create(old_comm, ndims, dims, periods, reorder, comm_cart);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(old_comm));
    WRITE_TRACE("%lli:", (long long int) (ndims));
  WRITE_TRACE("%p,%i[", (void*) dims, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (dims[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) periods, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (periods[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) (reorder));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*comm_cart));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Cart_get (MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Cart_get(comm, maxdims, dims, periods, coords);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Cart_get(comm, maxdims, dims, periods, coords);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (maxdims));
  WRITE_TRACE("%p,%i[", (void*) dims, (int) maxdims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxdims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (dims[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) periods, (int) maxdims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxdims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (periods[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) coords, (int) maxdims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxdims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (coords[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Cart_map (MPI_Comm comm, int ndims, const int dims[], const int periods[], int * newrank) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Cart_map(comm, ndims, dims, periods, newrank);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_map:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Cart_map(comm, ndims, dims, periods, newrank);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (ndims));
  WRITE_TRACE("%p,%i[", (void*) dims, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (dims[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) periods, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (periods[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *(newrank));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Cart_rank (MPI_Comm comm, const int coords[], int * rank) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Cart_rank(comm, coords, rank);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_rank:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Cart_rank(comm, coords, rank);
int ndims; PMPI_Cartdim_get(comm, &ndims);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%p,%i[", (void*) coords, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (coords[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *(rank));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Cart_shift (MPI_Comm comm, int direction, int disp, int * rank_source, int * rank_dest) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Cart_shift(comm, direction, disp, rank_source, rank_dest);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_shift:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Cart_shift(comm, direction, disp, rank_source, rank_dest);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (direction));
    WRITE_TRACE("%lli:", (long long int) (disp));
    WRITE_TRACE("%lli:", (long long int) *(rank_source));
    WRITE_TRACE("%lli:", (long long int) *(rank_dest));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Cart_sub (MPI_Comm comm, const int remain_dims[], MPI_Comm * new_comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Cart_sub(comm, remain_dims, new_comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_sub:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Cart_sub(comm, remain_dims, new_comm);
int ndims; PMPI_Cartdim_get(comm, &ndims);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%p,%i[", (void*) remain_dims, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (remain_dims[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*new_comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Cartdim_get (MPI_Comm comm, int * ndims) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Cartdim_get(comm, ndims);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Cartdim_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Cartdim_get(comm, ndims);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(ndims));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Close_port (const char * port_name) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Close_port(port_name);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Close_port:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Close_port(port_name);
  WRITE_TRACE("%p:", port_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_accept (const char * port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm * newcomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_accept(port_name, info, root, comm, newcomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_accept:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_accept(port_name, info, root, comm, newcomm);
  WRITE_TRACE("%p:", port_name);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_call_errhandler (MPI_Comm comm, int errorcode) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_call_errhandler(comm, errorcode);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_call_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_call_errhandler(comm, errorcode);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (errorcode));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_compare (MPI_Comm comm1, MPI_Comm comm2, int * result) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_compare(comm1, comm2, result);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_compare:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_compare(comm1, comm2, result);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm1));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm2));
    WRITE_TRACE("%lli:", (long long int) *(result));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_connect (const char * port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm * newcomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_connect(port_name, info, root, comm, newcomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_connect:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_connect(port_name, info, root, comm, newcomm);
  WRITE_TRACE("%p:", port_name);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_create (MPI_Comm comm, MPI_Group group, MPI_Comm * newcomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_create(comm, group, newcomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_create(comm, group, newcomm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_create_errhandler (MPI_Comm_errhandler_function * function, MPI_Errhandler * errhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_create_errhandler(function, errhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_create_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_create_errhandler(function, errhandler);
  WRITE_TRACE("%p:", function);
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(*errhandler));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_create_group (MPI_Comm comm, MPI_Group group, int tag, MPI_Comm * newcomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_create_group(comm, group, tag, newcomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_create_group:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_create_group(comm, group, tag, newcomm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_create_keyval (MPI_Comm_copy_attr_function * comm_copy_attr_fn, MPI_Comm_delete_attr_function * comm_delete_attr_fn, int * comm_keyval, void * extra_state) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_create_keyval(comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_create_keyval:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_create_keyval(comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state);
  WRITE_TRACE("%p:", comm_copy_attr_fn);
  WRITE_TRACE("%p:", comm_delete_attr_fn);
    WRITE_TRACE("%lli:", (long long int) *(comm_keyval));
  WRITE_TRACE("%p:", extra_state);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_delete_attr (MPI_Comm comm, int comm_keyval) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_delete_attr(comm, comm_keyval);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_delete_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (comm_keyval));
  pmpi_retval = PMPI_Comm_delete_attr(comm, comm_keyval);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_disconnect (MPI_Comm * comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_disconnect(comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_disconnect:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_disconnect(comm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_dup (MPI_Comm comm, MPI_Comm * newcomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_dup(comm, newcomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_dup:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_dup(comm, newcomm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_dup_with_info (MPI_Comm comm, MPI_Info info, MPI_Comm * newcomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_dup_with_info(comm, info, newcomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_dup_with_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_dup_with_info(comm, info, newcomm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_free (MPI_Comm * comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_free(comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*comm));
  pmpi_retval = PMPI_Comm_free(comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_free_keyval (int * comm_keyval) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_free_keyval(comm_keyval);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_free_keyval:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_free_keyval(comm_keyval);
    WRITE_TRACE("%lli:", (long long int) *(comm_keyval));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_get_attr (MPI_Comm comm, int comm_keyval, void * attribute_val, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_get_attr(comm, comm_keyval, attribute_val, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_get_attr(comm, comm_keyval, attribute_val, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (comm_keyval));
  WRITE_TRACE("%p:", attribute_val);
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_get_errhandler (MPI_Comm comm, MPI_Errhandler * erhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_get_errhandler(comm, erhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_get_errhandler(comm, erhandler);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(*erhandler));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_get_info (MPI_Comm comm, MPI_Info * info_used) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_get_info(comm, info_used);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_get_info(comm, info_used);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(*info_used));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_get_name (MPI_Comm comm, char * comm_name, int * resultlen) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_get_name(comm, comm_name, resultlen);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_get_name(comm, comm_name, resultlen);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%p:", comm_name);
    WRITE_TRACE("%lli:", (long long int) *(resultlen));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_get_parent (MPI_Comm * parent) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_get_parent(parent);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_parent:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_get_parent(parent);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*parent));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_group (MPI_Comm comm, MPI_Group * group) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_group(comm, group);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_group:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_group(comm, group);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*group));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_idup (MPI_Comm comm, MPI_Comm * newcomm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_idup(comm, newcomm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_idup:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_idup(comm, newcomm, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_join (int fd, MPI_Comm * intercomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_join(fd, intercomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_join:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_join(fd, intercomm);
    WRITE_TRACE("%lli:", (long long int) (fd));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*intercomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_rank (MPI_Comm comm, int * rank) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_rank(comm, rank);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_rank:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_rank(comm, rank);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(rank));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_remote_group (MPI_Comm comm, MPI_Group * group) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_remote_group(comm, group);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_remote_group:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_remote_group(comm, group);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*group));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_remote_size (MPI_Comm comm, int * size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_remote_size(comm, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_remote_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_remote_size(comm, size);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_set_attr (MPI_Comm comm, int comm_keyval, void * attribute_val) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_set_attr(comm, comm_keyval, attribute_val);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_set_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_set_attr(comm, comm_keyval, attribute_val);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (comm_keyval));
  WRITE_TRACE("%p:", attribute_val);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_set_errhandler (MPI_Comm comm, MPI_Errhandler errhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_set_errhandler(comm, errhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_set_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_set_errhandler(comm, errhandler);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(errhandler));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_set_info (MPI_Comm comm, MPI_Info info) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_set_info(comm, info);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_set_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_set_info(comm, info);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_set_name (MPI_Comm comm, const char * comm_name) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_set_name(comm, comm_name);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_set_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_set_name(comm, comm_name);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%p:", comm_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_size (MPI_Comm comm, int * size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_size(comm, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_size(comm, size);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_spawn (const char * command, char * argv[], int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm * intercomm, int array_of_errcodes[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_spawn(command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_spawn:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_spawn(command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes);
  WRITE_TRACE("%p:", command);
  WRITE_TRACE("%p,%i[", (void*) argv, (int) 0);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<0; trace_elem_idx++) {
    WRITE_TRACE("%p;", argv[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) (maxprocs));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*intercomm));
  WRITE_TRACE("%p,%i[", (void*) array_of_errcodes, (int) maxprocs);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxprocs; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_errcodes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_spawn_multiple (int count, char * array_of_commands[], char ** array_of_argv[], const int array_of_maxprocs[], const MPI_Info array_of_info[], int root, MPI_Comm comm, MPI_Comm * intercomm, int array_of_errcodes[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_spawn_multiple(count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_spawn_multiple:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_spawn_multiple(count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes);
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%p,%i[", (void*) array_of_commands, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
    WRITE_TRACE("%p;", array_of_commands[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_argv, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
    WRITE_TRACE("%p;", array_of_argv[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_maxprocs, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_maxprocs[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_info, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Info_c2f(array_of_info[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*intercomm));
  WRITE_TRACE("%p,%i[", (void*) array_of_errcodes, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_errcodes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_split (MPI_Comm comm, int color, int key, MPI_Comm * newcomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_split(comm, color, key, newcomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_split:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_split(comm, color, key, newcomm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (color));
    WRITE_TRACE("%lli:", (long long int) (key));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_split_type (MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm * newcomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_split_type(comm, split_type, key, info, newcomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_split_type:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_split_type(comm, split_type, key, info, newcomm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (split_type));
    WRITE_TRACE("%lli:", (long long int) (key));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Comm_test_inter (MPI_Comm comm, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Comm_test_inter(comm, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_test_inter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Comm_test_inter(comm, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Compare_and_swap (const void * origin_addr, const void * compare_addr, void * result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Compare_and_swap:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp, win);
  WRITE_TRACE("%p:", origin_addr);
  WRITE_TRACE("%p:", compare_addr);
  WRITE_TRACE("%p:", result_addr);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Dims_create (int nnodes, int ndims, int dims[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Dims_create(nnodes, ndims, dims);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Dims_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Dims_create(nnodes, ndims, dims);
    WRITE_TRACE("%lli:", (long long int) (nnodes));
    WRITE_TRACE("%lli:", (long long int) (ndims));
  WRITE_TRACE("%p,%i[", (void*) dims, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (dims[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Dist_graph_create (MPI_Comm comm_old, int n, const int nodes[], const int degrees[], const int targets[], const int weights[], MPI_Info info, int reorder, MPI_Comm * newcomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Dist_graph_create(comm_old, n, nodes, degrees, targets, weights, info, reorder, newcomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Dist_graph_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Dist_graph_create(comm_old, n, nodes, degrees, targets, weights, info, reorder, newcomm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm_old));
    WRITE_TRACE("%lli:", (long long int) (n));
  WRITE_TRACE("%p,%i[", (void*) nodes, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (nodes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) degrees, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (degrees[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) targets, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (targets[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) weights, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (weights[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) (reorder));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newcomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Dist_graph_create_adjacent (MPI_Comm comm_old, int indegree, const int sources[], const int sourceweights[], int outdegree, const int destinations[], const int destweights[], MPI_Info info, int reorder, MPI_Comm * comm_dist_graph) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Dist_graph_create_adjacent(comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Dist_graph_create_adjacent:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Dist_graph_create_adjacent(comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm_old));
    WRITE_TRACE("%lli:", (long long int) (indegree));
  WRITE_TRACE("%p,%i[", (void*) sources, (int) indegree);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<indegree; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sources[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sourceweights, (int) indegree);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<indegree; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sourceweights[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) (outdegree));
  WRITE_TRACE("%p,%i[", (void*) destinations, (int) outdegree);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<outdegree; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (destinations[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) destweights, (int) outdegree);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<outdegree; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (destweights[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) (reorder));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*comm_dist_graph));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Dist_graph_neighbors (MPI_Comm comm, int maxindegree, int sources[], int sourceweights[], int maxoutdegree, int destinations[], int destweights[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Dist_graph_neighbors(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations, destweights);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Dist_graph_neighbors:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Dist_graph_neighbors(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations, destweights);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (maxindegree));
  WRITE_TRACE("%p,%i[", (void*) sources, (int) maxindegree);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxindegree; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sources[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sourceweights, (int) maxindegree);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxindegree; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sourceweights[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) (maxoutdegree));
  WRITE_TRACE("%p,%i[", (void*) destinations, (int) maxoutdegree);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxoutdegree; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (destinations[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) destweights, (int) maxoutdegree);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxoutdegree; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (destweights[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Dist_graph_neighbors_count (MPI_Comm comm, int * inneighbors, int * outneighbors, int * weighted) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Dist_graph_neighbors_count(comm, inneighbors, outneighbors, weighted);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Dist_graph_neighbors_count:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Dist_graph_neighbors_count(comm, inneighbors, outneighbors, weighted);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(inneighbors));
    WRITE_TRACE("%lli:", (long long int) *(outneighbors));
    WRITE_TRACE("%lli:", (long long int) *(weighted));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Errhandler_free (MPI_Errhandler * errhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Errhandler_free(errhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Errhandler_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(*errhandler));
  pmpi_retval = PMPI_Errhandler_free(errhandler);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Error_class (int errorcode, int * errorclass) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Error_class(errorcode, errorclass);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Error_class:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Error_class(errorcode, errorclass);
    WRITE_TRACE("%lli:", (long long int) (errorcode));
    WRITE_TRACE("%lli:", (long long int) *(errorclass));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Error_string (int errorcode, char * string, int * resultlen) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Error_string(errorcode, string, resultlen);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Error_string:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Error_string(errorcode, string, resultlen);
    WRITE_TRACE("%lli:", (long long int) (errorcode));
  WRITE_TRACE("%p:", string);
    WRITE_TRACE("%lli:", (long long int) *(resultlen));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Exscan (const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Exscan:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Fetch_and_op (const void * origin_addr, void * result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Fetch_and_op(origin_addr, result_addr, datatype, target_rank, target_disp, op, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Fetch_and_op:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Fetch_and_op(origin_addr, result_addr, datatype, target_rank, target_disp, op, win);
  WRITE_TRACE("%p:", origin_addr);
  WRITE_TRACE("%p:", result_addr);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_call_errhandler (MPI_File fh, int errorcode) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_call_errhandler(fh, errorcode);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_call_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_call_errhandler(fh, errorcode);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (errorcode));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_close (MPI_File * fh) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_close(fh);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_close:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_close(fh);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(*fh));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_create_errhandler (MPI_File_errhandler_function * function, MPI_Errhandler * errhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_create_errhandler(function, errhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_create_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_create_errhandler(function, errhandler);
  WRITE_TRACE("%p:", function);
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(*errhandler));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_delete (const char * filename, MPI_Info info) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_delete(filename, info);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_delete:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_delete(filename, info);
  WRITE_TRACE("%p:", filename);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_amode (MPI_File fh, int * amode) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_amode(fh, amode);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_amode:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_amode(fh, amode);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) *(amode));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_atomicity (MPI_File fh, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_atomicity(fh, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_atomicity:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_atomicity(fh, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_byte_offset (MPI_File fh, MPI_Offset offset, MPI_Offset * disp) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_byte_offset(fh, offset, disp);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_byte_offset:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_byte_offset(fh, offset, disp);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
    WRITE_TRACE("%lli:", (long long int) *(disp));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_errhandler (MPI_File file, MPI_Errhandler * errhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_errhandler(file, errhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_errhandler(file, errhandler);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(file));
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(*errhandler));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_group (MPI_File fh, MPI_Group * group) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_group(fh, group);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_group:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_group(fh, group);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*group));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_info (MPI_File fh, MPI_Info * info_used) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_info(fh, info_used);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_info(fh, info_used);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(*info_used));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_position (MPI_File fh, MPI_Offset * offset) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_position(fh, offset);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_position:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_position(fh, offset);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) *(offset));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_position_shared (MPI_File fh, MPI_Offset * offset) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_position_shared(fh, offset);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_position_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_position_shared(fh, offset);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) *(offset));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_size (MPI_File fh, MPI_Offset * size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_size(fh, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_size(fh, size);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) *(size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_type_extent (MPI_File fh, MPI_Datatype datatype, MPI_Aint * extent) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_type_extent(fh, datatype, extent);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_type_extent:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_type_extent(fh, datatype, extent);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) *(extent));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_get_view (MPI_File fh, MPI_Offset * disp, MPI_Datatype * etype, MPI_Datatype * filetype, char * datarep) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_get_view(fh, disp, etype, filetype, datarep);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_view:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_get_view(fh, disp, etype, filetype, datarep);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) *(disp));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*etype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*filetype));
  WRITE_TRACE("%p:", datarep);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iread (MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iread(fh, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iread(fh, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iread_all (MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iread_all(fh, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iread_all(fh, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iread_at (MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iread_at(fh, offset, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread_at:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iread_at(fh, offset, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iread_at_all (MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iread_at_all(fh, offset, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread_at_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iread_at_all(fh, offset, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iread_shared (MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iread_shared(fh, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iread_shared(fh, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iwrite (MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iwrite(fh, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iwrite(fh, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iwrite_all (MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iwrite_all(fh, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iwrite_all(fh, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iwrite_at (MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iwrite_at(fh, offset, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite_at:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iwrite_at(fh, offset, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iwrite_at_all (MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iwrite_at_all(fh, offset, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite_at_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iwrite_at_all(fh, offset, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_iwrite_shared (MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_iwrite_shared(fh, buf, count, datatype, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_iwrite_shared(fh, buf, count, datatype, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_open (MPI_Comm comm, const char * filename, int amode, MPI_Info info, MPI_File * fh) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_open(comm, filename, amode, info, fh);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_open:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_open(comm, filename, amode, info, fh);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%p:", filename);
    WRITE_TRACE("%lli:", (long long int) (amode));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(*fh));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_preallocate (MPI_File fh, MPI_Offset size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_preallocate(fh, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_preallocate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_preallocate(fh, size);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read (MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read(fh, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read(fh, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_all (MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_all(fh, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_all(fh, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_all_begin (MPI_File fh, void * buf, int count, MPI_Datatype datatype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_all_begin(fh, buf, count, datatype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_all_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_all_begin(fh, buf, count, datatype);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_all_end (MPI_File fh, void * buf, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_all_end(fh, buf, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_all_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_all_end(fh, buf, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_at (MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_at(fh, offset, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_at:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_at(fh, offset, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_at_all (MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_at_all(fh, offset, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_at_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_at_all(fh, offset, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_at_all_begin (MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_at_all_begin(fh, offset, buf, count, datatype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_at_all_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_at_all_begin(fh, offset, buf, count, datatype);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_at_all_end (MPI_File fh, void * buf, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_at_all_end(fh, buf, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_at_all_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_at_all_end(fh, buf, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_ordered (MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_ordered(fh, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_ordered:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_ordered(fh, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_ordered_begin (MPI_File fh, void * buf, int count, MPI_Datatype datatype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_ordered_begin(fh, buf, count, datatype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_ordered_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_ordered_begin(fh, buf, count, datatype);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_ordered_end (MPI_File fh, void * buf, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_ordered_end(fh, buf, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_ordered_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_ordered_end(fh, buf, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_read_shared (MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_read_shared(fh, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_read_shared(fh, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_seek (MPI_File fh, MPI_Offset offset, int whence) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_seek(fh, offset, whence);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_seek:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_seek(fh, offset, whence);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
    WRITE_TRACE("%lli:", (long long int) (whence));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_seek_shared (MPI_File fh, MPI_Offset offset, int whence) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_seek_shared(fh, offset, whence);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_seek_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_seek_shared(fh, offset, whence);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
    WRITE_TRACE("%lli:", (long long int) (whence));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_set_atomicity (MPI_File fh, int flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_set_atomicity(fh, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_atomicity:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_set_atomicity(fh, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_set_errhandler (MPI_File file, MPI_Errhandler errhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_set_errhandler(file, errhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_set_errhandler(file, errhandler);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(file));
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(errhandler));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_set_info (MPI_File fh, MPI_Info info) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_set_info(fh, info);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_set_info(fh, info);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_set_size (MPI_File fh, MPI_Offset size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_set_size(fh, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_set_size(fh, size);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_set_view (MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char * datarep, MPI_Info info) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_set_view(fh, disp, etype, filetype, datarep, info);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_view:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_set_view(fh, disp, etype, filetype, datarep, info);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (disp));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(etype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(filetype));
  WRITE_TRACE("%p:", datarep);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_sync (MPI_File fh) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_sync(fh);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_sync:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_sync(fh);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write (MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write(fh, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write(fh, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_all (MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_all(fh, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_all(fh, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_all_begin (MPI_File fh, const void * buf, int count, MPI_Datatype datatype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_all_begin(fh, buf, count, datatype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_all_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_all_begin(fh, buf, count, datatype);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_all_end (MPI_File fh, const void * buf, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_all_end(fh, buf, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_all_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_all_end(fh, buf, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_at (MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_at(fh, offset, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_at:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_at(fh, offset, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_at_all (MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_at_all(fh, offset, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_at_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_at_all(fh, offset, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_at_all_begin (MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_at_all_begin(fh, offset, buf, count, datatype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_at_all_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_at_all_begin(fh, offset, buf, count, datatype);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
    WRITE_TRACE("%lli:", (long long int) (offset));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_at_all_end (MPI_File fh, const void * buf, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_at_all_end(fh, buf, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_at_all_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_at_all_end(fh, buf, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_ordered (MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_ordered(fh, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_ordered:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_ordered(fh, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_ordered_begin (MPI_File fh, const void * buf, int count, MPI_Datatype datatype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_ordered_begin(fh, buf, count, datatype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_ordered_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_ordered_begin(fh, buf, count, datatype);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_ordered_end (MPI_File fh, const void * buf, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_ordered_end(fh, buf, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_ordered_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_ordered_end(fh, buf, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_File_write_shared (MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_File_write_shared(fh, buf, count, datatype, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_File_write_shared(fh, buf, count, datatype, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_File_c2f(fh));
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Finalize () {
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Finalize:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
lap_collect_traces();
  pmpi_retval = PMPI_Finalize();
  lap_mpi_initialized = 0;
  return pmpi_retval;
}

int MPI_Finalized (int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Finalized(flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Finalized:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Finalized(flag);
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Free_mem (void * base) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Free_mem(base);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Free_mem:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  WRITE_TRACE("%p:", base);
  pmpi_retval = PMPI_Free_mem(base);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Gather (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Gather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Gatherv (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Gatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (displs[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Get (void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win);
  WRITE_TRACE("%p:", origin_addr);
    WRITE_TRACE("%lli:", (long long int) (origin_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(origin_datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) (target_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(target_datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Get_accumulate (const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, void * result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Get_accumulate(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_accumulate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Get_accumulate(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win);
  WRITE_TRACE("%p:", origin_addr);
    WRITE_TRACE("%lli:", (long long int) (origin_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(origin_datatype));
  WRITE_TRACE("%p:", result_addr);
    WRITE_TRACE("%lli:", (long long int) (result_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(result_datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) (target_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(target_datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Get_address (const void * location, MPI_Aint * address) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Get_address(location, address);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_address:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Get_address(location, address);
  WRITE_TRACE("%p:", location);
    WRITE_TRACE("%lli:", (long long int) *(address));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Get_count (const MPI_Status * status, MPI_Datatype datatype, int * count) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Get_count(status, datatype, count);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_count:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Get_count(status, datatype, count);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) *(count));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Get_elements (const MPI_Status * status, MPI_Datatype datatype, int * count) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Get_elements(status, datatype, count);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_elements:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Get_elements(status, datatype, count);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) *(count));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Get_elements_x (const MPI_Status * status, MPI_Datatype datatype, MPI_Count * count) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Get_elements_x(status, datatype, count);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_elements_x:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Get_elements_x(status, datatype, count);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) *(count));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Get_library_version (char * version, int * resultlen) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Get_library_version(version, resultlen);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_library_version:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Get_library_version(version, resultlen);
  WRITE_TRACE("%p:", version);
    WRITE_TRACE("%lli:", (long long int) *(resultlen));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Get_processor_name (char * name, int * resultlen) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Get_processor_name(name, resultlen);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_processor_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Get_processor_name(name, resultlen);
  WRITE_TRACE("%p:", name);
    WRITE_TRACE("%lli:", (long long int) *(resultlen));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Get_version (int * version, int * subversion) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Get_version(version, subversion);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_version:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Get_version(version, subversion);
    WRITE_TRACE("%lli:", (long long int) *(version));
    WRITE_TRACE("%lli:", (long long int) *(subversion));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Graph_create (MPI_Comm comm_old, int nnodes, const int index[], const int edges[], int reorder, MPI_Comm * comm_graph) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Graph_create(comm_old, nnodes, index, edges, reorder, comm_graph);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Graph_create(comm_old, nnodes, index, edges, reorder, comm_graph);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm_old));
    WRITE_TRACE("%lli:", (long long int) (nnodes));
  WRITE_TRACE("%p,%i[", (void*) index, (int) nnodes);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<nnodes; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (index[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) edges, (int) index[nnodes-1]);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<index[nnodes-1]; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (edges[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) (reorder));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*comm_graph));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Graph_get (MPI_Comm comm, int maxindex, int maxedges, int index[], int edges[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Graph_get(comm, maxindex, maxedges, index, edges);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Graph_get(comm, maxindex, maxedges, index, edges);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (maxindex));
    WRITE_TRACE("%lli:", (long long int) (maxedges));
  WRITE_TRACE("%p,%i[", (void*) index, (int) maxindex);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxindex; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (index[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) edges, (int) maxedges);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxedges; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (edges[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Graph_map (MPI_Comm comm, int nnodes, const int index[], const int edges[], int * newrank) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Graph_map(comm, nnodes, index, edges, newrank);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_map:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Graph_map(comm, nnodes, index, edges, newrank);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (nnodes));
  WRITE_TRACE("%p,%i[", (void*) index, (int) nnodes);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<nnodes; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (index[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) edges, (int) index[nnodes-1]);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<index[nnodes-1]; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (edges[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *(newrank));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Graph_neighbors (MPI_Comm comm, int rank, int maxneighbors, int neighbors[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Graph_neighbors(comm, rank, maxneighbors, neighbors);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_neighbors:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Graph_neighbors(comm, rank, maxneighbors, neighbors);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (rank));
    WRITE_TRACE("%lli:", (long long int) (maxneighbors));
  WRITE_TRACE("%p,%i[", (void*) neighbors, (int) maxneighbors);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<maxneighbors; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (neighbors[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Graph_neighbors_count (MPI_Comm comm, int rank, int * nneighbors) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Graph_neighbors_count(comm, rank, nneighbors);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_neighbors_count:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Graph_neighbors_count(comm, rank, nneighbors);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) (rank));
    WRITE_TRACE("%lli:", (long long int) *(nneighbors));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Graphdims_get (MPI_Comm comm, int * nnodes, int * nedges) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Graphdims_get(comm, nnodes, nedges);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Graphdims_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Graphdims_get(comm, nnodes, nedges);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(nnodes));
    WRITE_TRACE("%lli:", (long long int) *(nedges));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Grequest_complete (MPI_Request request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Grequest_complete(request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Grequest_complete:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Grequest_complete(request);
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Grequest_start (MPI_Grequest_query_function * query_fn, MPI_Grequest_free_function * free_fn, MPI_Grequest_cancel_function * cancel_fn, void * extra_state, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Grequest_start(query_fn, free_fn, cancel_fn, extra_state, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Grequest_start:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Grequest_start(query_fn, free_fn, cancel_fn, extra_state, request);
  WRITE_TRACE("%p:", query_fn);
  WRITE_TRACE("%p:", free_fn);
  WRITE_TRACE("%p:", cancel_fn);
  WRITE_TRACE("%p:", extra_state);
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_compare (MPI_Group group1, MPI_Group group2, int * result) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_compare(group1, group2, result);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_compare:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_compare(group1, group2, result);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group1));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group2));
    WRITE_TRACE("%lli:", (long long int) *(result));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_difference (MPI_Group group1, MPI_Group group2, MPI_Group * newgroup) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_difference(group1, group2, newgroup);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_difference:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_difference(group1, group2, newgroup);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group1));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group2));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*newgroup));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_excl (MPI_Group group, int n, const int ranks[], MPI_Group * newgroup) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_excl(group, n, ranks, newgroup);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_excl:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_excl(group, n, ranks, newgroup);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) (n));
  WRITE_TRACE("%p,%i[", (void*) ranks, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (ranks[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*newgroup));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_free (MPI_Group * group) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_free(group);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*group));
  pmpi_retval = PMPI_Group_free(group);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_incl (MPI_Group group, int n, const int ranks[], MPI_Group * newgroup) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_incl(group, n, ranks, newgroup);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_incl:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_incl(group, n, ranks, newgroup);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) (n));
  WRITE_TRACE("%p,%i[", (void*) ranks, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (ranks[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*newgroup));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_intersection (MPI_Group group1, MPI_Group group2, MPI_Group * newgroup) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_intersection(group1, group2, newgroup);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_intersection:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_intersection(group1, group2, newgroup);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group1));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group2));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*newgroup));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_range_excl (MPI_Group group, int n, int ranges[][3], MPI_Group * newgroup) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_range_excl(group, n, ranges, newgroup);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_range_excl:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_range_excl(group, n, ranges, newgroup);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) (n));
  WRITE_TRACE("%p,%i[", (void*) ranges, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
    WRITE_TRACE("[%i,%i,%i];", ranges[trace_elem_idx][0], ranges[trace_elem_idx][1], ranges[trace_elem_idx][2]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*newgroup));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_range_incl (MPI_Group group, int n, int ranges[][3], MPI_Group * newgroup) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_range_incl(group, n, ranges, newgroup);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_range_incl:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_range_incl(group, n, ranges, newgroup);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) (n));
  WRITE_TRACE("%p,%i[", (void*) ranges, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
    WRITE_TRACE("[%i,%i,%i];", ranges[trace_elem_idx][0], ranges[trace_elem_idx][1], ranges[trace_elem_idx][2]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*newgroup));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_rank (MPI_Group group, int * rank) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_rank(group, rank);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_rank:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_rank(group, rank);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) *(rank));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_size (MPI_Group group, int * size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_size(group, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_size(group, size);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) *(size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_translate_ranks (MPI_Group group1, int n, const int ranks1[], MPI_Group group2, int ranks2[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_translate_ranks(group1, n, ranks1, group2, ranks2);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_translate_ranks:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_translate_ranks(group1, n, ranks1, group2, ranks2);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group1));
    WRITE_TRACE("%lli:", (long long int) (n));
  WRITE_TRACE("%p,%i[", (void*) ranks1, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (ranks1[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group2));
  WRITE_TRACE("%p,%i[", (void*) ranks2, (int) n);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<n; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (ranks2[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Group_union (MPI_Group group1, MPI_Group group2, MPI_Group * newgroup) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Group_union(group1, group2, newgroup);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_union:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Group_union(group1, group2, newgroup);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group1));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group2));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*newgroup));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Iallgather (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Iallgather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Iallgatherv (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Iallgatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (displs[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Iallreduce (const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Iallreduce:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ialltoall (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ialltoall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ialltoallv (const void * sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ialltoallv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (rdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ialltoallw (const void * sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void * recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ialltoallw:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sendtypes, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(sendtypes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (rdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) recvtypes, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(recvtypes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ibarrier (MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ibarrier(comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ibarrier:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ibarrier(comm, request);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ibcast (void * buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ibcast(buffer, count, datatype, root, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ibcast:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ibcast(buffer, count, datatype, root, comm, request);
  WRITE_TRACE("%p:", buffer);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ibsend (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ibsend(buf, count, datatype, dest, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ibsend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ibsend(buf, count, datatype, dest, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Iexscan (const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Iexscan(sendbuf, recvbuf, count, datatype, op, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Iexscan:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Iexscan(sendbuf, recvbuf, count, datatype, op, comm, request);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Igather (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Igather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Igatherv (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Igatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, request);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (displs[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Improbe (int source, int tag, MPI_Comm comm, int * flag, MPI_Message * message, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Improbe(source, tag, comm, flag, message, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Improbe:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Improbe(source, tag, comm, flag, message, status);
    WRITE_TRACE("%lli:", (long long int) (source));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(flag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Message_c2f(*message));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Imrecv (void * buf, int count, MPI_Datatype type, MPI_Message * message, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Imrecv(buf, count, type, message, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Imrecv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Imrecv(buf, count, type, message, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) PMPI_Message_c2f(*message));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ineighbor_allgather (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ineighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_allgather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ineighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ineighbor_allgatherv (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ineighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_allgatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ineighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count(comm, &ideg, &odeg, &wted);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (displs[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ineighbor_alltoall (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ineighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_alltoall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ineighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ineighbor_alltoallv (const void * sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_alltoallv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count(comm, &ideg, &odeg, &wted);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (rdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ineighbor_alltoallw (const void * sendbuf, const int sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype sendtypes[], void * recvbuf, const int recvcounts[], const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ineighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_alltoallw:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ineighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count(comm, &ideg, &odeg, &wted);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sendtypes, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(sendtypes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (rdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) recvtypes, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(recvtypes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Info_create (MPI_Info * info) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Info_create(info);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Info_create(info);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(*info));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Info_delete (MPI_Info info, const char * key) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Info_delete(info, key);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_delete:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Info_delete(info, key);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%p:", key);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Info_dup (MPI_Info info, MPI_Info * newinfo) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Info_dup(info, newinfo);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_dup:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Info_dup(info, newinfo);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(*newinfo));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Info_free (MPI_Info * info) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Info_free(info);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(*info));
  pmpi_retval = PMPI_Info_free(info);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Info_get (MPI_Info info, const char * key, int valuelen, char * value, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Info_get(info, key, valuelen, value, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Info_get(info, key, valuelen, value, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%p:", key);
    WRITE_TRACE("%lli:", (long long int) (valuelen));
  WRITE_TRACE("%p:", value);
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Info_get_nkeys (MPI_Info info, int * nkeys) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Info_get_nkeys(info, nkeys);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_get_nkeys:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Info_get_nkeys(info, nkeys);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) *(nkeys));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Info_get_nthkey (MPI_Info info, int n, char * key) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Info_get_nthkey(info, n, key);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_get_nthkey:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Info_get_nthkey(info, n, key);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) (n));
  WRITE_TRACE("%p:", key);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Info_get_valuelen (MPI_Info info, const char * key, int * valuelen, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Info_get_valuelen(info, key, valuelen, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_get_valuelen:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Info_get_valuelen(info, key, valuelen, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%p:", key);
    WRITE_TRACE("%lli:", (long long int) *(valuelen));
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Info_set (MPI_Info info, const char * key, const char * value) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Info_set(info, key, value);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_set:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Info_set(info, key, value);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%p:", key);
  WRITE_TRACE("%p:", value);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Init (int * argc, char *** argv) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Init(argc, argv);
    return pmpi_retval;
  }
  int pmpi_retval;
  double curr_time;
  lap_check();
  WRITE_TRACE("%s", "MPI_Init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Init(argc, argv);
  curr_time = PMPI_Wtime();
    WRITE_TRACE("%lli:", (long long int) *(argc));
  WRITE_TRACE("%p:", argv);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  WRITE_TRACE("# Curren timestamp: %0.2f\n", curr_time * 1e6);
  return pmpi_retval;
}

int MPI_Init_thread (int * argc, char *** argv, int required, int * provided) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Init_thread(argc, argv, required, provided);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Init_thread:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Init_thread(argc, argv, required, provided);
    WRITE_TRACE("%lli:", (long long int) *(argc));
  WRITE_TRACE("%p:", argv);
    WRITE_TRACE("%lli:", (long long int) (required));
    WRITE_TRACE("%lli:", (long long int) *(provided));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Initialized (int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Initialized(flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Initialized:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Initialized(flag);
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Intercomm_create (MPI_Comm local_comm, int local_leader, MPI_Comm bridge_comm, int remote_leader, int tag, MPI_Comm * newintercomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Intercomm_create(local_comm, local_leader, bridge_comm, remote_leader, tag, newintercomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Intercomm_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Intercomm_create(local_comm, local_leader, bridge_comm, remote_leader, tag, newintercomm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(local_comm));
    WRITE_TRACE("%lli:", (long long int) (local_leader));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(bridge_comm));
    WRITE_TRACE("%lli:", (long long int) (remote_leader));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newintercomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Intercomm_merge (MPI_Comm intercomm, int high, MPI_Comm * newintercomm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Intercomm_merge(intercomm, high, newintercomm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Intercomm_merge:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Intercomm_merge(intercomm, high, newintercomm);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(intercomm));
    WRITE_TRACE("%lli:", (long long int) (high));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(*newintercomm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Iprobe (int source, int tag, MPI_Comm comm, int * flag, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Iprobe(source, tag, comm, flag, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Iprobe:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Iprobe(source, tag, comm, flag, status);
    WRITE_TRACE("%lli:", (long long int) (source));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(flag));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Irecv (void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Irecv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (source));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ireduce (const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ireduce:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm, request);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ireduce_scatter (const void * sendbuf, void * recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ireduce_scatter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm, request);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ireduce_scatter_block (const void * sendbuf, void * recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ireduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ireduce_scatter_block:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ireduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm, request);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Irsend (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Irsend(buf, count, datatype, dest, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Irsend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Irsend(buf, count, datatype, dest, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Is_thread_main (int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Is_thread_main(flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Is_thread_main:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Is_thread_main(flag);
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Iscan (const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Iscan(sendbuf, recvbuf, count, datatype, op, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Iscan:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Iscan(sendbuf, recvbuf, count, datatype, op, comm, request);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Iscatter (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Iscatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Iscatter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Iscatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Iscatterv (const void * sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Iscatterv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) (rank == root ? size : 0));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(rank == root ? size : 0); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) (rank == root ? size : 0));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(rank == root ? size : 0); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (displs[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Isend (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Isend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Issend (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Issend(buf, count, datatype, dest, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Issend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Issend(buf, count, datatype, dest, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

/* int MPI_Keyval_create (MPI_Copy_function * copy_fn, MPI_Delete_function * delete_fn, int * keyval, void * extra_state) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Keyval_create(copy_fn, delete_fn, keyval, extra_state);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Keyval_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Keyval_create(copy_fn, delete_fn, keyval, extra_state);
  WRITE_TRACE("%p:", copy_fn);
  WRITE_TRACE("%p:", delete_fn);
    WRITE_TRACE("%lli:", (long long int) *(keyval));
  WRITE_TRACE("%p:", extra_state);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
} */

/* int MPI_Keyval_free (int * keyval) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Keyval_free(keyval);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Keyval_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *(keyval));
  pmpi_retval = PMPI_Keyval_free(keyval);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
} */

int MPI_Lookup_name (const char * service_name, MPI_Info info, char * port_name) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Lookup_name(service_name, info, port_name);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Lookup_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Lookup_name(service_name, info, port_name);
  WRITE_TRACE("%p:", service_name);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%p:", port_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Mprobe (int source, int tag, MPI_Comm comm, MPI_Message * message, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Mprobe(source, tag, comm, message, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Mprobe:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Mprobe(source, tag, comm, message, status);
    WRITE_TRACE("%lli:", (long long int) (source));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Message_c2f(*message));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Mrecv (void * buf, int count, MPI_Datatype type, MPI_Message * message, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Mrecv(buf, count, type, message, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Mrecv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Mrecv(buf, count, type, message, status);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) PMPI_Message_c2f(*message));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Neighbor_allgather (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Neighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_allgather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Neighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Neighbor_allgatherv (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Neighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_allgatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Neighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count(comm, &ideg, &odeg, &wted);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (displs[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Neighbor_alltoall (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Neighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_alltoall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Neighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Neighbor_alltoallv (const void * sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Neighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_alltoallv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Neighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count(comm, &ideg, &odeg, &wted);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (rdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Neighbor_alltoallw (const void * sendbuf, const int sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype sendtypes[], void * recvbuf, const int recvcounts[], const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_alltoallw:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count(comm, &ideg, &odeg, &wted);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sendtypes, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(sendtypes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (rdispls[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) recvtypes, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(recvtypes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Op_commutative (MPI_Op op, int * commute) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Op_commutative(op, commute);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Op_commutative:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Op_commutative(op, commute);
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) *(commute));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Op_create (MPI_User_function * function, int commute, MPI_Op * op) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Op_create(function, commute, op);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Op_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Op_create(function, commute, op);
  WRITE_TRACE("%p:", function);
    WRITE_TRACE("%lli:", (long long int) (commute));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(*op));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Op_free (MPI_Op * op) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Op_free(op);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Op_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(*op));
  pmpi_retval = PMPI_Op_free(op);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Open_port (MPI_Info info, char * port_name) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Open_port(info, port_name);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Open_port:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Open_port(info, port_name);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%p:", port_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Pack (const void * inbuf, int incount, MPI_Datatype datatype, void * outbuf, int outsize, int * position, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Pack(inbuf, incount, datatype, outbuf, outsize, position, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Pack:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Pack(inbuf, incount, datatype, outbuf, outsize, position, comm);
  WRITE_TRACE("%p:", inbuf);
    WRITE_TRACE("%lli:", (long long int) (incount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  WRITE_TRACE("%p:", outbuf);
    WRITE_TRACE("%lli:", (long long int) (outsize));
    WRITE_TRACE("%lli:", (long long int) *(position));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Pack_external (const char datarep[], const void * inbuf, int incount, MPI_Datatype datatype, void * outbuf, MPI_Aint outsize, MPI_Aint * position) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Pack_external(datarep, inbuf, incount, datatype, outbuf, outsize, position);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Pack_external:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Pack_external(datarep, inbuf, incount, datatype, outbuf, outsize, position);
  WRITE_TRACE("%p,%i[", (void*) datarep, (int) strlen(datarep));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<strlen(datarep); trace_elem_idx++) {
    WRITE_TRACE("%c;", datarep[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p:", inbuf);
    WRITE_TRACE("%lli:", (long long int) (incount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  WRITE_TRACE("%p:", outbuf);
    WRITE_TRACE("%lli:", (long long int) (outsize));
    WRITE_TRACE("%lli:", (long long int) *(position));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Pack_external_size (const char datarep[], int incount, MPI_Datatype datatype, MPI_Aint * size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Pack_external_size(datarep, incount, datatype, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Pack_external_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Pack_external_size(datarep, incount, datatype, size);
  WRITE_TRACE("%p,%i[", (void*) datarep, (int) strlen(datarep));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<strlen(datarep); trace_elem_idx++) {
    WRITE_TRACE("%c;", datarep[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) (incount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) *(size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Pack_size (int incount, MPI_Datatype datatype, MPI_Comm comm, int * size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Pack_size(incount, datatype, comm, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Pack_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Pack_size(incount, datatype, comm, size);
    WRITE_TRACE("%lli:", (long long int) (incount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Pcontrol (const int level,  ...) {
  if (level == 0) { lap_tracing_enabled = 0; lap_elem_tracing_enabled = 0; lap_backtrace_enabled = 0; }
  if (level == 1) { lap_tracing_enabled = 1; lap_elem_tracing_enabled = 0; lap_backtrace_enabled = 0; }
  if (level == 2) { lap_tracing_enabled = 1; lap_elem_tracing_enabled = 1; lap_backtrace_enabled = 0; }
  if (level == 3) { lap_tracing_enabled = 1; lap_elem_tracing_enabled = 1; lap_backtrace_enabled = 1; }
  WRITE_TRACE("# pcontrol with value / epoch %i)\n", level);
  return MPI_SUCCESS;
}

int MPI_Probe (int source, int tag, MPI_Comm comm, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Probe(source, tag, comm, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Probe:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Probe(source, tag, comm, status);
    WRITE_TRACE("%lli:", (long long int) (source));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Publish_name (const char * service_name, MPI_Info info, const char * port_name) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Publish_name(service_name, info, port_name);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Publish_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Publish_name(service_name, info, port_name);
  WRITE_TRACE("%p:", service_name);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%p:", port_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Put (const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Put:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win);
  WRITE_TRACE("%p:", origin_addr);
    WRITE_TRACE("%lli:", (long long int) (origin_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(origin_datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) (target_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(target_datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Query_thread (int * provided) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Query_thread(provided);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Query_thread:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Query_thread(provided);
    WRITE_TRACE("%lli:", (long long int) *(provided));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Raccumulate (const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Raccumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Raccumulate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Raccumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request);
  WRITE_TRACE("%p:", origin_addr);
    WRITE_TRACE("%lli:", (long long int) (origin_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(origin_datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) (target_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(target_datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Recv (void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Recv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (source));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Recv_init (void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Recv_init(buf, count, datatype, source, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Recv_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Recv_init(buf, count, datatype, source, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (source));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Reduce (const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Reduce:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Reduce_local (const void * inbuf, void * inoutbuf, int count, MPI_Datatype datatype, MPI_Op op) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Reduce_local(inbuf, inoutbuf, count, datatype, op);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Reduce_local:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Reduce_local(inbuf, inoutbuf, count, datatype, op);
  WRITE_TRACE("%p:", inbuf);
  WRITE_TRACE("%p:", inoutbuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Reduce_scatter (const void * sendbuf, void * recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Reduce_scatter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (recvcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Reduce_scatter_block (const void * sendbuf, void * recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Reduce_scatter_block:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Register_datarep (const char * datarep, MPI_Datarep_conversion_function * read_conversion_fn, MPI_Datarep_conversion_function * write_conversion_fn, MPI_Datarep_extent_function * dtype_file_extent_fn, void * extra_state) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Register_datarep(datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Register_datarep:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Register_datarep(datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state);
  WRITE_TRACE("%p:", datarep);
  WRITE_TRACE("%p:", read_conversion_fn);
  WRITE_TRACE("%p:", write_conversion_fn);
  WRITE_TRACE("%p:", dtype_file_extent_fn);
  WRITE_TRACE("%p:", extra_state);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Request_free (MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Request_free(request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Request_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  pmpi_retval = PMPI_Request_free(request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Request_get_status (MPI_Request request, int * flag, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Request_get_status(request, flag, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Request_get_status:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Request_get_status(request, flag, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(request));
    WRITE_TRACE("%lli:", (long long int) *(flag));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Rget (void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Rget(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Rget:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Rget(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, request);
  WRITE_TRACE("%p:", origin_addr);
    WRITE_TRACE("%lli:", (long long int) (origin_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(origin_datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) (target_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(target_datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Rget_accumulate (const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, void * result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Rget_accumulate(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Rget_accumulate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Rget_accumulate(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request);
  WRITE_TRACE("%p:", origin_addr);
    WRITE_TRACE("%lli:", (long long int) (origin_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(origin_datatype));
  WRITE_TRACE("%p:", result_addr);
    WRITE_TRACE("%lli:", (long long int) (result_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(result_datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) (target_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(target_datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Rput (const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_cout, MPI_Datatype target_datatype, MPI_Win win, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Rput(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_cout, target_datatype, win, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Rput:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Rput(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_cout, target_datatype, win, request);
  WRITE_TRACE("%p:", origin_addr);
    WRITE_TRACE("%lli:", (long long int) (origin_count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(origin_datatype));
    WRITE_TRACE("%lli:", (long long int) (target_rank));
    WRITE_TRACE("%lli:", (long long int) (target_disp));
    WRITE_TRACE("%lli:", (long long int) (target_cout));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(target_datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Rsend (const void * ibuf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Rsend(ibuf, count, datatype, dest, tag, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Rsend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Rsend(ibuf, count, datatype, dest, tag, comm);
  WRITE_TRACE("%p:", ibuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Rsend_init (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Rsend_init(buf, count, datatype, dest, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Rsend_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Rsend_init(buf, count, datatype, dest, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Scan (const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Scan:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Op_c2f(op));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Scatter (const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Scatter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Scatterv (const void * sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Scatterv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
int rank, size; PMPI_Comm_size(comm, &size); PMPI_Comm_rank(comm, &rank);
//end of prologs
  WRITE_TRACE("%p:", sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) (rank==root ? size : 0));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(rank==root ? size : 0); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (sendcounts[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) (rank==root ? size : 0));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(rank==root ? size : 0); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (displs[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) (root));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Send (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Send(buf, count, datatype, dest, tag, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Send:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Send(buf, count, datatype, dest, tag, comm);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Send_init (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Send_init(buf, count, datatype, dest, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Send_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Send_init(buf, count, datatype, dest, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Sendrecv (const void * sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void * recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Sendrecv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);
  WRITE_TRACE("%p:", sendbuf);
    WRITE_TRACE("%lli:", (long long int) (sendcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(sendtype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (sendtag));
  WRITE_TRACE("%p:", recvbuf);
    WRITE_TRACE("%lli:", (long long int) (recvcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(recvtype));
    WRITE_TRACE("%lli:", (long long int) (source));
    WRITE_TRACE("%lli:", (long long int) (recvtag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Sendrecv_replace (void * buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Sendrecv_replace:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm, status);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (sendtag));
    WRITE_TRACE("%lli:", (long long int) (source));
    WRITE_TRACE("%lli:", (long long int) (recvtag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ssend (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ssend(buf, count, datatype, dest, tag, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ssend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ssend(buf, count, datatype, dest, tag, comm);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Ssend_init (const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Ssend_init(buf, count, datatype, dest, tag, comm, request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Ssend_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Ssend_init(buf, count, datatype, dest, tag, comm, request);
  WRITE_TRACE("%p:", buf);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (dest));
    WRITE_TRACE("%lli:", (long long int) (tag));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Start (MPI_Request * request) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Start(request);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Start:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Start(request);
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Startall (int count, MPI_Request array_of_requests[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Startall(count, array_of_requests);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Startall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Startall(count, array_of_requests);
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%p,%i[", (void*) array_of_requests, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Request_c2f(array_of_requests[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Status_c2f (const MPI_Status * c_status, int * f_status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Status_c2f(c_status, f_status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Status_c2f:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Status_c2f(c_status, f_status);
  if ((c_status == MPI_STATUSES_IGNORE) || (c_status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) c_status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", c_status, (c_status)->MPI_SOURCE, (c_status)->MPI_TAG, (c_status)->MPI_ERROR);}
    WRITE_TRACE("%lli:", (long long int) *(f_status));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Status_f2c (const int * f_status, MPI_Status * c_status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Status_f2c(f_status, c_status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Status_f2c:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Status_f2c(f_status, c_status);
    WRITE_TRACE("%lli:", (long long int) *(f_status));
  if ((c_status == MPI_STATUSES_IGNORE) || (c_status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) c_status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", c_status, (c_status)->MPI_SOURCE, (c_status)->MPI_TAG, (c_status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Status_set_cancelled (MPI_Status * status, int flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Status_set_cancelled(status, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Status_set_cancelled:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Status_set_cancelled(status, flag);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
    WRITE_TRACE("%lli:", (long long int) (flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Status_set_elements (MPI_Status * status, MPI_Datatype datatype, int count) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Status_set_elements(status, datatype, count);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Status_set_elements:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Status_set_elements(status, datatype, count);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Status_set_elements_x (MPI_Status * status, MPI_Datatype datatype, MPI_Count count) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Status_set_elements_x(status, datatype, count);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Status_set_elements_x:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Status_set_elements_x(status, datatype, count);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Test (MPI_Request * request, int * flag, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Test(request, flag, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Test:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Test(request, flag, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
    WRITE_TRACE("%lli:", (long long int) *(flag));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Test_cancelled (const MPI_Status * status, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Test_cancelled(status, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Test_cancelled:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Test_cancelled(status, flag);
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Testall (int count, MPI_Request array_of_requests[], int * flag, MPI_Status array_of_statuses[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Testall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%p,%i[", (void*) array_of_requests, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Request_c2f(array_of_requests[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%p,%i[", (void*) array_of_statuses, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
    WRITE_TRACE("%p:[%i,%i,%i];", array_of_statuses[trace_elem_idx], (&array_of_statuses[trace_elem_idx])->MPI_SOURCE, (&array_of_statuses[trace_elem_idx])->MPI_TAG, (&array_of_statuses[trace_elem_idx])->MPI_ERROR);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Testany (int count, MPI_Request array_of_requests[], int * index, int * flag, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Testany(count, array_of_requests, index, flag, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Testany:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Testany(count, array_of_requests, index, flag, status);
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%p,%i[", (void*) array_of_requests, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Request_c2f(array_of_requests[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *(index));
    WRITE_TRACE("%lli:", (long long int) *(flag));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Testsome (int incount, MPI_Request array_of_requests[], int * outcount, int array_of_indices[], MPI_Status array_of_statuses[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Testsome:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
    WRITE_TRACE("%lli:", (long long int) (incount));
  WRITE_TRACE("%p,%i[", (void*) array_of_requests, (int) incount);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<incount; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Request_c2f(array_of_requests[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *(outcount));
  WRITE_TRACE("%p,%i[", (void*) array_of_indices, (int) *outcount);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<*outcount; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_indices[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_statuses, (int) *outcount);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<*outcount; trace_elem_idx++) {
    WRITE_TRACE("%p:[%i,%i,%i];", array_of_statuses[trace_elem_idx], (&array_of_statuses[trace_elem_idx])->MPI_SOURCE, (&array_of_statuses[trace_elem_idx])->MPI_TAG, (&array_of_statuses[trace_elem_idx])->MPI_ERROR);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Topo_test (MPI_Comm comm, int * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Topo_test(comm, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Topo_test:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Topo_test(comm, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) *(status));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_commit (MPI_Datatype * type) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_commit(type);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_commit:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_commit(type);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*type));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_contiguous (int count, MPI_Datatype oldtype, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_contiguous(count, oldtype, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_contiguous:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_contiguous(count, oldtype, newtype);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_darray (int size, int rank, int ndims, const int gsize_array[], const int distrib_array[], const int darg_array[], const int psize_array[], int order, MPI_Datatype oldtype, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_darray(size, rank, ndims, gsize_array, distrib_array, darg_array, psize_array, order, oldtype, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_darray:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_darray(size, rank, ndims, gsize_array, distrib_array, darg_array, psize_array, order, oldtype, newtype);
    WRITE_TRACE("%lli:", (long long int) (size));
    WRITE_TRACE("%lli:", (long long int) (rank));
    WRITE_TRACE("%lli:", (long long int) (ndims));
  WRITE_TRACE("%p,%i[", (void*) gsize_array, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (gsize_array[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) distrib_array, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (distrib_array[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) darg_array, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (darg_array[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) psize_array, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (psize_array[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) (order));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_f90_complex (int p, int r, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_f90_complex(p, r, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_f90_complex:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_f90_complex(p, r, newtype);
    WRITE_TRACE("%lli:", (long long int) (p));
    WRITE_TRACE("%lli:", (long long int) (r));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_f90_integer (int r, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_f90_integer(r, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_f90_integer:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_f90_integer(r, newtype);
    WRITE_TRACE("%lli:", (long long int) (r));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_f90_real (int p, int r, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_f90_real(p, r, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_f90_real:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_f90_real(p, r, newtype);
    WRITE_TRACE("%lli:", (long long int) (p));
    WRITE_TRACE("%lli:", (long long int) (r));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_hindexed (int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_hindexed(count, array_of_blocklengths, array_of_displacements, oldtype, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_hindexed:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_hindexed(count, array_of_blocklengths, array_of_displacements, oldtype, newtype);
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%p,%i[", (void*) array_of_blocklengths, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_blocklengths[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_displacements, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_displacements[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_hindexed_block (int count, int blocklength, const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_hindexed_block(count, blocklength, array_of_displacements, oldtype, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_hindexed_block:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_hindexed_block(count, blocklength, array_of_displacements, oldtype, newtype);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) (blocklength));
  WRITE_TRACE("%p,%i[", (void*) array_of_displacements, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_displacements[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_hvector (int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_hvector(count, blocklength, stride, oldtype, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_hvector:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_hvector(count, blocklength, stride, oldtype, newtype);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) (blocklength));
    WRITE_TRACE("%lli:", (long long int) (stride));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_indexed_block (int count, int blocklength, const int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_indexed_block(count, blocklength, array_of_displacements, oldtype, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_indexed_block:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_indexed_block(count, blocklength, array_of_displacements, oldtype, newtype);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) (blocklength));
  WRITE_TRACE("%p,%i[", (void*) array_of_displacements, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_displacements[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_keyval (MPI_Type_copy_attr_function * type_copy_attr_fn, MPI_Type_delete_attr_function * type_delete_attr_fn, int * type_keyval, void * extra_state) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_keyval(type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_keyval:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_keyval(type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state);
  WRITE_TRACE("%p:", type_copy_attr_fn);
  WRITE_TRACE("%p:", type_delete_attr_fn);
    WRITE_TRACE("%lli:", (long long int) *(type_keyval));
  WRITE_TRACE("%p:", extra_state);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_resized (MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_resized(oldtype, lb, extent, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_resized:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_resized(oldtype, lb, extent, newtype);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) (lb));
    WRITE_TRACE("%lli:", (long long int) (extent));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_struct (int count, const int array_of_block_lengths[], const MPI_Aint array_of_displacements[], const MPI_Datatype array_of_types[], MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_struct(count, array_of_block_lengths, array_of_displacements, array_of_types, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_struct:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_struct(count, array_of_block_lengths, array_of_displacements, array_of_types, newtype);
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%p,%i[", (void*) array_of_block_lengths, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_block_lengths[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_displacements, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_displacements[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_types, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(array_of_types[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_create_subarray (int ndims, const int size_array[], const int subsize_array[], const int start_array[], int order, MPI_Datatype oldtype, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_create_subarray(ndims, size_array, subsize_array, start_array, order, oldtype, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_create_subarray:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_create_subarray(ndims, size_array, subsize_array, start_array, order, oldtype, newtype);
    WRITE_TRACE("%lli:", (long long int) (ndims));
  WRITE_TRACE("%p,%i[", (void*) size_array, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (size_array[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) subsize_array, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (subsize_array[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) start_array, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (start_array[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) (order));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_delete_attr (MPI_Datatype type, int type_keyval) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_delete_attr(type, type_keyval);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_delete_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) (type_keyval));
  pmpi_retval = PMPI_Type_delete_attr(type, type_keyval);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_dup (MPI_Datatype type, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_dup(type, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_dup:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_dup(type, newtype);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_free (MPI_Datatype * type) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_free(type);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*type));
  pmpi_retval = PMPI_Type_free(type);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_free_keyval (int * type_keyval) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_free_keyval(type_keyval);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_free_keyval:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_free_keyval(type_keyval);
    WRITE_TRACE("%lli:", (long long int) *(type_keyval));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_get_attr (MPI_Datatype type, int type_keyval, void * attribute_val, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_get_attr(type, type_keyval, attribute_val, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_get_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_get_attr(type, type_keyval, attribute_val, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) (type_keyval));
  WRITE_TRACE("%p:", attribute_val);
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_get_contents (MPI_Datatype mtype, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_get_contents(mtype, max_integers, max_addresses, max_datatypes, array_of_integers, array_of_addresses, array_of_datatypes);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_get_contents:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_get_contents(mtype, max_integers, max_addresses, max_datatypes, array_of_integers, array_of_addresses, array_of_datatypes);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(mtype));
    WRITE_TRACE("%lli:", (long long int) (max_integers));
    WRITE_TRACE("%lli:", (long long int) (max_addresses));
    WRITE_TRACE("%lli:", (long long int) (max_datatypes));
  WRITE_TRACE("%p,%i[", (void*) array_of_integers, (int) max_integers);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<max_integers; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_integers[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_addresses, (int) max_addresses);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<max_addresses; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_addresses[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_datatypes, (int) max_datatypes);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<max_datatypes; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Type_c2f(array_of_datatypes[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_get_envelope (MPI_Datatype type, int * num_integers, int * num_addresses, int * num_datatypes, int * combiner) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_get_envelope(type, num_integers, num_addresses, num_datatypes, combiner);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_get_envelope:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_get_envelope(type, num_integers, num_addresses, num_datatypes, combiner);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) *(num_integers));
    WRITE_TRACE("%lli:", (long long int) *(num_addresses));
    WRITE_TRACE("%lli:", (long long int) *(num_datatypes));
    WRITE_TRACE("%lli:", (long long int) *(combiner));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_get_extent (MPI_Datatype type, MPI_Aint * lb, MPI_Aint * extent) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_get_extent(type, lb, extent);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_get_extent:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_get_extent(type, lb, extent);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) *(lb));
    WRITE_TRACE("%lli:", (long long int) *(extent));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_get_extent_x (MPI_Datatype type, MPI_Count * lb, MPI_Count * extent) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_get_extent_x(type, lb, extent);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_get_extent_x:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_get_extent_x(type, lb, extent);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) *(lb));
    WRITE_TRACE("%lli:", (long long int) *(extent));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_get_name (MPI_Datatype type, char * type_name, int * resultlen) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_get_name(type, type_name, resultlen);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_get_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_get_name(type, type_name, resultlen);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
  WRITE_TRACE("%p:", type_name);
    WRITE_TRACE("%lli:", (long long int) *(resultlen));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_get_true_extent (MPI_Datatype datatype, MPI_Aint * true_lb, MPI_Aint * true_extent) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_get_true_extent(datatype, true_lb, true_extent);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_get_true_extent:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_get_true_extent(datatype, true_lb, true_extent);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) *(true_lb));
    WRITE_TRACE("%lli:", (long long int) *(true_extent));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_get_true_extent_x (MPI_Datatype datatype, MPI_Count * true_lb, MPI_Count * true_extent) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_get_true_extent_x(datatype, true_lb, true_extent);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_get_true_extent_x:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_get_true_extent_x(datatype, true_lb, true_extent);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) *(true_lb));
    WRITE_TRACE("%lli:", (long long int) *(true_extent));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_indexed (int count, const int array_of_blocklengths[], const int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_indexed(count, array_of_blocklengths, array_of_displacements, oldtype, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_indexed:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_indexed(count, array_of_blocklengths, array_of_displacements, oldtype, newtype);
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%p,%i[", (void*) array_of_blocklengths, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_blocklengths[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_displacements, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_displacements[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_match_size (int typeclass, int size, MPI_Datatype * type) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_match_size(typeclass, size, type);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_match_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_match_size(typeclass, size, type);
    WRITE_TRACE("%lli:", (long long int) (typeclass));
    WRITE_TRACE("%lli:", (long long int) (size));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*type));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_set_attr (MPI_Datatype type, int type_keyval, void * attr_val) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_set_attr(type, type_keyval, attr_val);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_set_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_set_attr(type, type_keyval, attr_val);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) (type_keyval));
  WRITE_TRACE("%p:", attr_val);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_set_name (MPI_Datatype type, const char * type_name) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_set_name(type, type_name);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_set_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_set_name(type, type_name);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
  WRITE_TRACE("%p:", type_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_size (MPI_Datatype type, int * size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_size(type, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_size(type, size);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) *(size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_size_x (MPI_Datatype type, MPI_Count * size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_size_x(type, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_size_x:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_size_x(type, size);
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(type));
    WRITE_TRACE("%lli:", (long long int) *(size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Type_vector (int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype * newtype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Type_vector(count, blocklength, stride, oldtype, newtype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Type_vector:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Type_vector(count, blocklength, stride, oldtype, newtype);
    WRITE_TRACE("%lli:", (long long int) (count));
    WRITE_TRACE("%lli:", (long long int) (blocklength));
    WRITE_TRACE("%lli:", (long long int) (stride));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(oldtype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(*newtype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Unpack (const void * inbuf, int insize, int * position, void * outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Unpack(inbuf, insize, position, outbuf, outcount, datatype, comm);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Unpack:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Unpack(inbuf, insize, position, outbuf, outcount, datatype, comm);
  WRITE_TRACE("%p:", inbuf);
    WRITE_TRACE("%lli:", (long long int) (insize));
    WRITE_TRACE("%lli:", (long long int) *(position));
  WRITE_TRACE("%p:", outbuf);
    WRITE_TRACE("%lli:", (long long int) (outcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Unpack_external (const char datarep[], const void * inbuf, MPI_Aint insize, MPI_Aint * position, void * outbuf, int outcount, MPI_Datatype datatype) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Unpack_external(datarep, inbuf, insize, position, outbuf, outcount, datatype);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Unpack_external:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Unpack_external(datarep, inbuf, insize, position, outbuf, outcount, datatype);
  WRITE_TRACE("%p,%i[", (void*) datarep, (int) strlen(datarep));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<strlen(datarep); trace_elem_idx++) {
    WRITE_TRACE("%c;", datarep[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p:", inbuf);
    WRITE_TRACE("%lli:", (long long int) (insize));
    WRITE_TRACE("%lli:", (long long int) *(position));
  WRITE_TRACE("%p:", outbuf);
    WRITE_TRACE("%lli:", (long long int) (outcount));
    WRITE_TRACE("%lli:", (long long int) PMPI_Type_c2f(datatype));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Unpublish_name (const char * service_name, MPI_Info info, const char * port_name) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Unpublish_name(service_name, info, port_name);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Unpublish_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Unpublish_name(service_name, info, port_name);
  WRITE_TRACE("%p:", service_name);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%p:", port_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Wait (MPI_Request * request, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Wait(request, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Wait:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Wait(request, status);
    WRITE_TRACE("%lli:", (long long int) PMPI_Request_c2f(*request));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Waitall (int count, MPI_Request array_of_requests[], MPI_Status * array_of_statuses) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Waitall(count, array_of_requests, array_of_statuses);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Waitall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Waitall(count, array_of_requests, array_of_statuses);
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%p,%i[", (void*) array_of_requests, (int) (array_of_statuses != MPI_STATUSES_IGNORE ? count : 0));
  if (lap_elem_tracing_enabled == 0)
  {
  }
  else
  {
    for (int trace_elem_idx=0; trace_elem_idx<(array_of_statuses != MPI_STATUSES_IGNORE ? count : 0); trace_elem_idx++)
    {
      WRITE_TRACE("%lli;", (long long int) PMPI_Request_c2f(array_of_requests[trace_elem_idx]));
    }
    WRITE_TRACE("]%s", ":");
  }

  /* if ((array_of_statuses == MPI_STATUSES_IGNORE) || (array_of_statuses == MPI_STATUSES_IGNORE)) */
  /* { */
  /*   WRITE_TRACE("%lli", (long long int) array_of_statuses); */
  /* } */
  /* else */
  /* { */
  /*   WRITE_TRACE("%p:[%i,%i,%i]:", array_of_statuses, (array_of_statuses)->MPI_SOURCE, (array_of_statuses)->MPI_TAG, (array_of_statuses)->MPI_ERROR); */
  /* } */
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  if (lap_backtrace_enabled)
  {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Waitany (int count, MPI_Request array_of_requests[], int * index, MPI_Status * status) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Waitany(count, array_of_requests, index, status);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Waitany:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Waitany(count, array_of_requests, index, status);
    WRITE_TRACE("%lli:", (long long int) (count));
  WRITE_TRACE("%p,%i[", (void*) array_of_requests, (int) count);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<count; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Request_c2f(array_of_requests[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *(index));
  if ((status == MPI_STATUSES_IGNORE) || (status == MPI_STATUSES_IGNORE)) {WRITE_TRACE("%lli", (long long int) status);} else {WRITE_TRACE("%p:[%i,%i,%i]:", status, (status)->MPI_SOURCE, (status)->MPI_TAG, (status)->MPI_ERROR);}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Waitsome (int incount, MPI_Request array_of_requests[], int * outcount, int array_of_indices[], MPI_Status array_of_statuses[]) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Waitsome:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
    WRITE_TRACE("%lli:", (long long int) (incount));
  WRITE_TRACE("%p,%i[", (void*) array_of_requests, (int) incount);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<incount; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) PMPI_Request_c2f(array_of_requests[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *(outcount));
  WRITE_TRACE("%p,%i[", (void*) array_of_indices, (int) *outcount);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<*outcount; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) (array_of_indices[trace_elem_idx]));
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_statuses, (int) (array_of_statuses != MPI_STATUSES_IGNORE ? *outcount : 0));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(array_of_statuses != MPI_STATUSES_IGNORE ? *outcount : 0); trace_elem_idx++) {
    WRITE_TRACE("%p:[%i,%i,%i];", array_of_statuses[trace_elem_idx], (&array_of_statuses[trace_elem_idx])->MPI_SOURCE, (&array_of_statuses[trace_elem_idx])->MPI_TAG, (&array_of_statuses[trace_elem_idx])->MPI_ERROR);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_allocate (MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void * baseptr, MPI_Win * win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_allocate(size, disp_unit, info, comm, baseptr, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_allocate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_allocate(size, disp_unit, info, comm, baseptr, win);
    WRITE_TRACE("%lli:", (long long int) (size));
    WRITE_TRACE("%lli:", (long long int) (disp_unit));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%p:", baseptr);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(*win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_allocate_shared (MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void * baseptr, MPI_Win * win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_allocate_shared(size, disp_unit, info, comm, baseptr, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_allocate_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_allocate_shared(size, disp_unit, info, comm, baseptr, win);
    WRITE_TRACE("%lli:", (long long int) (size));
    WRITE_TRACE("%lli:", (long long int) (disp_unit));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
  WRITE_TRACE("%p:", baseptr);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(*win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_attach (MPI_Win win, void * base, MPI_Aint size) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_attach(win, base, size);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_attach:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_attach(win, base, size);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%p:", base);
    WRITE_TRACE("%lli:", (long long int) (size));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_call_errhandler (MPI_Win win, int errorcode) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_call_errhandler(win, errorcode);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_call_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_call_errhandler(win, errorcode);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) (errorcode));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_complete (MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_complete(win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_complete:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_complete(win);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_create (void * base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win * win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_create(base, size, disp_unit, info, comm, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_create(base, size, disp_unit, info, comm, win);
  WRITE_TRACE("%p:", base);
    WRITE_TRACE("%lli:", (long long int) (size));
    WRITE_TRACE("%lli:", (long long int) (disp_unit));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(*win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_create_dynamic (MPI_Info info, MPI_Comm comm, MPI_Win * win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_create_dynamic(info, comm, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_create_dynamic:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_create_dynamic(info, comm, win);
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
    WRITE_TRACE("%lli:", (long long int) PMPI_Comm_c2f(comm));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(*win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_create_errhandler (MPI_Win_errhandler_function * function, MPI_Errhandler * errhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_create_errhandler(function, errhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_create_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_create_errhandler(function, errhandler);
  WRITE_TRACE("%p:", function);
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(*errhandler));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_create_keyval (MPI_Win_copy_attr_function * win_copy_attr_fn, MPI_Win_delete_attr_function * win_delete_attr_fn, int * win_keyval, void * extra_state) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_create_keyval(win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_create_keyval:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_create_keyval(win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state);
  WRITE_TRACE("%p:", win_copy_attr_fn);
  WRITE_TRACE("%p:", win_delete_attr_fn);
    WRITE_TRACE("%lli:", (long long int) *(win_keyval));
  WRITE_TRACE("%p:", extra_state);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_delete_attr (MPI_Win win, int win_keyval) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_delete_attr(win, win_keyval);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_delete_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) (win_keyval));
  pmpi_retval = PMPI_Win_delete_attr(win, win_keyval);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_detach (MPI_Win win, const void * base) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_detach(win, base);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_detach:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_detach(win, base);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%p:", base);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_fence (int assert, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_fence(assert, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_fence:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_fence(assert, win);
    WRITE_TRACE("%lli:", (long long int) (assert));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_flush (int rank, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_flush(rank, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_flush:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_flush(rank, win);
    WRITE_TRACE("%lli:", (long long int) (rank));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_flush_all (MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_flush_all(win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_flush_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_flush_all(win);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_flush_local (int rank, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_flush_local(rank, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_flush_local:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_flush_local(rank, win);
    WRITE_TRACE("%lli:", (long long int) (rank));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_flush_local_all (MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_flush_local_all(win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_flush_local_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_flush_local_all(win);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_free (MPI_Win * win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_free(win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(*win));
  pmpi_retval = PMPI_Win_free(win);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_free_keyval (int * win_keyval) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_free_keyval(win_keyval);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_free_keyval:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_free_keyval(win_keyval);
    WRITE_TRACE("%lli:", (long long int) *(win_keyval));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_get_attr (MPI_Win win, int win_keyval, void * attribute_val, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_get_attr(win, win_keyval, attribute_val, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_get_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_get_attr(win, win_keyval, attribute_val, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) (win_keyval));
  WRITE_TRACE("%p:", attribute_val);
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_get_errhandler (MPI_Win win, MPI_Errhandler * errhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_get_errhandler(win, errhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_get_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_get_errhandler(win, errhandler);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(*errhandler));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_get_group (MPI_Win win, MPI_Group * group) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_get_group(win, group);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_get_group:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_get_group(win, group);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(*group));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_get_info (MPI_Win win, MPI_Info * info_used) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_get_info(win, info_used);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_get_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_get_info(win, info_used);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(*info_used));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_get_name (MPI_Win win, char * win_name, int * resultlen) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_get_name(win, win_name, resultlen);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_get_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_get_name(win, win_name, resultlen);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%p:", win_name);
    WRITE_TRACE("%lli:", (long long int) *(resultlen));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_lock (int lock_type, int rank, int assert, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_lock(lock_type, rank, assert, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_lock:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_lock(lock_type, rank, assert, win);
    WRITE_TRACE("%lli:", (long long int) (lock_type));
    WRITE_TRACE("%lli:", (long long int) (rank));
    WRITE_TRACE("%lli:", (long long int) (assert));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_lock_all (int assert, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_lock_all(assert, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_lock_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_lock_all(assert, win);
    WRITE_TRACE("%lli:", (long long int) (assert));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_post (MPI_Group group, int assert, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_post(group, assert, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_post:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_post(group, assert, win);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) (assert));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_set_attr (MPI_Win win, int win_keyval, void * attribute_val) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_set_attr(win, win_keyval, attribute_val);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_set_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_set_attr(win, win_keyval, attribute_val);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) (win_keyval));
  WRITE_TRACE("%p:", attribute_val);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_set_errhandler (MPI_Win win, MPI_Errhandler errhandler) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_set_errhandler(win, errhandler);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_set_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_set_errhandler(win, errhandler);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) PMPI_Errhandler_c2f(errhandler));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_set_info (MPI_Win win, MPI_Info info) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_set_info(win, info);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_set_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_set_info(win, info);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) PMPI_Info_c2f(info));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_set_name (MPI_Win win, const char * win_name) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_set_name(win, win_name);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_set_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_set_name(win, win_name);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%p:", win_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_shared_query (MPI_Win win, int rank, MPI_Aint * size, int * disp_unit, void * baseptr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_shared_query(win, rank, size, disp_unit, baseptr);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_shared_query:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_shared_query(win, rank, size, disp_unit, baseptr);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) (rank));
    WRITE_TRACE("%lli:", (long long int) *(size));
    WRITE_TRACE("%lli:", (long long int) *(disp_unit));
  WRITE_TRACE("%p:", baseptr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_start (MPI_Group group, int assert, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_start(group, assert, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_start:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_start(group, assert, win);
    WRITE_TRACE("%lli:", (long long int) PMPI_Group_c2f(group));
    WRITE_TRACE("%lli:", (long long int) (assert));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_sync (MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_sync(win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_sync:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_sync(win);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_test (MPI_Win win, int * flag) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_test(win, flag);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_test:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_test(win, flag);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
    WRITE_TRACE("%lli:", (long long int) *(flag));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_unlock (int rank, MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_unlock(rank, win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_unlock:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_unlock(rank, win);
    WRITE_TRACE("%lli:", (long long int) (rank));
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_unlock_all (MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_unlock_all(win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_unlock_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_unlock_all(win);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

int MPI_Win_wait (MPI_Win win) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval;  pmpi_retval = PMPI_Win_wait(win);
    return pmpi_retval;
  }
  int pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Win_wait:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Win_wait(win);
    WRITE_TRACE("%lli:", (long long int) PMPI_Win_c2f(win));
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

double MPI_Wtick () {
  if (lap_tracing_enabled == 0) { 
    double pmpi_retval;  pmpi_retval = PMPI_Wtick();
    return pmpi_retval;
  }
  double pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Wtick:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Wtick();
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

double MPI_Wtime () {
  if (lap_tracing_enabled == 0) { 
    double pmpi_retval;  pmpi_retval = PMPI_Wtime();
    return pmpi_retval;
  }
  double pmpi_retval;
  lap_check();
  WRITE_TRACE("%s", "MPI_Wtime:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  pmpi_retval = PMPI_Wtime();
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  return pmpi_retval;
}

