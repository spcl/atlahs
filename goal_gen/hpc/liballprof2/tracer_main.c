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
int lap_backtrace_enabled = 0;
int lap_elem_tracing_enabled = 1;


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

int main(int argc, char** argv) { 
  // this exists just for debugging, we paste this file into the generated code
  // main must be the last function in this file and will be cut!
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  lap_check();
  for (int i=0; i<42*size+rank*rank; i++) {
    WRITE_TRACE("Rank %i of %i line %i\n", rank, size, i);
  }
  lap_collect_traces();
  MPI_Finalize();
}
