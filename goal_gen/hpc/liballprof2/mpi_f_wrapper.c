#include "fc_mangle.h"
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#define UNW_LOCAL_ONLY //we do not need to unwind frames in another process
#include <libunwind.h>
#include <linux/limits.h>
#include <errno.h>

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

    printf("[DEBUG] Trace size: %i\n", trace_size);
    char trace_fname[PATH_MAX];
    char *env = getenv("HTOR_PMPI_FILE_PREFIX");
    if (env == NULL)
      snprintf(trace_fname, PATH_MAX, "/scratch/sshen/traces/lap2-trace-rank-%i.txt", comm_rank);
    else
    {
      snprintf(trace_fname, PATH_MAX, "%s%i.txt", env, comm_rank);
    }
    FILE* trace_fh = fopen(trace_fname, "w");
    if (!trace_fh)
    {
      printf("[ERROR] Failed to open file: %s\n", trace_fname);
      printf("[ERROR] Errno: %d\n", errno);
    }

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

void FortranCInterface_GLOBAL(pmpi_abort,PMPI_ABORT) (int* comm, int* errorcode, int* ierr);
void FortranCInterface_GLOBAL(pmpi_accumulate,PMPI_ACCUMULATE) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* op, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_add_error_class,PMPI_ADD_ERROR_CLASS) (int* errorclass, int* ierr);
void FortranCInterface_GLOBAL(pmpi_add_error_code,PMPI_ADD_ERROR_CODE) (int* errorclass, int* errorcode, int* ierr);
void FortranCInterface_GLOBAL(pmpi_add_error_string,PMPI_ADD_ERROR_STRING) (int* errorcode, char* string, int* ierr);
void FortranCInterface_GLOBAL(pmpi_allgather,PMPI_ALLGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_allgatherv,PMPI_ALLGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_alloc_mem,PMPI_ALLOC_MEM) (int* size, int* info, int* baseptr, int* ierr);
void FortranCInterface_GLOBAL(pmpi_allreduce,PMPI_ALLREDUCE) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_alltoall,PMPI_ALLTOALL) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_alltoallv,PMPI_ALLTOALLV) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtype, int* recvbuf, int* recvcounts, int* rdispls, int* recvtype, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_alltoallw,PMPI_ALLTOALLW) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtypes, int* recvbuf, int* recvcounts, int* rdispls, int* recvtypes, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_attr_delete,PMPI_ATTR_DELETE) (int* comm, int* keyval, int* ierr);
void FortranCInterface_GLOBAL(pmpi_attr_get,PMPI_ATTR_GET) (int* comm, int* keyval, int* attribute_val, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_attr_put,PMPI_ATTR_PUT) (int* comm, int* keyval, int* attribute_val, int* ierr);
void FortranCInterface_GLOBAL(pmpi_barrier,PMPI_BARRIER) (int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_bcast,PMPI_BCAST) (int* buffer, int* count, int* datatype, int* root, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_bsend,PMPI_BSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_bsend_init,PMPI_BSEND_INIT) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_buffer_attach,PMPI_BUFFER_ATTACH) (int* buffer, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_buffer_detach,PMPI_BUFFER_DETACH) (int* buffer, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_cancel,PMPI_CANCEL) (int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_cart_coords,PMPI_CART_COORDS) (int* comm, int* rank, int* maxdims, int* coords, int* ierr);
void FortranCInterface_GLOBAL(pmpi_cart_create,PMPI_CART_CREATE) (int* old_comm, int* ndims, int* dims, int* periods, int* reorder, int* comm_cart, int* ierr);
void FortranCInterface_GLOBAL(pmpi_cart_get,PMPI_CART_GET) (int* comm, int* maxdims, int* dims, int* periods, int* coords, int* ierr);
void FortranCInterface_GLOBAL(pmpi_cart_map,PMPI_CART_MAP) (int* comm, int* ndims, int* dims, int* periods, int* newrank, int* ierr);
void FortranCInterface_GLOBAL(pmpi_cart_rank,PMPI_CART_RANK) (int* comm, int* coords, int* rank, int* ierr);
void FortranCInterface_GLOBAL(pmpi_cart_shift,PMPI_CART_SHIFT) (int* comm, int* direction, int* disp, int* rank_source, int* rank_dest, int* ierr);
void FortranCInterface_GLOBAL(pmpi_cart_sub,PMPI_CART_SUB) (int* comm, int* remain_dims, int* new_comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_cartdim_get,PMPI_CARTDIM_GET) (int* comm, int* ndims, int* ierr);
void FortranCInterface_GLOBAL(pmpi_close_port,PMPI_CLOSE_PORT) (char* port_name, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_accept,PMPI_COMM_ACCEPT) (char* port_name, int* info, int* root, int* comm, int* newcomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_call_errhandler,PMPI_COMM_CALL_ERRHANDLER) (int* comm, int* errorcode, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_compare,PMPI_COMM_COMPARE) (int* comm1, int* comm2, int* result, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_connect,PMPI_COMM_CONNECT) (char* port_name, int* info, int* root, int* comm, int* newcomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_create,PMPI_COMM_CREATE) (int* comm, int* group, int* newcomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_create_errhandler,PMPI_COMM_CREATE_ERRHANDLER) (int* function, int* errhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_create_group,PMPI_COMM_CREATE_GROUP) (int* comm, int* group, int* tag, int* newcomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_create_keyval,PMPI_COMM_CREATE_KEYVAL) (int* comm_copy_attr_fn, int* comm_delete_attr_fn, int* comm_keyval, int* extra_state, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_delete_attr,PMPI_COMM_DELETE_ATTR) (int* comm, int* comm_keyval, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_disconnect,PMPI_COMM_DISCONNECT) (int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_dup,PMPI_COMM_DUP) (int* comm, int* newcomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_dup_with_info,PMPI_COMM_DUP_WITH_INFO) (int* comm, int* info, int* newcomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_free,PMPI_COMM_FREE) (int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_free_keyval,PMPI_COMM_FREE_KEYVAL) (int* comm_keyval, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_get_attr,PMPI_COMM_GET_ATTR) (int* comm, int* comm_keyval, int* attribute_val, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_get_errhandler,PMPI_COMM_GET_ERRHANDLER) (int* comm, int* erhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_get_info,PMPI_COMM_GET_INFO) (int* comm, int* info_used, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_get_name,PMPI_COMM_GET_NAME) (int* comm, char* comm_name, int* resultlen, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_get_parent,PMPI_COMM_GET_PARENT) (int* parent, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_group,PMPI_COMM_GROUP) (int* comm, int* group, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_idup,PMPI_COMM_IDUP) (int* comm, int* newcomm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_join,PMPI_COMM_JOIN) (int* fd, int* intercomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_rank,PMPI_COMM_RANK) (int* comm, int* rank, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_remote_group,PMPI_COMM_REMOTE_GROUP) (int* comm, int* group, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_remote_size,PMPI_COMM_REMOTE_SIZE) (int* comm, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_set_attr,PMPI_COMM_SET_ATTR) (int* comm, int* comm_keyval, int* attribute_val, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_set_errhandler,PMPI_COMM_SET_ERRHANDLER) (int* comm, int* errhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_set_info,PMPI_COMM_SET_INFO) (int* comm, int* info, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_set_name,PMPI_COMM_SET_NAME) (int* comm, char* comm_name, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_size,PMPI_COMM_SIZE) (int* comm, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_spawn,PMPI_COMM_SPAWN) (char* command, char* argv, int* maxprocs, int* info, int* root, int* comm, int* intercomm, int* array_of_errcodes, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_spawn_multiple,PMPI_COMM_SPAWN_MULTIPLE) (int* count, char* array_of_commands, char* array_of_argv, int* array_of_maxprocs, int* array_of_info, int* root, int* comm, int* intercomm, int* array_of_errcodes, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_split,PMPI_COMM_SPLIT) (int* comm, int* color, int* key, int* newcomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_split_type,PMPI_COMM_SPLIT_TYPE) (int* comm, int* split_type, int* key, int* info, int* newcomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_comm_test_inter,PMPI_COMM_TEST_INTER) (int* comm, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_compare_and_swap,PMPI_COMPARE_AND_SWAP) (int* origin_addr, int* compare_addr, int* result_addr, int* datatype, int* target_rank, int* target_disp, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_dims_create,PMPI_DIMS_CREATE) (int* nnodes, int* ndims, int* dims, int* ierr);
void FortranCInterface_GLOBAL(pmpi_dist_graph_create,PMPI_DIST_GRAPH_CREATE) (int* comm_old, int* n, int* nodes, int* degrees, int* targets, int* weights, int* info, int* reorder, int* newcomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_dist_graph_create_adjacent,PMPI_DIST_GRAPH_CREATE_ADJACENT) (int* comm_old, int* indegree, int* sources, int* sourceweights, int* outdegree, int* destinations, int* destweights, int* info, int* reorder, int* comm_dist_graph, int* ierr);
void FortranCInterface_GLOBAL(pmpi_dist_graph_neighbors,PMPI_DIST_GRAPH_NEIGHBORS) (int* comm, int* maxindegree, int* sources, int* sourceweights, int* maxoutdegree, int* destinations, int* destweights, int* ierr);
void FortranCInterface_GLOBAL(pmpi_dist_graph_neighbors_count,PMPI_DIST_GRAPH_NEIGHBORS_COUNT) (int* comm, int* inneighbors, int* outneighbors, int* weighted, int* ierr);
void FortranCInterface_GLOBAL(pmpi_errhandler_free,PMPI_ERRHANDLER_FREE) (int* errhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_error_class,PMPI_ERROR_CLASS) (int* errorcode, int* errorclass, int* ierr);
void FortranCInterface_GLOBAL(pmpi_error_string,PMPI_ERROR_STRING) (int* errorcode, char* string, int* resultlen, int* ierr);
void FortranCInterface_GLOBAL(pmpi_exscan,PMPI_EXSCAN) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_fetch_and_op,PMPI_FETCH_AND_OP) (int* origin_addr, int* result_addr, int* datatype, int* target_rank, int* target_disp, int* op, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_call_errhandler,PMPI_FILE_CALL_ERRHANDLER) (int* fh, int* errorcode, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_close,PMPI_FILE_CLOSE) (int* fh, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_create_errhandler,PMPI_FILE_CREATE_ERRHANDLER) (int* function, int* errhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_delete,PMPI_FILE_DELETE) (char* filename, int* info, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_amode,PMPI_FILE_GET_AMODE) (int* fh, int* amode, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_atomicity,PMPI_FILE_GET_ATOMICITY) (int* fh, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_byte_offset,PMPI_FILE_GET_BYTE_OFFSET) (int* fh, int* offset, int* disp, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_errhandler,PMPI_FILE_GET_ERRHANDLER) (int* file, int* errhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_group,PMPI_FILE_GET_GROUP) (int* fh, int* group, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_info,PMPI_FILE_GET_INFO) (int* fh, int* info_used, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_position,PMPI_FILE_GET_POSITION) (int* fh, int* offset, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_position_shared,PMPI_FILE_GET_POSITION_SHARED) (int* fh, int* offset, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_size,PMPI_FILE_GET_SIZE) (int* fh, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_type_extent,PMPI_FILE_GET_TYPE_EXTENT) (int* fh, int* datatype, int* extent, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_get_view,PMPI_FILE_GET_VIEW) (int* fh, int* disp, int* etype, int* filetype, char* datarep, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iread,PMPI_FILE_IREAD) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iread_all,PMPI_FILE_IREAD_ALL) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iread_at,PMPI_FILE_IREAD_AT) (int* fh, int* offset, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iread_at_all,PMPI_FILE_IREAD_AT_ALL) (int* fh, int* offset, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iread_shared,PMPI_FILE_IREAD_SHARED) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iwrite,PMPI_FILE_IWRITE) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iwrite_all,PMPI_FILE_IWRITE_ALL) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iwrite_at,PMPI_FILE_IWRITE_AT) (int* fh, int* offset, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iwrite_at_all,PMPI_FILE_IWRITE_AT_ALL) (int* fh, int* offset, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_iwrite_shared,PMPI_FILE_IWRITE_SHARED) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_open,PMPI_FILE_OPEN) (int* comm, char* filename, int* amode, int* info, int* fh, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_preallocate,PMPI_FILE_PREALLOCATE) (int* fh, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read,PMPI_FILE_READ) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_all,PMPI_FILE_READ_ALL) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_all_begin,PMPI_FILE_READ_ALL_BEGIN) (int* fh, int* buf, int* count, int* datatype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_all_end,PMPI_FILE_READ_ALL_END) (int* fh, int* buf, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_at,PMPI_FILE_READ_AT) (int* fh, int* offset, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_at_all,PMPI_FILE_READ_AT_ALL) (int* fh, int* offset, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_at_all_begin,PMPI_FILE_READ_AT_ALL_BEGIN) (int* fh, int* offset, int* buf, int* count, int* datatype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_at_all_end,PMPI_FILE_READ_AT_ALL_END) (int* fh, int* buf, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_ordered,PMPI_FILE_READ_ORDERED) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_ordered_begin,PMPI_FILE_READ_ORDERED_BEGIN) (int* fh, int* buf, int* count, int* datatype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_ordered_end,PMPI_FILE_READ_ORDERED_END) (int* fh, int* buf, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_read_shared,PMPI_FILE_READ_SHARED) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_seek,PMPI_FILE_SEEK) (int* fh, int* offset, int* whence, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_seek_shared,PMPI_FILE_SEEK_SHARED) (int* fh, int* offset, int* whence, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_set_atomicity,PMPI_FILE_SET_ATOMICITY) (int* fh, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_set_errhandler,PMPI_FILE_SET_ERRHANDLER) (int* file, int* errhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_set_info,PMPI_FILE_SET_INFO) (int* fh, int* info, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_set_size,PMPI_FILE_SET_SIZE) (int* fh, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_set_view,PMPI_FILE_SET_VIEW) (int* fh, int* disp, int* etype, int* filetype, char* datarep, int* info, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_sync,PMPI_FILE_SYNC) (int* fh, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write,PMPI_FILE_WRITE) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_all,PMPI_FILE_WRITE_ALL) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_all_begin,PMPI_FILE_WRITE_ALL_BEGIN) (int* fh, int* buf, int* count, int* datatype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_all_end,PMPI_FILE_WRITE_ALL_END) (int* fh, int* buf, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_at,PMPI_FILE_WRITE_AT) (int* fh, int* offset, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_at_all,PMPI_FILE_WRITE_AT_ALL) (int* fh, int* offset, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_at_all_begin,PMPI_FILE_WRITE_AT_ALL_BEGIN) (int* fh, int* offset, int* buf, int* count, int* datatype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_at_all_end,PMPI_FILE_WRITE_AT_ALL_END) (int* fh, int* buf, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_ordered,PMPI_FILE_WRITE_ORDERED) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_ordered_begin,PMPI_FILE_WRITE_ORDERED_BEGIN) (int* fh, int* buf, int* count, int* datatype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_ordered_end,PMPI_FILE_WRITE_ORDERED_END) (int* fh, int* buf, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_file_write_shared,PMPI_FILE_WRITE_SHARED) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_finalize,PMPI_FINALIZE) (int* ierr);
void FortranCInterface_GLOBAL(pmpi_finalized,PMPI_FINALIZED) (int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_free_mem,PMPI_FREE_MEM) (int* base, int* ierr);
void FortranCInterface_GLOBAL(pmpi_gather,PMPI_GATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_gatherv,PMPI_GATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* root, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_get,PMPI_GET) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_get_accumulate,PMPI_GET_ACCUMULATE) (int* origin_addr, int* origin_count, int* origin_datatype, int* result_addr, int* result_count, int* result_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* op, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_get_address,PMPI_GET_ADDRESS) (int* location, int* address, int* ierr);
void FortranCInterface_GLOBAL(pmpi_get_count,PMPI_GET_COUNT) (int* status, int* datatype, int* count, int* ierr);
void FortranCInterface_GLOBAL(pmpi_get_elements,PMPI_GET_ELEMENTS) (int* status, int* datatype, int* count, int* ierr);
void FortranCInterface_GLOBAL(pmpi_get_elements_x,PMPI_GET_ELEMENTS_X) (int* status, int* datatype, int* count, int* ierr);
void FortranCInterface_GLOBAL(pmpi_get_library_version,PMPI_GET_LIBRARY_VERSION) (char* version, int* resultlen, int* ierr);
void FortranCInterface_GLOBAL(pmpi_get_processor_name,PMPI_GET_PROCESSOR_NAME) (char* name, int* resultlen, int* ierr);
void FortranCInterface_GLOBAL(pmpi_get_version,PMPI_GET_VERSION) (int* version, int* subversion, int* ierr);
void FortranCInterface_GLOBAL(pmpi_graph_create,PMPI_GRAPH_CREATE) (int* comm_old, int* nnodes, int* index, int* edges, int* reorder, int* comm_graph, int* ierr);
void FortranCInterface_GLOBAL(pmpi_graph_get,PMPI_GRAPH_GET) (int* comm, int* maxindex, int* maxedges, int* index, int* edges, int* ierr);
void FortranCInterface_GLOBAL(pmpi_graph_map,PMPI_GRAPH_MAP) (int* comm, int* nnodes, int* index, int* edges, int* newrank, int* ierr);
void FortranCInterface_GLOBAL(pmpi_graph_neighbors,PMPI_GRAPH_NEIGHBORS) (int* comm, int* rank, int* maxneighbors, int* neighbors, int* ierr);
void FortranCInterface_GLOBAL(pmpi_graph_neighbors_count,PMPI_GRAPH_NEIGHBORS_COUNT) (int* comm, int* rank, int* nneighbors, int* ierr);
void FortranCInterface_GLOBAL(pmpi_graphdims_get,PMPI_GRAPHDIMS_GET) (int* comm, int* nnodes, int* nedges, int* ierr);
void FortranCInterface_GLOBAL(pmpi_grequest_complete,PMPI_GREQUEST_COMPLETE) (int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_grequest_start,PMPI_GREQUEST_START) (int* query_fn, int* free_fn, int* cancel_fn, int* extra_state, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_compare,PMPI_GROUP_COMPARE) (int* group1, int* group2, int* result, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_difference,PMPI_GROUP_DIFFERENCE) (int* group1, int* group2, int* newgroup, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_excl,PMPI_GROUP_EXCL) (int* group, int* n, int* ranks, int* newgroup, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_free,PMPI_GROUP_FREE) (int* group, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_incl,PMPI_GROUP_INCL) (int* group, int* n, int* ranks, int* newgroup, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_intersection,PMPI_GROUP_INTERSECTION) (int* group1, int* group2, int* newgroup, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_range_excl,PMPI_GROUP_RANGE_EXCL) (int* group, int* n, int* ranges, int* newgroup, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_range_incl,PMPI_GROUP_RANGE_INCL) (int* group, int* n, int* ranges, int* newgroup, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_rank,PMPI_GROUP_RANK) (int* group, int* rank, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_size,PMPI_GROUP_SIZE) (int* group, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_translate_ranks,PMPI_GROUP_TRANSLATE_RANKS) (int* group1, int* n, int* ranks1, int* group2, int* ranks2, int* ierr);
void FortranCInterface_GLOBAL(pmpi_group_union,PMPI_GROUP_UNION) (int* group1, int* group2, int* newgroup, int* ierr);
void FortranCInterface_GLOBAL(pmpi_iallgather,PMPI_IALLGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_iallgatherv,PMPI_IALLGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_iallreduce,PMPI_IALLREDUCE) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ialltoall,PMPI_IALLTOALL) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ialltoallv,PMPI_IALLTOALLV) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtype, int* recvbuf, int* recvcounts, int* rdispls, int* recvtype, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ialltoallw,PMPI_IALLTOALLW) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtypes, int* recvbuf, int* recvcounts, int* rdispls, int* recvtypes, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ibarrier,PMPI_IBARRIER) (int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ibcast,PMPI_IBCAST) (int* buffer, int* count, int* datatype, int* root, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ibsend,PMPI_IBSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_iexscan,PMPI_IEXSCAN) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_igather,PMPI_IGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_igatherv,PMPI_IGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* root, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_improbe,PMPI_IMPROBE) (int* source, int* tag, int* comm, int* flag, int* message, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_imrecv,PMPI_IMRECV) (int* buf, int* count, int* type, int* message, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ineighbor_allgather,PMPI_INEIGHBOR_ALLGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ineighbor_allgatherv,PMPI_INEIGHBOR_ALLGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ineighbor_alltoall,PMPI_INEIGHBOR_ALLTOALL) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ineighbor_alltoallv,PMPI_INEIGHBOR_ALLTOALLV) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtype, int* recvbuf, int* recvcounts, int* rdispls, int* recvtype, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ineighbor_alltoallw,PMPI_INEIGHBOR_ALLTOALLW) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtypes, int* recvbuf, int* recvcounts, int* rdispls, int* recvtypes, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_info_create,PMPI_INFO_CREATE) (int* info, int* ierr);
void FortranCInterface_GLOBAL(pmpi_info_delete,PMPI_INFO_DELETE) (int* info, char* key, int* ierr);
void FortranCInterface_GLOBAL(pmpi_info_dup,PMPI_INFO_DUP) (int* info, int* newinfo, int* ierr);
void FortranCInterface_GLOBAL(pmpi_info_free,PMPI_INFO_FREE) (int* info, int* ierr);
void FortranCInterface_GLOBAL(pmpi_info_get,PMPI_INFO_GET) (int* info, char* key, int* valuelen, char* value, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_info_get_nkeys,PMPI_INFO_GET_NKEYS) (int* info, int* nkeys, int* ierr);
void FortranCInterface_GLOBAL(pmpi_info_get_nthkey,PMPI_INFO_GET_NTHKEY) (int* info, int* n, char* key, int* ierr);
void FortranCInterface_GLOBAL(pmpi_info_get_valuelen,PMPI_INFO_GET_VALUELEN) (int* info, char* key, int* valuelen, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_info_set,PMPI_INFO_SET) (int* info, char* key, char* value, int* ierr);
void FortranCInterface_GLOBAL(pmpi_init,PMPI_INIT) (int* ierr);
void FortranCInterface_GLOBAL(pmpi_init_thread,PMPI_INIT_THREAD) (int *required, int *provided, int *ierr);
void FortranCInterface_GLOBAL(pmpi_initialized,PMPI_INITIALIZED) (int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_intercomm_create,PMPI_INTERCOMM_CREATE) (int* local_comm, int* local_leader, int* bridge_comm, int* remote_leader, int* tag, int* newintercomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_intercomm_merge,PMPI_INTERCOMM_MERGE) (int* intercomm, int* high, int* newintercomm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_iprobe,PMPI_IPROBE) (int* source, int* tag, int* comm, int* flag, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_irecv,PMPI_IRECV) (int* buf, int* count, int* datatype, int* source, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ireduce,PMPI_IREDUCE) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* root, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ireduce_scatter,PMPI_IREDUCE_SCATTER) (int* sendbuf, int* recvbuf, int* recvcounts, int* datatype, int* op, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ireduce_scatter_block,PMPI_IREDUCE_SCATTER_BLOCK) (int* sendbuf, int* recvbuf, int* recvcount, int* datatype, int* op, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_irsend,PMPI_IRSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_is_thread_main,PMPI_IS_THREAD_MAIN) (int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_iscan,PMPI_ISCAN) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_iscatter,PMPI_ISCATTER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_iscatterv,PMPI_ISCATTERV) (int* sendbuf, int* sendcounts, int* displs, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_isend,PMPI_ISEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_issend,PMPI_ISSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_keyval_create,PMPI_KEYVAL_CREATE) (int* copy_fn, int* delete_fn, int* keyval, int* extra_state, int* ierr);
void FortranCInterface_GLOBAL(pmpi_keyval_free,PMPI_KEYVAL_FREE) (int* keyval, int* ierr);
void FortranCInterface_GLOBAL(pmpi_lookup_name,PMPI_LOOKUP_NAME) (char* service_name, int* info, char* port_name, int* ierr);
void FortranCInterface_GLOBAL(pmpi_mprobe,PMPI_MPROBE) (int* source, int* tag, int* comm, int* message, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_mrecv,PMPI_MRECV) (int* buf, int* count, int* type, int* message, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_neighbor_allgather,PMPI_NEIGHBOR_ALLGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_neighbor_allgatherv,PMPI_NEIGHBOR_ALLGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_neighbor_alltoall,PMPI_NEIGHBOR_ALLTOALL) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_neighbor_alltoallv,PMPI_NEIGHBOR_ALLTOALLV) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtype, int* recvbuf, int* recvcounts, int* rdispls, int* recvtype, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_neighbor_alltoallw,PMPI_NEIGHBOR_ALLTOALLW) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtypes, int* recvbuf, int* recvcounts, int* rdispls, int* recvtypes, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_op_commutative,PMPI_OP_COMMUTATIVE) (int* op, int* commute, int* ierr);
void FortranCInterface_GLOBAL(pmpi_op_create,PMPI_OP_CREATE) (int* function, int* commute, int* op, int* ierr);
void FortranCInterface_GLOBAL(pmpi_op_free,PMPI_OP_FREE) (int* op, int* ierr);
void FortranCInterface_GLOBAL(pmpi_open_port,PMPI_OPEN_PORT) (int* info, char* port_name, int* ierr);
void FortranCInterface_GLOBAL(pmpi_pack,PMPI_PACK) (int* inbuf, int* incount, int* datatype, int* outbuf, int* outsize, int* position, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_pack_external,PMPI_PACK_EXTERNAL) (char* datarep, int* inbuf, int* incount, int* datatype, int* outbuf, int* outsize, int* position, int* ierr);
void FortranCInterface_GLOBAL(pmpi_pack_external_size,PMPI_PACK_EXTERNAL_SIZE) (char* datarep, int* incount, int* datatype, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_pack_size,PMPI_PACK_SIZE) (int* incount, int* datatype, int* comm, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_pcontrol,PMPI_PCONTROL) (int* level);
void FortranCInterface_GLOBAL(pmpi_probe,PMPI_PROBE) (int* source, int* tag, int* comm, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_publish_name,PMPI_PUBLISH_NAME) (char* service_name, int* info, char* port_name, int* ierr);
void FortranCInterface_GLOBAL(pmpi_put,PMPI_PUT) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_query_thread,PMPI_QUERY_THREAD) (int* provided, int* ierr);
void FortranCInterface_GLOBAL(pmpi_raccumulate,PMPI_RACCUMULATE) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* op, int* win, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_recv,PMPI_RECV) (int* buf, int* count, int* datatype, int* source, int* tag, int* comm, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_recv_init,PMPI_RECV_INIT) (int* buf, int* count, int* datatype, int* source, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_reduce,PMPI_REDUCE) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* root, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_reduce_local,PMPI_REDUCE_LOCAL) (int* inbuf, int* inoutbuf, int* count, int* datatype, int* op, int* ierr);
void FortranCInterface_GLOBAL(pmpi_reduce_scatter,PMPI_REDUCE_SCATTER) (int* sendbuf, int* recvbuf, int* recvcounts, int* datatype, int* op, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_reduce_scatter_block,PMPI_REDUCE_SCATTER_BLOCK) (int* sendbuf, int* recvbuf, int* recvcount, int* datatype, int* op, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_register_datarep,PMPI_REGISTER_DATAREP) (char* datarep, int* read_conversion_fn, int* write_conversion_fn, int* dtype_file_extent_fn, int* extra_state, int* ierr);
void FortranCInterface_GLOBAL(pmpi_request_free,PMPI_REQUEST_FREE) (int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_request_get_status,PMPI_REQUEST_GET_STATUS) (int* request, int* flag, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_rget,PMPI_RGET) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* win, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_rget_accumulate,PMPI_RGET_ACCUMULATE) (int* origin_addr, int* origin_count, int* origin_datatype, int* result_addr, int* result_count, int* result_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* op, int* win, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_rput,PMPI_RPUT) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_cout, int* target_datatype, int* win, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_rsend,PMPI_RSEND) (int* ibuf, int* count, int* datatype, int* dest, int* tag, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_rsend_init,PMPI_RSEND_INIT) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_scan,PMPI_SCAN) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_scatter,PMPI_SCATTER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_scatterv,PMPI_SCATTERV) (int* sendbuf, int* sendcounts, int* displs, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_send,PMPI_SEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_send_init,PMPI_SEND_INIT) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_sendrecv,PMPI_SENDRECV) (int* sendbuf, int* sendcount, int* sendtype, int* dest, int* sendtag, int* recvbuf, int* recvcount, int* recvtype, int* source, int* recvtag, int* comm, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_sendrecv_replace,PMPI_SENDRECV_REPLACE) (int* buf, int* count, int* datatype, int* dest, int* sendtag, int* source, int* recvtag, int* comm, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ssend,PMPI_SSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_ssend_init,PMPI_SSEND_INIT) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_start,PMPI_START) (int* request, int* ierr);
void FortranCInterface_GLOBAL(pmpi_startall,PMPI_STARTALL) (int* count, int* array_of_requests, int* ierr);
void FortranCInterface_GLOBAL(pmpi_status_c2f,PMPI_STATUS_C2F) (int* c_status, int* f_status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_status_f2c,PMPI_STATUS_F2C) (int* f_status, int* c_status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_status_set_cancelled,PMPI_STATUS_SET_CANCELLED) (int* status, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_status_set_elements,PMPI_STATUS_SET_ELEMENTS) (int* status, int* datatype, int* count, int* ierr);
void FortranCInterface_GLOBAL(pmpi_status_set_elements_x,PMPI_STATUS_SET_ELEMENTS_X) (int* status, int* datatype, int* count, int* ierr);
void FortranCInterface_GLOBAL(pmpi_test,PMPI_TEST) (int* request, int* flag, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_test_cancelled,PMPI_TEST_CANCELLED) (int* status, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_testall,PMPI_TESTALL) (int* count, int* array_of_requests, int* flag, int* array_of_statuses, int* ierr);
void FortranCInterface_GLOBAL(pmpi_testany,PMPI_TESTANY) (int* count, int* array_of_requests, int* index, int* flag, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_testsome,PMPI_TESTSOME) (int* incount, int* array_of_requests, int* outcount, int* array_of_indices, int* array_of_statuses, int* ierr);
void FortranCInterface_GLOBAL(pmpi_topo_test,PMPI_TOPO_TEST) (int* comm, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_commit,PMPI_TYPE_COMMIT) (int* type, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_contiguous,PMPI_TYPE_CONTIGUOUS) (int* count, int* oldtype, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_darray,PMPI_TYPE_CREATE_DARRAY) (int* size, int* rank, int* ndims, int* gsize_array, int* distrib_array, int* darg_array, int* psize_array, int* order, int* oldtype, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_f90_complex,PMPI_TYPE_CREATE_F90_COMPLEX) (int* p, int* r, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_f90_integer,PMPI_TYPE_CREATE_F90_INTEGER) (int* r, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_f90_real,PMPI_TYPE_CREATE_F90_REAL) (int* p, int* r, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_hindexed,PMPI_TYPE_CREATE_HINDEXED) (int* count, int* array_of_blocklengths, int* array_of_displacements, int* oldtype, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_hindexed_block,PMPI_TYPE_CREATE_HINDEXED_BLOCK) (int* count, int* blocklength, int* array_of_displacements, int* oldtype, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_hvector,PMPI_TYPE_CREATE_HVECTOR) (int* count, int* blocklength, int* stride, int* oldtype, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_indexed_block,PMPI_TYPE_CREATE_INDEXED_BLOCK) (int* count, int* blocklength, int* array_of_displacements, int* oldtype, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_keyval,PMPI_TYPE_CREATE_KEYVAL) (int* type_copy_attr_fn, int* type_delete_attr_fn, int* type_keyval, int* extra_state, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_resized,PMPI_TYPE_CREATE_RESIZED) (int* oldtype, int* lb, int* extent, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_struct,PMPI_TYPE_CREATE_STRUCT) (int* count, int* array_of_block_lengths, int* array_of_displacements, int* array_of_types, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_create_subarray,PMPI_TYPE_CREATE_SUBARRAY) (int* ndims, int* size_array, int* subsize_array, int* start_array, int* order, int* oldtype, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_delete_attr,PMPI_TYPE_DELETE_ATTR) (int* type, int* type_keyval, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_dup,PMPI_TYPE_DUP) (int* type, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_free,PMPI_TYPE_FREE) (int* type, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_free_keyval,PMPI_TYPE_FREE_KEYVAL) (int* type_keyval, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_get_attr,PMPI_TYPE_GET_ATTR) (int* type, int* type_keyval, int* attribute_val, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_get_contents,PMPI_TYPE_GET_CONTENTS) (int* mtype, int* max_integers, int* max_addresses, int* max_datatypes, int* array_of_integers, int* array_of_addresses, int* array_of_datatypes, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_get_envelope,PMPI_TYPE_GET_ENVELOPE) (int* type, int* num_integers, int* num_addresses, int* num_datatypes, int* combiner, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_get_extent,PMPI_TYPE_GET_EXTENT) (int* type, int* lb, int* extent, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_get_extent_x,PMPI_TYPE_GET_EXTENT_X) (int* type, int* lb, int* extent, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_get_name,PMPI_TYPE_GET_NAME) (int* type, char* type_name, int* resultlen, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_get_true_extent,PMPI_TYPE_GET_TRUE_EXTENT) (int* datatype, int* true_lb, int* true_extent, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_get_true_extent_x,PMPI_TYPE_GET_TRUE_EXTENT_X) (int* datatype, int* true_lb, int* true_extent, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_indexed,PMPI_TYPE_INDEXED) (int* count, int* array_of_blocklengths, int* array_of_displacements, int* oldtype, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_match_size,PMPI_TYPE_MATCH_SIZE) (int* typeclass, int* size, int* type, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_set_attr,PMPI_TYPE_SET_ATTR) (int* type, int* type_keyval, int* attr_val, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_set_name,PMPI_TYPE_SET_NAME) (int* type, char* type_name, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_size,PMPI_TYPE_SIZE) (int* type, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_size_x,PMPI_TYPE_SIZE_X) (int* type, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_type_vector,PMPI_TYPE_VECTOR) (int* count, int* blocklength, int* stride, int* oldtype, int* newtype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_unpack,PMPI_UNPACK) (int* inbuf, int* insize, int* position, int* outbuf, int* outcount, int* datatype, int* comm, int* ierr);
void FortranCInterface_GLOBAL(pmpi_unpack_external,PMPI_UNPACK_EXTERNAL) (char* datarep, int* inbuf, int* insize, int* position, int* outbuf, int* outcount, int* datatype, int* ierr);
void FortranCInterface_GLOBAL(pmpi_unpublish_name,PMPI_UNPUBLISH_NAME) (char* service_name, int* info, char* port_name, int* ierr);
void FortranCInterface_GLOBAL(pmpi_wait,PMPI_WAIT) (int* request, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_waitall,PMPI_WAITALL) (int* count, int* array_of_requests, int* array_of_statuses, int* ierr);
void FortranCInterface_GLOBAL(pmpi_waitany,PMPI_WAITANY) (int* count, int* array_of_requests, int* index, int* status, int* ierr);
void FortranCInterface_GLOBAL(pmpi_waitsome,PMPI_WAITSOME) (int* incount, int* array_of_requests, int* outcount, int* array_of_indices, int* array_of_statuses, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_allocate,PMPI_WIN_ALLOCATE) (int* size, int* disp_unit, int* info, int* comm, int* baseptr, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_allocate_shared,PMPI_WIN_ALLOCATE_SHARED) (int* size, int* disp_unit, int* info, int* comm, int* baseptr, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_attach,PMPI_WIN_ATTACH) (int* win, int* base, int* size, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_call_errhandler,PMPI_WIN_CALL_ERRHANDLER) (int* win, int* errorcode, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_complete,PMPI_WIN_COMPLETE) (int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_create,PMPI_WIN_CREATE) (int* base, int* size, int* disp_unit, int* info, int* comm, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_create_dynamic,PMPI_WIN_CREATE_DYNAMIC) (int* info, int* comm, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_create_errhandler,PMPI_WIN_CREATE_ERRHANDLER) (int* function, int* errhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_create_keyval,PMPI_WIN_CREATE_KEYVAL) (int* win_copy_attr_fn, int* win_delete_attr_fn, int* win_keyval, int* extra_state, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_delete_attr,PMPI_WIN_DELETE_ATTR) (int* win, int* win_keyval, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_detach,PMPI_WIN_DETACH) (int* win, int* base, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_fence,PMPI_WIN_FENCE) (int* assert, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_flush,PMPI_WIN_FLUSH) (int* rank, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_flush_all,PMPI_WIN_FLUSH_ALL) (int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_flush_local,PMPI_WIN_FLUSH_LOCAL) (int* rank, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_flush_local_all,PMPI_WIN_FLUSH_LOCAL_ALL) (int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_free,PMPI_WIN_FREE) (int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_free_keyval,PMPI_WIN_FREE_KEYVAL) (int* win_keyval, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_get_attr,PMPI_WIN_GET_ATTR) (int* win, int* win_keyval, int* attribute_val, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_get_errhandler,PMPI_WIN_GET_ERRHANDLER) (int* win, int* errhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_get_group,PMPI_WIN_GET_GROUP) (int* win, int* group, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_get_info,PMPI_WIN_GET_INFO) (int* win, int* info_used, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_get_name,PMPI_WIN_GET_NAME) (int* win, char* win_name, int* resultlen, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_lock,PMPI_WIN_LOCK) (int* lock_type, int* rank, int* assert, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_lock_all,PMPI_WIN_LOCK_ALL) (int* assert, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_post,PMPI_WIN_POST) (int* group, int* assert, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_set_attr,PMPI_WIN_SET_ATTR) (int* win, int* win_keyval, int* attribute_val, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_set_errhandler,PMPI_WIN_SET_ERRHANDLER) (int* win, int* errhandler, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_set_info,PMPI_WIN_SET_INFO) (int* win, int* info, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_set_name,PMPI_WIN_SET_NAME) (int* win, char* win_name, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_shared_query,PMPI_WIN_SHARED_QUERY) (int* win, int* rank, int* size, int* disp_unit, int* baseptr, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_start,PMPI_WIN_START) (int* group, int* assert, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_sync,PMPI_WIN_SYNC) (int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_test,PMPI_WIN_TEST) (int* win, int* flag, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_unlock,PMPI_WIN_UNLOCK) (int* rank, int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_unlock_all,PMPI_WIN_UNLOCK_ALL) (int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_win_wait,PMPI_WIN_WAIT) (int* win, int* ierr);
void FortranCInterface_GLOBAL(pmpi_wtick,PMPI_WTICK) (int* ierr);
void FortranCInterface_GLOBAL(pmpi_wtime,PMPI_WTIME) (int* ierr);

void FortranCInterface_GLOBAL(mpi_wait,MPI_WAIT) (int* request, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_wait,PMPI_WAIT)(request, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Wait:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  FortranCInterface_GLOBAL(pmpi_wait,PMPI_WAIT)(request, status, ierr);
  WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  if (lap_backtrace_enabled)
  {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_waitall,MPI_WAITALL) (int* count, int* array_of_requests, int* array_of_statuses, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_waitall,PMPI_WAITALL)(count, array_of_requests, array_of_statuses, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Waitall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  FortranCInterface_GLOBAL(pmpi_waitall,PMPI_WAITALL)(count, array_of_requests, array_of_statuses, ierr);
  WRITE_TRACE("%lli:", (long long int)*count);
  WRITE_TRACE("%lu:", (unsigned long)array_of_requests);
  WRITE_TRACE("%lli:", (long long int)*array_of_statuses);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  if (lap_backtrace_enabled)
  {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n"); 
}
void FortranCInterface_GLOBAL(mpi_waitany,MPI_WAITANY) (int* count, int* array_of_requests, int* index, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_waitany,PMPI_WAITANY)(count, array_of_requests, index, status, ierr);
    return;
  }
  WRITE_TRACE("%s", "MPI_Waitany:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  FortranCInterface_GLOBAL(pmpi_waitany,PMPI_WAITANY)(count, array_of_requests, index, status, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  if (lap_backtrace_enabled)
  {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}
void FortranCInterface_GLOBAL(mpi_waitsome,MPI_WAITSOME) (int* incount, int* array_of_requests, int* outcount, int* array_of_indices, int* array_of_statuses, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_waitsome,PMPI_WAITSOME)(incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr);
    return;
  }
  WRITE_TRACE("%s", "MPI_Waitsome:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  FortranCInterface_GLOBAL(pmpi_waitsome,PMPI_WAITSOME)(incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  if (lap_backtrace_enabled)
  {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");  
}



void FortranCInterface_GLOBAL(mpi_abort,MPI_ABORT) (int* comm, int* errorcode, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_abort,PMPI_ABORT)(comm, errorcode, ierr);
  lap_mpi_initialized = 0;
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Abort:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_abort,PMPI_ABORT)(comm, errorcode, ierr);
  lap_mpi_initialized = 0;
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *errorcode);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_accumulate,MPI_ACCUMULATE) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* op, int* win, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_accumulate,PMPI_ACCUMULATE)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Accumulate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_accumulate,PMPI_ACCUMULATE)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *origin_count);
    WRITE_TRACE("%lli:", (long long int) *origin_datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *target_count);
    WRITE_TRACE("%lli:", (long long int) *target_datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *win);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_add_error_class,MPI_ADD_ERROR_CLASS) (int* errorclass, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_add_error_class,PMPI_ADD_ERROR_CLASS)(errorclass, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Add_error_class:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_add_error_class,PMPI_ADD_ERROR_CLASS)(errorclass, ierr);
    WRITE_TRACE("%lli:", (long long int) *errorclass);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_add_error_code,MPI_ADD_ERROR_CODE) (int* errorclass, int* errorcode, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_add_error_code,PMPI_ADD_ERROR_CODE)(errorclass, errorcode, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Add_error_code:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_add_error_code,PMPI_ADD_ERROR_CODE)(errorclass, errorcode, ierr);
    WRITE_TRACE("%lli:", (long long int) *errorclass);
    WRITE_TRACE("%lli:", (long long int) *errorcode);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_add_error_string,MPI_ADD_ERROR_STRING) (int* errorcode, char* string, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_add_error_string,PMPI_ADD_ERROR_STRING)(errorcode, string, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Add_error_string:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_add_error_string,PMPI_ADD_ERROR_STRING)(errorcode, string, ierr);
    WRITE_TRACE("%lli:", (long long int) *errorcode);
    WRITE_TRACE("%lli:", (long long int) *string);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_allgather,MPI_ALLGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_allgather,PMPI_ALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Allgather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_allgather,PMPI_ALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_allgatherv,MPI_ALLGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_allgatherv,PMPI_ALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Allgatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_allgatherv,PMPI_ALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) displs[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_alloc_mem,MPI_ALLOC_MEM) (int* size, int* info, int* baseptr, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_alloc_mem,PMPI_ALLOC_MEM)(size, info, baseptr, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Alloc_mem:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_alloc_mem,PMPI_ALLOC_MEM)(size, info, baseptr, ierr);
    WRITE_TRACE("%lli:", (long long int) *size);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *baseptr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_allreduce,MPI_ALLREDUCE) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_allreduce,PMPI_ALLREDUCE)(sendbuf, recvbuf, count, datatype, op, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Allreduce:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_allreduce,PMPI_ALLREDUCE)(sendbuf, recvbuf, count, datatype, op, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_alltoall,MPI_ALLTOALL) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_alltoall,PMPI_ALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Alltoall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_alltoall,PMPI_ALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_alltoallv,MPI_ALLTOALLV) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtype, int* recvbuf, int* recvcounts, int* rdispls, int* recvtype, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_alltoallv,PMPI_ALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Alltoallv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_alltoallv,PMPI_ALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) rdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_alltoallw,MPI_ALLTOALLW) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtypes, int* recvbuf, int* recvcounts, int* rdispls, int* recvtypes, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_alltoallw,PMPI_ALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Alltoallw:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_alltoallw,PMPI_ALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sendtypes, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendtypes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) rdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) recvtypes, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvtypes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_attr_delete,MPI_ATTR_DELETE) (int* comm, int* keyval, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_attr_delete,PMPI_ATTR_DELETE)(comm, keyval, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Attr_delete:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_attr_delete,PMPI_ATTR_DELETE)(comm, keyval, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *keyval);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_attr_get,MPI_ATTR_GET) (int* comm, int* keyval, int* attribute_val, int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_attr_get,PMPI_ATTR_GET)(comm, keyval, attribute_val, flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Attr_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_attr_get,PMPI_ATTR_GET)(comm, keyval, attribute_val, flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *keyval);
    WRITE_TRACE("%lli:", (long long int) *attribute_val);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_attr_put,MPI_ATTR_PUT) (int* comm, int* keyval, int* attribute_val, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_attr_put,PMPI_ATTR_PUT)(comm, keyval, attribute_val, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Attr_put:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_attr_put,PMPI_ATTR_PUT)(comm, keyval, attribute_val, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *keyval);
    WRITE_TRACE("%lli:", (long long int) *attribute_val);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_barrier,MPI_BARRIER) (int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_barrier,PMPI_BARRIER)(comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Barrier:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_barrier,PMPI_BARRIER)(comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_bcast,MPI_BCAST) (int* buffer, int* count, int* datatype, int* root, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_bcast,PMPI_BCAST)(buffer, count, datatype, root, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Bcast:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_bcast,PMPI_BCAST)(buffer, count, datatype, root, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *buffer);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_bsend,MPI_BSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_bsend,PMPI_BSEND)(buf, count, datatype, dest, tag, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Bsend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_bsend,PMPI_BSEND)(buf, count, datatype, dest, tag, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_bsend_init,MPI_BSEND_INIT) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_bsend_init,PMPI_BSEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Bsend_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_bsend_init,PMPI_BSEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_buffer_attach,MPI_BUFFER_ATTACH) (int* buffer, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_buffer_attach,PMPI_BUFFER_ATTACH)(buffer, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Buffer_attach:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_buffer_attach,PMPI_BUFFER_ATTACH)(buffer, size, ierr);
    WRITE_TRACE("%lli:", (long long int) *buffer);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_buffer_detach,MPI_BUFFER_DETACH) (int* buffer, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_buffer_detach,PMPI_BUFFER_DETACH)(buffer, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Buffer_detach:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_buffer_detach,PMPI_BUFFER_DETACH)(buffer, size, ierr);
    WRITE_TRACE("%lli:", (long long int) *buffer);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_cancel,MPI_CANCEL) (int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_cancel,PMPI_CANCEL)(request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Cancel:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_cancel,PMPI_CANCEL)(request, ierr);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_cart_coords,MPI_CART_COORDS) (int* comm, int* rank, int* maxdims, int* coords, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_cart_coords,PMPI_CART_COORDS)(comm, rank, maxdims, coords, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_coords:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_cart_coords,PMPI_CART_COORDS)(comm, rank, maxdims, coords, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *rank);
    WRITE_TRACE("%lli:", (long long int) *maxdims);
  WRITE_TRACE("%p,%i[", (void*) coords, (int) (*maxdims));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxdims); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) coords[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_cart_create,MPI_CART_CREATE) (int* old_comm, int* ndims, int* dims, int* periods, int* reorder, int* comm_cart, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_cart_create,PMPI_CART_CREATE)(old_comm, ndims, dims, periods, reorder, comm_cart, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_cart_create,PMPI_CART_CREATE)(old_comm, ndims, dims, periods, reorder, comm_cart, ierr);
    WRITE_TRACE("%lli:", (long long int) *old_comm);
    WRITE_TRACE("%lli:", (long long int) *ndims);
  WRITE_TRACE("%p,%i[", (void*) dims, (int) (*ndims));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*ndims); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) dims[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) periods, (int) (*ndims));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*ndims); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) periods[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *reorder);
    WRITE_TRACE("%lli:", (long long int) *comm_cart);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_cart_get,MPI_CART_GET) (int* comm, int* maxdims, int* dims, int* periods, int* coords, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_cart_get,PMPI_CART_GET)(comm, maxdims, dims, periods, coords, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_cart_get,PMPI_CART_GET)(comm, maxdims, dims, periods, coords, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *maxdims);
  WRITE_TRACE("%p,%i[", (void*) dims, (int) (*maxdims));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxdims); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) dims[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) periods, (int) (*maxdims));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxdims); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) periods[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) coords, (int) (*maxdims));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxdims); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) coords[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_cart_map,MPI_CART_MAP) (int* comm, int* ndims, int* dims, int* periods, int* newrank, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_cart_map,PMPI_CART_MAP)(comm, ndims, dims, periods, newrank, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_map:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_cart_map,PMPI_CART_MAP)(comm, ndims, dims, periods, newrank, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *ndims);
  WRITE_TRACE("%p,%i[", (void*) dims, (int) (*ndims));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*ndims); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) dims[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) periods, (int) (*ndims));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*ndims); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) periods[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *newrank);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_cart_rank,MPI_CART_RANK) (int* comm, int* coords, int* rank, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_cart_rank,PMPI_CART_RANK)(comm, coords, rank, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_rank:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_cart_rank,PMPI_CART_RANK)(comm, coords, rank, ierr);
int ndims; PMPI_Cartdim_get( MPI_Comm_f2c((*comm)),  &ndims);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%p,%i[", (void*) coords, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) coords[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *rank);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_cart_shift,MPI_CART_SHIFT) (int* comm, int* direction, int* disp, int* rank_source, int* rank_dest, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_cart_shift,PMPI_CART_SHIFT)(comm, direction, disp, rank_source, rank_dest, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_shift:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_cart_shift,PMPI_CART_SHIFT)(comm, direction, disp, rank_source, rank_dest, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *direction);
    WRITE_TRACE("%lli:", (long long int) *disp);
    WRITE_TRACE("%lli:", (long long int) *rank_source);
    WRITE_TRACE("%lli:", (long long int) *rank_dest);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_cart_sub,MPI_CART_SUB) (int* comm, int* remain_dims, int* new_comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_cart_sub,PMPI_CART_SUB)(comm, remain_dims, new_comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Cart_sub:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_cart_sub,PMPI_CART_SUB)(comm, remain_dims, new_comm, ierr);
int ndims; PMPI_Cartdim_get( MPI_Comm_f2c((*comm)),  &ndims);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%p,%i[", (void*) remain_dims, (int) ndims);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ndims; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) remain_dims[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *new_comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_cartdim_get,MPI_CARTDIM_GET) (int* comm, int* ndims, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_cartdim_get,PMPI_CARTDIM_GET)(comm, ndims, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Cartdim_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_cartdim_get,PMPI_CARTDIM_GET)(comm, ndims, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *ndims);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_close_port,MPI_CLOSE_PORT) (char* port_name, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_close_port,PMPI_CLOSE_PORT)(port_name, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Close_port:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_close_port,PMPI_CLOSE_PORT)(port_name, ierr);
    WRITE_TRACE("%lli:", (long long int) *port_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_accept,MPI_COMM_ACCEPT) (char* port_name, int* info, int* root, int* comm, int* newcomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_accept,PMPI_COMM_ACCEPT)(port_name, info, root, comm, newcomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_accept:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_accept,PMPI_COMM_ACCEPT)(port_name, info, root, comm, newcomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *port_name);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_call_errhandler,MPI_COMM_CALL_ERRHANDLER) (int* comm, int* errorcode, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_call_errhandler,PMPI_COMM_CALL_ERRHANDLER)(comm, errorcode, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_call_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_call_errhandler,PMPI_COMM_CALL_ERRHANDLER)(comm, errorcode, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *errorcode);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_compare,MPI_COMM_COMPARE) (int* comm1, int* comm2, int* result, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_compare,PMPI_COMM_COMPARE)(comm1, comm2, result, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_compare:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_compare,PMPI_COMM_COMPARE)(comm1, comm2, result, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm1);
    WRITE_TRACE("%lli:", (long long int) *comm2);
    WRITE_TRACE("%lli:", (long long int) *result);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_connect,MPI_COMM_CONNECT) (char* port_name, int* info, int* root, int* comm, int* newcomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_connect,PMPI_COMM_CONNECT)(port_name, info, root, comm, newcomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_connect:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_connect,PMPI_COMM_CONNECT)(port_name, info, root, comm, newcomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *port_name);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_create,MPI_COMM_CREATE) (int* comm, int* group, int* newcomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_create,PMPI_COMM_CREATE)(comm, group, newcomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_create,PMPI_COMM_CREATE)(comm, group, newcomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *group);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_create_errhandler,MPI_COMM_CREATE_ERRHANDLER) (int* function, int* errhandler, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_create_errhandler,PMPI_COMM_CREATE_ERRHANDLER)(function, errhandler, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_create_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_create_errhandler,PMPI_COMM_CREATE_ERRHANDLER)(function, errhandler, ierr);
    WRITE_TRACE("%lli:", (long long int) *function);
    WRITE_TRACE("%lli:", (long long int) *errhandler);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_create_group,MPI_COMM_CREATE_GROUP) (int* comm, int* group, int* tag, int* newcomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_create_group,PMPI_COMM_CREATE_GROUP)(comm, group, tag, newcomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_create_group:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_create_group,PMPI_COMM_CREATE_GROUP)(comm, group, tag, newcomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *group);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_create_keyval,MPI_COMM_CREATE_KEYVAL) (int* comm_copy_attr_fn, int* comm_delete_attr_fn, int* comm_keyval, int* extra_state, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_create_keyval,PMPI_COMM_CREATE_KEYVAL)(comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_create_keyval:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_create_keyval,PMPI_COMM_CREATE_KEYVAL)(comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm_copy_attr_fn);
    WRITE_TRACE("%lli:", (long long int) *comm_delete_attr_fn);
    WRITE_TRACE("%lli:", (long long int) *comm_keyval);
    WRITE_TRACE("%lli:", (long long int) *extra_state);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_delete_attr,MPI_COMM_DELETE_ATTR) (int* comm, int* comm_keyval, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_delete_attr,PMPI_COMM_DELETE_ATTR)(comm, comm_keyval, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_delete_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *comm_keyval);
 FortranCInterface_GLOBAL(pmpi_comm_delete_attr,PMPI_COMM_DELETE_ATTR)(comm, comm_keyval, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_disconnect,MPI_COMM_DISCONNECT) (int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_disconnect,PMPI_COMM_DISCONNECT)(comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_disconnect:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_disconnect,PMPI_COMM_DISCONNECT)(comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_dup,MPI_COMM_DUP) (int* comm, int* newcomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_dup,PMPI_COMM_DUP)(comm, newcomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_dup:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_dup,PMPI_COMM_DUP)(comm, newcomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_dup_with_info,MPI_COMM_DUP_WITH_INFO) (int* comm, int* info, int* newcomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_dup_with_info,PMPI_COMM_DUP_WITH_INFO)(comm, info, newcomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_dup_with_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_dup_with_info,PMPI_COMM_DUP_WITH_INFO)(comm, info, newcomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_free,MPI_COMM_FREE) (int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_free,PMPI_COMM_FREE)(comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *comm);
 FortranCInterface_GLOBAL(pmpi_comm_free,PMPI_COMM_FREE)(comm, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_free_keyval,MPI_COMM_FREE_KEYVAL) (int* comm_keyval, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_free_keyval,PMPI_COMM_FREE_KEYVAL)(comm_keyval, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_free_keyval:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_free_keyval,PMPI_COMM_FREE_KEYVAL)(comm_keyval, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm_keyval);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_get_attr,MPI_COMM_GET_ATTR) (int* comm, int* comm_keyval, int* attribute_val, int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_get_attr,PMPI_COMM_GET_ATTR)(comm, comm_keyval, attribute_val, flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_get_attr,PMPI_COMM_GET_ATTR)(comm, comm_keyval, attribute_val, flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *comm_keyval);
    WRITE_TRACE("%lli:", (long long int) *attribute_val);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_get_errhandler,MPI_COMM_GET_ERRHANDLER) (int* comm, int* erhandler, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_get_errhandler,PMPI_COMM_GET_ERRHANDLER)(comm, erhandler, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_get_errhandler,PMPI_COMM_GET_ERRHANDLER)(comm, erhandler, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *erhandler);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_get_info,MPI_COMM_GET_INFO) (int* comm, int* info_used, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_get_info,PMPI_COMM_GET_INFO)(comm, info_used, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_get_info,PMPI_COMM_GET_INFO)(comm, info_used, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *info_used);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_get_name,MPI_COMM_GET_NAME) (int* comm, char* comm_name, int* resultlen, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_get_name,PMPI_COMM_GET_NAME)(comm, comm_name, resultlen, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_get_name,PMPI_COMM_GET_NAME)(comm, comm_name, resultlen, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *comm_name);
    WRITE_TRACE("%lli:", (long long int) *resultlen);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_get_parent,MPI_COMM_GET_PARENT) (int* parent, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_get_parent,PMPI_COMM_GET_PARENT)(parent, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_get_parent:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_get_parent,PMPI_COMM_GET_PARENT)(parent, ierr);
    WRITE_TRACE("%lli:", (long long int) *parent);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_group,MPI_COMM_GROUP) (int* comm, int* group, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_group,PMPI_COMM_GROUP)(comm, group, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_group:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_group,PMPI_COMM_GROUP)(comm, group, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *group);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_idup,MPI_COMM_IDUP) (int* comm, int* newcomm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_idup,PMPI_COMM_IDUP)(comm, newcomm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_idup:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_idup,PMPI_COMM_IDUP)(comm, newcomm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_join,MPI_COMM_JOIN) (int* fd, int* intercomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_join,PMPI_COMM_JOIN)(fd, intercomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_join:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_join,PMPI_COMM_JOIN)(fd, intercomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *fd);
    WRITE_TRACE("%lli:", (long long int) *intercomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_rank,MPI_COMM_RANK) (int* comm, int* rank, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_rank,PMPI_COMM_RANK)(comm, rank, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_rank:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_rank,PMPI_COMM_RANK)(comm, rank, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *rank);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_remote_group,MPI_COMM_REMOTE_GROUP) (int* comm, int* group, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_remote_group,PMPI_COMM_REMOTE_GROUP)(comm, group, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_remote_group:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_remote_group,PMPI_COMM_REMOTE_GROUP)(comm, group, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *group);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_remote_size,MPI_COMM_REMOTE_SIZE) (int* comm, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_remote_size,PMPI_COMM_REMOTE_SIZE)(comm, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_remote_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_remote_size,PMPI_COMM_REMOTE_SIZE)(comm, size, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_set_attr,MPI_COMM_SET_ATTR) (int* comm, int* comm_keyval, int* attribute_val, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_set_attr,PMPI_COMM_SET_ATTR)(comm, comm_keyval, attribute_val, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_set_attr:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_set_attr,PMPI_COMM_SET_ATTR)(comm, comm_keyval, attribute_val, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *comm_keyval);
    WRITE_TRACE("%lli:", (long long int) *attribute_val);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_set_errhandler,MPI_COMM_SET_ERRHANDLER) (int* comm, int* errhandler, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_set_errhandler,PMPI_COMM_SET_ERRHANDLER)(comm, errhandler, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_set_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_set_errhandler,PMPI_COMM_SET_ERRHANDLER)(comm, errhandler, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *errhandler);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_set_info,MPI_COMM_SET_INFO) (int* comm, int* info, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_set_info,PMPI_COMM_SET_INFO)(comm, info, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_set_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_set_info,PMPI_COMM_SET_INFO)(comm, info, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *info);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_set_name,MPI_COMM_SET_NAME) (int* comm, char* comm_name, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_set_name,PMPI_COMM_SET_NAME)(comm, comm_name, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_set_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_set_name,PMPI_COMM_SET_NAME)(comm, comm_name, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *comm_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_size,MPI_COMM_SIZE) (int* comm, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_size,PMPI_COMM_SIZE)(comm, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_size,PMPI_COMM_SIZE)(comm, size, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_spawn,MPI_COMM_SPAWN) (char* command, char* argv, int* maxprocs, int* info, int* root, int* comm, int* intercomm, int* array_of_errcodes, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_spawn,PMPI_COMM_SPAWN)(command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_spawn:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_spawn,PMPI_COMM_SPAWN)(command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes, ierr);
    WRITE_TRACE("%lli:", (long long int) *command);
  WRITE_TRACE("%p,%i[", (void*) argv, (int) 0);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<0; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) argv[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *maxprocs);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *intercomm);
  WRITE_TRACE("%p,%i[", (void*) array_of_errcodes, (int) (*maxprocs));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxprocs); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) array_of_errcodes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_spawn_multiple,MPI_COMM_SPAWN_MULTIPLE) (int* count, char* array_of_commands, char* array_of_argv, int* array_of_maxprocs, int* array_of_info, int* root, int* comm, int* intercomm, int* array_of_errcodes, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_spawn_multiple,PMPI_COMM_SPAWN_MULTIPLE)(count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_spawn_multiple:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_spawn_multiple,PMPI_COMM_SPAWN_MULTIPLE)(count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes, ierr);
    WRITE_TRACE("%lli:", (long long int) *count);
  WRITE_TRACE("%p,%i[", (void*) array_of_commands, (int) (*count));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*count); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) array_of_commands[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_argv, (int) (*count));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*count); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) array_of_argv[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_maxprocs, (int) (*count));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*count); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) array_of_maxprocs[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) array_of_info, (int) (*count));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*count); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) array_of_info[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *intercomm);
  WRITE_TRACE("%p,%i[", (void*) array_of_errcodes, (int) (*count));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*count); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) array_of_errcodes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_split,MPI_COMM_SPLIT) (int* comm, int* color, int* key, int* newcomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_split,PMPI_COMM_SPLIT)(comm, color, key, newcomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_split:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_split,PMPI_COMM_SPLIT)(comm, color, key, newcomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *color);
    WRITE_TRACE("%lli:", (long long int) *key);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_split_type,MPI_COMM_SPLIT_TYPE) (int* comm, int* split_type, int* key, int* info, int* newcomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_split_type,PMPI_COMM_SPLIT_TYPE)(comm, split_type, key, info, newcomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_split_type:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_split_type,PMPI_COMM_SPLIT_TYPE)(comm, split_type, key, info, newcomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *split_type);
    WRITE_TRACE("%lli:", (long long int) *key);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_comm_test_inter,MPI_COMM_TEST_INTER) (int* comm, int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_comm_test_inter,PMPI_COMM_TEST_INTER)(comm, flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Comm_test_inter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_comm_test_inter,PMPI_COMM_TEST_INTER)(comm, flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_compare_and_swap,MPI_COMPARE_AND_SWAP) (int* origin_addr, int* compare_addr, int* result_addr, int* datatype, int* target_rank, int* target_disp, int* win, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_compare_and_swap,PMPI_COMPARE_AND_SWAP)(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp, win, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Compare_and_swap:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_compare_and_swap,PMPI_COMPARE_AND_SWAP)(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp, win, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *compare_addr);
    WRITE_TRACE("%lli:", (long long int) *result_addr);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *win);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_dims_create,MPI_DIMS_CREATE) (int* nnodes, int* ndims, int* dims, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_dims_create,PMPI_DIMS_CREATE)(nnodes, ndims, dims, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Dims_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_dims_create,PMPI_DIMS_CREATE)(nnodes, ndims, dims, ierr);
    WRITE_TRACE("%lli:", (long long int) *nnodes);
    WRITE_TRACE("%lli:", (long long int) *ndims);
  WRITE_TRACE("%p,%i[", (void*) dims, (int) (*ndims));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*ndims); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) dims[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_dist_graph_create,MPI_DIST_GRAPH_CREATE) (int* comm_old, int* n, int* nodes, int* degrees, int* targets, int* weights, int* info, int* reorder, int* newcomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_dist_graph_create,PMPI_DIST_GRAPH_CREATE)(comm_old, n, nodes, degrees, targets, weights, info, reorder, newcomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Dist_graph_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_dist_graph_create,PMPI_DIST_GRAPH_CREATE)(comm_old, n, nodes, degrees, targets, weights, info, reorder, newcomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm_old);
    WRITE_TRACE("%lli:", (long long int) *n);
  WRITE_TRACE("%p,%i[", (void*) nodes, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) nodes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) degrees, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) degrees[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) targets, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) targets[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) weights, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) weights[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *reorder);
    WRITE_TRACE("%lli:", (long long int) *newcomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_dist_graph_create_adjacent,MPI_DIST_GRAPH_CREATE_ADJACENT) (int* comm_old, int* indegree, int* sources, int* sourceweights, int* outdegree, int* destinations, int* destweights, int* info, int* reorder, int* comm_dist_graph, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_dist_graph_create_adjacent,PMPI_DIST_GRAPH_CREATE_ADJACENT)(comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Dist_graph_create_adjacent:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_dist_graph_create_adjacent,PMPI_DIST_GRAPH_CREATE_ADJACENT)(comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm_old);
    WRITE_TRACE("%lli:", (long long int) *indegree);
  WRITE_TRACE("%p,%i[", (void*) sources, (int) (*indegree));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*indegree); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sources[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sourceweights, (int) (*indegree));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*indegree); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sourceweights[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *outdegree);
  WRITE_TRACE("%p,%i[", (void*) destinations, (int) (*outdegree));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*outdegree); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) destinations[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) destweights, (int) (*outdegree));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*outdegree); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) destweights[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *reorder);
    WRITE_TRACE("%lli:", (long long int) *comm_dist_graph);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_dist_graph_neighbors,MPI_DIST_GRAPH_NEIGHBORS) (int* comm, int* maxindegree, int* sources, int* sourceweights, int* maxoutdegree, int* destinations, int* destweights, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_dist_graph_neighbors,PMPI_DIST_GRAPH_NEIGHBORS)(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations, destweights, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Dist_graph_neighbors:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_dist_graph_neighbors,PMPI_DIST_GRAPH_NEIGHBORS)(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations, destweights, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *maxindegree);
  WRITE_TRACE("%p,%i[", (void*) sources, (int) (*maxindegree));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxindegree); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sources[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sourceweights, (int) (*maxindegree));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxindegree); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sourceweights[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *maxoutdegree);
  WRITE_TRACE("%p,%i[", (void*) destinations, (int) (*maxoutdegree));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxoutdegree); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) destinations[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) destweights, (int) (*maxoutdegree));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxoutdegree); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) destweights[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_dist_graph_neighbors_count,MPI_DIST_GRAPH_NEIGHBORS_COUNT) (int* comm, int* inneighbors, int* outneighbors, int* weighted, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_dist_graph_neighbors_count,PMPI_DIST_GRAPH_NEIGHBORS_COUNT)(comm, inneighbors, outneighbors, weighted, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Dist_graph_neighbors_count:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_dist_graph_neighbors_count,PMPI_DIST_GRAPH_NEIGHBORS_COUNT)(comm, inneighbors, outneighbors, weighted, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *inneighbors);
    WRITE_TRACE("%lli:", (long long int) *outneighbors);
    WRITE_TRACE("%lli:", (long long int) *weighted);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_errhandler_free,MPI_ERRHANDLER_FREE) (int* errhandler, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_errhandler_free,PMPI_ERRHANDLER_FREE)(errhandler, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Errhandler_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *errhandler);
 FortranCInterface_GLOBAL(pmpi_errhandler_free,PMPI_ERRHANDLER_FREE)(errhandler, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_error_class,MPI_ERROR_CLASS) (int* errorcode, int* errorclass, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_error_class,PMPI_ERROR_CLASS)(errorcode, errorclass, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Error_class:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_error_class,PMPI_ERROR_CLASS)(errorcode, errorclass, ierr);
    WRITE_TRACE("%lli:", (long long int) *errorcode);
    WRITE_TRACE("%lli:", (long long int) *errorclass);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_error_string,MPI_ERROR_STRING) (int* errorcode, char* string, int* resultlen, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_error_string,PMPI_ERROR_STRING)(errorcode, string, resultlen, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Error_string:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_error_string,PMPI_ERROR_STRING)(errorcode, string, resultlen, ierr);
    WRITE_TRACE("%lli:", (long long int) *errorcode);
    WRITE_TRACE("%lli:", (long long int) *string);
    WRITE_TRACE("%lli:", (long long int) *resultlen);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_exscan,MPI_EXSCAN) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_exscan,PMPI_EXSCAN)(sendbuf, recvbuf, count, datatype, op, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Exscan:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_exscan,PMPI_EXSCAN)(sendbuf, recvbuf, count, datatype, op, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_fetch_and_op,MPI_FETCH_AND_OP) (int* origin_addr, int* result_addr, int* datatype, int* target_rank, int* target_disp, int* op, int* win, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_fetch_and_op,PMPI_FETCH_AND_OP)(origin_addr, result_addr, datatype, target_rank, target_disp, op, win, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Fetch_and_op:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_fetch_and_op,PMPI_FETCH_AND_OP)(origin_addr, result_addr, datatype, target_rank, target_disp, op, win, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *result_addr);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *win);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_call_errhandler,MPI_FILE_CALL_ERRHANDLER) (int* fh, int* errorcode, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_call_errhandler,PMPI_FILE_CALL_ERRHANDLER)(fh, errorcode, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_call_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_call_errhandler,PMPI_FILE_CALL_ERRHANDLER)(fh, errorcode, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *errorcode);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_close,MPI_FILE_CLOSE) (int* fh, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_close,PMPI_FILE_CLOSE)(fh, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_close:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_close,PMPI_FILE_CLOSE)(fh, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_create_errhandler,MPI_FILE_CREATE_ERRHANDLER) (int* function, int* errhandler, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_create_errhandler,PMPI_FILE_CREATE_ERRHANDLER)(function, errhandler, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_create_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_create_errhandler,PMPI_FILE_CREATE_ERRHANDLER)(function, errhandler, ierr);
    WRITE_TRACE("%lli:", (long long int) *function);
    WRITE_TRACE("%lli:", (long long int) *errhandler);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_delete,MPI_FILE_DELETE) (char* filename, int* info, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_delete,PMPI_FILE_DELETE)(filename, info, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_delete:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_delete,PMPI_FILE_DELETE)(filename, info, ierr);
    WRITE_TRACE("%lli:", (long long int) *filename);
    WRITE_TRACE("%lli:", (long long int) *info);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_amode,MPI_FILE_GET_AMODE) (int* fh, int* amode, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_amode,PMPI_FILE_GET_AMODE)(fh, amode, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_amode:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_amode,PMPI_FILE_GET_AMODE)(fh, amode, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *amode);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_atomicity,MPI_FILE_GET_ATOMICITY) (int* fh, int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_atomicity,PMPI_FILE_GET_ATOMICITY)(fh, flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_atomicity:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_atomicity,PMPI_FILE_GET_ATOMICITY)(fh, flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_byte_offset,MPI_FILE_GET_BYTE_OFFSET) (int* fh, int* offset, int* disp, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_byte_offset,PMPI_FILE_GET_BYTE_OFFSET)(fh, offset, disp, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_byte_offset:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_byte_offset,PMPI_FILE_GET_BYTE_OFFSET)(fh, offset, disp, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *disp);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_errhandler,MPI_FILE_GET_ERRHANDLER) (int* file, int* errhandler, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_errhandler,PMPI_FILE_GET_ERRHANDLER)(file, errhandler, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_errhandler,PMPI_FILE_GET_ERRHANDLER)(file, errhandler, ierr);
    WRITE_TRACE("%lli:", (long long int) *file);
    WRITE_TRACE("%lli:", (long long int) *errhandler);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_group,MPI_FILE_GET_GROUP) (int* fh, int* group, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_group,PMPI_FILE_GET_GROUP)(fh, group, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_group:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_group,PMPI_FILE_GET_GROUP)(fh, group, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *group);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_info,MPI_FILE_GET_INFO) (int* fh, int* info_used, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_info,PMPI_FILE_GET_INFO)(fh, info_used, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_info,PMPI_FILE_GET_INFO)(fh, info_used, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *info_used);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_position,MPI_FILE_GET_POSITION) (int* fh, int* offset, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_position,PMPI_FILE_GET_POSITION)(fh, offset, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_position:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_position,PMPI_FILE_GET_POSITION)(fh, offset, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_position_shared,MPI_FILE_GET_POSITION_SHARED) (int* fh, int* offset, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_position_shared,PMPI_FILE_GET_POSITION_SHARED)(fh, offset, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_position_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_position_shared,PMPI_FILE_GET_POSITION_SHARED)(fh, offset, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_size,MPI_FILE_GET_SIZE) (int* fh, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_size,PMPI_FILE_GET_SIZE)(fh, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_size,PMPI_FILE_GET_SIZE)(fh, size, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_type_extent,MPI_FILE_GET_TYPE_EXTENT) (int* fh, int* datatype, int* extent, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_type_extent,PMPI_FILE_GET_TYPE_EXTENT)(fh, datatype, extent, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_type_extent:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_type_extent,PMPI_FILE_GET_TYPE_EXTENT)(fh, datatype, extent, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *extent);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_get_view,MPI_FILE_GET_VIEW) (int* fh, int* disp, int* etype, int* filetype, char* datarep, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_get_view,PMPI_FILE_GET_VIEW)(fh, disp, etype, filetype, datarep, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_get_view:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_get_view,PMPI_FILE_GET_VIEW)(fh, disp, etype, filetype, datarep, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *disp);
    WRITE_TRACE("%lli:", (long long int) *etype);
    WRITE_TRACE("%lli:", (long long int) *filetype);
    WRITE_TRACE("%lli:", (long long int) *datarep);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iread,MPI_FILE_IREAD) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iread,PMPI_FILE_IREAD)(fh, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iread,PMPI_FILE_IREAD)(fh, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iread_all,MPI_FILE_IREAD_ALL) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iread_all,PMPI_FILE_IREAD_ALL)(fh, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iread_all,PMPI_FILE_IREAD_ALL)(fh, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iread_at,MPI_FILE_IREAD_AT) (int* fh, int* offset, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iread_at,PMPI_FILE_IREAD_AT)(fh, offset, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread_at:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iread_at,PMPI_FILE_IREAD_AT)(fh, offset, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iread_at_all,MPI_FILE_IREAD_AT_ALL) (int* fh, int* offset, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iread_at_all,PMPI_FILE_IREAD_AT_ALL)(fh, offset, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread_at_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iread_at_all,PMPI_FILE_IREAD_AT_ALL)(fh, offset, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iread_shared,MPI_FILE_IREAD_SHARED) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iread_shared,PMPI_FILE_IREAD_SHARED)(fh, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iread_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iread_shared,PMPI_FILE_IREAD_SHARED)(fh, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iwrite,MPI_FILE_IWRITE) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iwrite,PMPI_FILE_IWRITE)(fh, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iwrite,PMPI_FILE_IWRITE)(fh, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iwrite_all,MPI_FILE_IWRITE_ALL) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iwrite_all,PMPI_FILE_IWRITE_ALL)(fh, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iwrite_all,PMPI_FILE_IWRITE_ALL)(fh, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iwrite_at,MPI_FILE_IWRITE_AT) (int* fh, int* offset, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iwrite_at,PMPI_FILE_IWRITE_AT)(fh, offset, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite_at:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iwrite_at,PMPI_FILE_IWRITE_AT)(fh, offset, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iwrite_at_all,MPI_FILE_IWRITE_AT_ALL) (int* fh, int* offset, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iwrite_at_all,PMPI_FILE_IWRITE_AT_ALL)(fh, offset, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite_at_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iwrite_at_all,PMPI_FILE_IWRITE_AT_ALL)(fh, offset, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_iwrite_shared,MPI_FILE_IWRITE_SHARED) (int* fh, int* buf, int* count, int* datatype, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_iwrite_shared,PMPI_FILE_IWRITE_SHARED)(fh, buf, count, datatype, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_iwrite_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_iwrite_shared,PMPI_FILE_IWRITE_SHARED)(fh, buf, count, datatype, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_open,MPI_FILE_OPEN) (int* comm, char* filename, int* amode, int* info, int* fh, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_open,PMPI_FILE_OPEN)(comm, filename, amode, info, fh, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_open:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_open,PMPI_FILE_OPEN)(comm, filename, amode, info, fh, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *filename);
    WRITE_TRACE("%lli:", (long long int) *amode);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *fh);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_preallocate,MPI_FILE_PREALLOCATE) (int* fh, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_preallocate,PMPI_FILE_PREALLOCATE)(fh, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_preallocate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_preallocate,PMPI_FILE_PREALLOCATE)(fh, size, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read,MPI_FILE_READ) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read,PMPI_FILE_READ)(fh, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read,PMPI_FILE_READ)(fh, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_all,MPI_FILE_READ_ALL) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_all,PMPI_FILE_READ_ALL)(fh, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_all,PMPI_FILE_READ_ALL)(fh, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_all_begin,MPI_FILE_READ_ALL_BEGIN) (int* fh, int* buf, int* count, int* datatype, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_all_begin,PMPI_FILE_READ_ALL_BEGIN)(fh, buf, count, datatype, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_all_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_all_begin,PMPI_FILE_READ_ALL_BEGIN)(fh, buf, count, datatype, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_all_end,MPI_FILE_READ_ALL_END) (int* fh, int* buf, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_all_end,PMPI_FILE_READ_ALL_END)(fh, buf, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_all_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_all_end,PMPI_FILE_READ_ALL_END)(fh, buf, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_at,MPI_FILE_READ_AT) (int* fh, int* offset, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_at,PMPI_FILE_READ_AT)(fh, offset, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_at:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_at,PMPI_FILE_READ_AT)(fh, offset, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_at_all,MPI_FILE_READ_AT_ALL) (int* fh, int* offset, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_at_all,PMPI_FILE_READ_AT_ALL)(fh, offset, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_at_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_at_all,PMPI_FILE_READ_AT_ALL)(fh, offset, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_at_all_begin,MPI_FILE_READ_AT_ALL_BEGIN) (int* fh, int* offset, int* buf, int* count, int* datatype, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_at_all_begin,PMPI_FILE_READ_AT_ALL_BEGIN)(fh, offset, buf, count, datatype, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_at_all_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_at_all_begin,PMPI_FILE_READ_AT_ALL_BEGIN)(fh, offset, buf, count, datatype, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_at_all_end,MPI_FILE_READ_AT_ALL_END) (int* fh, int* buf, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_at_all_end,PMPI_FILE_READ_AT_ALL_END)(fh, buf, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_at_all_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_at_all_end,PMPI_FILE_READ_AT_ALL_END)(fh, buf, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_ordered,MPI_FILE_READ_ORDERED) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_ordered,PMPI_FILE_READ_ORDERED)(fh, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_ordered:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_ordered,PMPI_FILE_READ_ORDERED)(fh, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_ordered_begin,MPI_FILE_READ_ORDERED_BEGIN) (int* fh, int* buf, int* count, int* datatype, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_ordered_begin,PMPI_FILE_READ_ORDERED_BEGIN)(fh, buf, count, datatype, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_ordered_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_ordered_begin,PMPI_FILE_READ_ORDERED_BEGIN)(fh, buf, count, datatype, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_ordered_end,MPI_FILE_READ_ORDERED_END) (int* fh, int* buf, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_ordered_end,PMPI_FILE_READ_ORDERED_END)(fh, buf, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_ordered_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_ordered_end,PMPI_FILE_READ_ORDERED_END)(fh, buf, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_read_shared,MPI_FILE_READ_SHARED) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_read_shared,PMPI_FILE_READ_SHARED)(fh, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_read_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_read_shared,PMPI_FILE_READ_SHARED)(fh, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_seek,MPI_FILE_SEEK) (int* fh, int* offset, int* whence, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_seek,PMPI_FILE_SEEK)(fh, offset, whence, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_seek:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_seek,PMPI_FILE_SEEK)(fh, offset, whence, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *whence);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_seek_shared,MPI_FILE_SEEK_SHARED) (int* fh, int* offset, int* whence, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_seek_shared,PMPI_FILE_SEEK_SHARED)(fh, offset, whence, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_seek_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_seek_shared,PMPI_FILE_SEEK_SHARED)(fh, offset, whence, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *whence);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_set_atomicity,MPI_FILE_SET_ATOMICITY) (int* fh, int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_set_atomicity,PMPI_FILE_SET_ATOMICITY)(fh, flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_atomicity:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_set_atomicity,PMPI_FILE_SET_ATOMICITY)(fh, flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_set_errhandler,MPI_FILE_SET_ERRHANDLER) (int* file, int* errhandler, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_set_errhandler,PMPI_FILE_SET_ERRHANDLER)(file, errhandler, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_errhandler:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_set_errhandler,PMPI_FILE_SET_ERRHANDLER)(file, errhandler, ierr);
    WRITE_TRACE("%lli:", (long long int) *file);
    WRITE_TRACE("%lli:", (long long int) *errhandler);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_set_info,MPI_FILE_SET_INFO) (int* fh, int* info, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_set_info,PMPI_FILE_SET_INFO)(fh, info, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_info:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_set_info,PMPI_FILE_SET_INFO)(fh, info, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *info);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_set_size,MPI_FILE_SET_SIZE) (int* fh, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_set_size,PMPI_FILE_SET_SIZE)(fh, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_set_size,PMPI_FILE_SET_SIZE)(fh, size, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_set_view,MPI_FILE_SET_VIEW) (int* fh, int* disp, int* etype, int* filetype, char* datarep, int* info, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_set_view,PMPI_FILE_SET_VIEW)(fh, disp, etype, filetype, datarep, info, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_set_view:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_set_view,PMPI_FILE_SET_VIEW)(fh, disp, etype, filetype, datarep, info, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *disp);
    WRITE_TRACE("%lli:", (long long int) *etype);
    WRITE_TRACE("%lli:", (long long int) *filetype);
    WRITE_TRACE("%lli:", (long long int) *datarep);
    WRITE_TRACE("%lli:", (long long int) *info);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_sync,MPI_FILE_SYNC) (int* fh, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_sync,PMPI_FILE_SYNC)(fh, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_sync:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_sync,PMPI_FILE_SYNC)(fh, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write,MPI_FILE_WRITE) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write,PMPI_FILE_WRITE)(fh, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write,PMPI_FILE_WRITE)(fh, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_all,MPI_FILE_WRITE_ALL) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_all,PMPI_FILE_WRITE_ALL)(fh, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_all,PMPI_FILE_WRITE_ALL)(fh, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_all_begin,MPI_FILE_WRITE_ALL_BEGIN) (int* fh, int* buf, int* count, int* datatype, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_all_begin,PMPI_FILE_WRITE_ALL_BEGIN)(fh, buf, count, datatype, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_all_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_all_begin,PMPI_FILE_WRITE_ALL_BEGIN)(fh, buf, count, datatype, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_all_end,MPI_FILE_WRITE_ALL_END) (int* fh, int* buf, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_all_end,PMPI_FILE_WRITE_ALL_END)(fh, buf, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_all_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_all_end,PMPI_FILE_WRITE_ALL_END)(fh, buf, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_at,MPI_FILE_WRITE_AT) (int* fh, int* offset, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_at,PMPI_FILE_WRITE_AT)(fh, offset, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_at:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_at,PMPI_FILE_WRITE_AT)(fh, offset, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_at_all,MPI_FILE_WRITE_AT_ALL) (int* fh, int* offset, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_at_all,PMPI_FILE_WRITE_AT_ALL)(fh, offset, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_at_all:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_at_all,PMPI_FILE_WRITE_AT_ALL)(fh, offset, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_at_all_begin,MPI_FILE_WRITE_AT_ALL_BEGIN) (int* fh, int* offset, int* buf, int* count, int* datatype, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_at_all_begin,PMPI_FILE_WRITE_AT_ALL_BEGIN)(fh, offset, buf, count, datatype, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_at_all_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_at_all_begin,PMPI_FILE_WRITE_AT_ALL_BEGIN)(fh, offset, buf, count, datatype, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *offset);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_at_all_end,MPI_FILE_WRITE_AT_ALL_END) (int* fh, int* buf, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_at_all_end,PMPI_FILE_WRITE_AT_ALL_END)(fh, buf, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_at_all_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_at_all_end,PMPI_FILE_WRITE_AT_ALL_END)(fh, buf, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_ordered,MPI_FILE_WRITE_ORDERED) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_ordered,PMPI_FILE_WRITE_ORDERED)(fh, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_ordered:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_ordered,PMPI_FILE_WRITE_ORDERED)(fh, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_ordered_begin,MPI_FILE_WRITE_ORDERED_BEGIN) (int* fh, int* buf, int* count, int* datatype, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_ordered_begin,PMPI_FILE_WRITE_ORDERED_BEGIN)(fh, buf, count, datatype, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_ordered_begin:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_ordered_begin,PMPI_FILE_WRITE_ORDERED_BEGIN)(fh, buf, count, datatype, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_ordered_end,MPI_FILE_WRITE_ORDERED_END) (int* fh, int* buf, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_ordered_end,PMPI_FILE_WRITE_ORDERED_END)(fh, buf, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_ordered_end:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_ordered_end,PMPI_FILE_WRITE_ORDERED_END)(fh, buf, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_file_write_shared,MPI_FILE_WRITE_SHARED) (int* fh, int* buf, int* count, int* datatype, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_file_write_shared,PMPI_FILE_WRITE_SHARED)(fh, buf, count, datatype, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_File_write_shared:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_file_write_shared,PMPI_FILE_WRITE_SHARED)(fh, buf, count, datatype, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *fh);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_finalize,MPI_FINALIZE) (int* ierr) {
  lap_check();
  WRITE_TRACE("%s", "MPI_Finalize:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
lap_collect_traces();
 FortranCInterface_GLOBAL(pmpi_finalize,PMPI_FINALIZE)(ierr);
  lap_mpi_initialized = 0;
}

void FortranCInterface_GLOBAL(mpi_finalized,MPI_FINALIZED) (int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_finalized,PMPI_FINALIZED)(flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Finalized:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_finalized,PMPI_FINALIZED)(flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_free_mem,MPI_FREE_MEM) (int* base, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_free_mem,PMPI_FREE_MEM)(base, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Free_mem:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *base);
 FortranCInterface_GLOBAL(pmpi_free_mem,PMPI_FREE_MEM)(base, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_gather,MPI_GATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_gather,PMPI_GATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Gather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_gather,PMPI_GATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_gatherv,MPI_GATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* root, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_gatherv,PMPI_GATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Gatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_gatherv,PMPI_GATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) displs[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_get,MPI_GET) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* win, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_get,PMPI_GET)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_get,PMPI_GET)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *origin_count);
    WRITE_TRACE("%lli:", (long long int) *origin_datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *target_count);
    WRITE_TRACE("%lli:", (long long int) *target_datatype);
    WRITE_TRACE("%lli:", (long long int) *win);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_get_accumulate,MPI_GET_ACCUMULATE) (int* origin_addr, int* origin_count, int* origin_datatype, int* result_addr, int* result_count, int* result_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* op, int* win, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_get_accumulate,PMPI_GET_ACCUMULATE)(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_accumulate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_get_accumulate,PMPI_GET_ACCUMULATE)(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *origin_count);
    WRITE_TRACE("%lli:", (long long int) *origin_datatype);
    WRITE_TRACE("%lli:", (long long int) *result_addr);
    WRITE_TRACE("%lli:", (long long int) *result_count);
    WRITE_TRACE("%lli:", (long long int) *result_datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *target_count);
    WRITE_TRACE("%lli:", (long long int) *target_datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *win);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_get_address,MPI_GET_ADDRESS) (int* location, int* address, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_get_address,PMPI_GET_ADDRESS)(location, address, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_address:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_get_address,PMPI_GET_ADDRESS)(location, address, ierr);
    WRITE_TRACE("%lli:", (long long int) *location);
    WRITE_TRACE("%lli:", (long long int) *address);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_get_count,MPI_GET_COUNT) (int* status, int* datatype, int* count, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_get_count,PMPI_GET_COUNT)(status, datatype, count, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_count:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_get_count,PMPI_GET_COUNT)(status, datatype, count, ierr);
    WRITE_TRACE("%lli:", (long long int) *status);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *count);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_get_elements,MPI_GET_ELEMENTS) (int* status, int* datatype, int* count, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_get_elements,PMPI_GET_ELEMENTS)(status, datatype, count, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_elements:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_get_elements,PMPI_GET_ELEMENTS)(status, datatype, count, ierr);
    WRITE_TRACE("%lli:", (long long int) *status);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *count);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_get_elements_x,MPI_GET_ELEMENTS_X) (int* status, int* datatype, int* count, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_get_elements_x,PMPI_GET_ELEMENTS_X)(status, datatype, count, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_elements_x:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_get_elements_x,PMPI_GET_ELEMENTS_X)(status, datatype, count, ierr);
    WRITE_TRACE("%lli:", (long long int) *status);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *count);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_get_library_version,MPI_GET_LIBRARY_VERSION) (char* version, int* resultlen, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_get_library_version,PMPI_GET_LIBRARY_VERSION)(version, resultlen, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_library_version:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_get_library_version,PMPI_GET_LIBRARY_VERSION)(version, resultlen, ierr);
    WRITE_TRACE("%lli:", (long long int) *version);
    WRITE_TRACE("%lli:", (long long int) *resultlen);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_get_processor_name,MPI_GET_PROCESSOR_NAME) (char* name, int* resultlen, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_get_processor_name,PMPI_GET_PROCESSOR_NAME)(name, resultlen, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_processor_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_get_processor_name,PMPI_GET_PROCESSOR_NAME)(name, resultlen, ierr);
    WRITE_TRACE("%lli:", (long long int) *name);
    WRITE_TRACE("%lli:", (long long int) *resultlen);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_get_version,MPI_GET_VERSION) (int* version, int* subversion, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_get_version,PMPI_GET_VERSION)(version, subversion, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Get_version:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_get_version,PMPI_GET_VERSION)(version, subversion, ierr);
    WRITE_TRACE("%lli:", (long long int) *version);
    WRITE_TRACE("%lli:", (long long int) *subversion);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_graph_create,MPI_GRAPH_CREATE) (int* comm_old, int* nnodes, int* index, int* edges, int* reorder, int* comm_graph, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_graph_create,PMPI_GRAPH_CREATE)(comm_old, nnodes, index, edges, reorder, comm_graph, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_graph_create,PMPI_GRAPH_CREATE)(comm_old, nnodes, index, edges, reorder, comm_graph, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm_old);
    WRITE_TRACE("%lli:", (long long int) *nnodes);
  WRITE_TRACE("%p,%i[", (void*) index, (int) (*nnodes));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*nnodes); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) index[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) edges, (int) index[(*nnodes)-1]);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<index[(*nnodes)-1]; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) edges[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *reorder);
    WRITE_TRACE("%lli:", (long long int) *comm_graph);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_graph_get,MPI_GRAPH_GET) (int* comm, int* maxindex, int* maxedges, int* index, int* edges, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_graph_get,PMPI_GRAPH_GET)(comm, maxindex, maxedges, index, edges, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_graph_get,PMPI_GRAPH_GET)(comm, maxindex, maxedges, index, edges, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *maxindex);
    WRITE_TRACE("%lli:", (long long int) *maxedges);
  WRITE_TRACE("%p,%i[", (void*) index, (int) (*maxindex));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxindex); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) index[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) edges, (int) (*maxedges));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxedges); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) edges[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_graph_map,MPI_GRAPH_MAP) (int* comm, int* nnodes, int* index, int* edges, int* newrank, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_graph_map,PMPI_GRAPH_MAP)(comm, nnodes, index, edges, newrank, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_map:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_graph_map,PMPI_GRAPH_MAP)(comm, nnodes, index, edges, newrank, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *nnodes);
  WRITE_TRACE("%p,%i[", (void*) index, (int) (*nnodes));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*nnodes); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) index[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) edges, (int) index[(*nnodes)-1]);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<index[(*nnodes)-1]; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) edges[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *newrank);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_graph_neighbors,MPI_GRAPH_NEIGHBORS) (int* comm, int* rank, int* maxneighbors, int* neighbors, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_graph_neighbors,PMPI_GRAPH_NEIGHBORS)(comm, rank, maxneighbors, neighbors, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_neighbors:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_graph_neighbors,PMPI_GRAPH_NEIGHBORS)(comm, rank, maxneighbors, neighbors, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *rank);
    WRITE_TRACE("%lli:", (long long int) *maxneighbors);
  WRITE_TRACE("%p,%i[", (void*) neighbors, (int) (*maxneighbors));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*maxneighbors); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) neighbors[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_graph_neighbors_count,MPI_GRAPH_NEIGHBORS_COUNT) (int* comm, int* rank, int* nneighbors, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_graph_neighbors_count,PMPI_GRAPH_NEIGHBORS_COUNT)(comm, rank, nneighbors, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Graph_neighbors_count:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_graph_neighbors_count,PMPI_GRAPH_NEIGHBORS_COUNT)(comm, rank, nneighbors, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *rank);
    WRITE_TRACE("%lli:", (long long int) *nneighbors);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_graphdims_get,MPI_GRAPHDIMS_GET) (int* comm, int* nnodes, int* nedges, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_graphdims_get,PMPI_GRAPHDIMS_GET)(comm, nnodes, nedges, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Graphdims_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_graphdims_get,PMPI_GRAPHDIMS_GET)(comm, nnodes, nedges, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *nnodes);
    WRITE_TRACE("%lli:", (long long int) *nedges);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_grequest_complete,MPI_GREQUEST_COMPLETE) (int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_grequest_complete,PMPI_GREQUEST_COMPLETE)(request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Grequest_complete:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_grequest_complete,PMPI_GREQUEST_COMPLETE)(request, ierr);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_grequest_start,MPI_GREQUEST_START) (int* query_fn, int* free_fn, int* cancel_fn, int* extra_state, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_grequest_start,PMPI_GREQUEST_START)(query_fn, free_fn, cancel_fn, extra_state, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Grequest_start:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_grequest_start,PMPI_GREQUEST_START)(query_fn, free_fn, cancel_fn, extra_state, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *query_fn);
    WRITE_TRACE("%lli:", (long long int) *free_fn);
    WRITE_TRACE("%lli:", (long long int) *cancel_fn);
    WRITE_TRACE("%lli:", (long long int) *extra_state);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_compare,MPI_GROUP_COMPARE) (int* group1, int* group2, int* result, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_compare,PMPI_GROUP_COMPARE)(group1, group2, result, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_compare:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_compare,PMPI_GROUP_COMPARE)(group1, group2, result, ierr);
    WRITE_TRACE("%lli:", (long long int) *group1);
    WRITE_TRACE("%lli:", (long long int) *group2);
    WRITE_TRACE("%lli:", (long long int) *result);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_difference,MPI_GROUP_DIFFERENCE) (int* group1, int* group2, int* newgroup, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_difference,PMPI_GROUP_DIFFERENCE)(group1, group2, newgroup, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_difference:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_difference,PMPI_GROUP_DIFFERENCE)(group1, group2, newgroup, ierr);
    WRITE_TRACE("%lli:", (long long int) *group1);
    WRITE_TRACE("%lli:", (long long int) *group2);
    WRITE_TRACE("%lli:", (long long int) *newgroup);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_excl,MPI_GROUP_EXCL) (int* group, int* n, int* ranks, int* newgroup, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_excl,PMPI_GROUP_EXCL)(group, n, ranks, newgroup, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_excl:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_excl,PMPI_GROUP_EXCL)(group, n, ranks, newgroup, ierr);
    WRITE_TRACE("%lli:", (long long int) *group);
    WRITE_TRACE("%lli:", (long long int) *n);
  WRITE_TRACE("%p,%i[", (void*) ranks, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) ranks[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *newgroup);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_free,MPI_GROUP_FREE) (int* group, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_free,PMPI_GROUP_FREE)(group, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *group);
 FortranCInterface_GLOBAL(pmpi_group_free,PMPI_GROUP_FREE)(group, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_incl,MPI_GROUP_INCL) (int* group, int* n, int* ranks, int* newgroup, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_incl,PMPI_GROUP_INCL)(group, n, ranks, newgroup, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_incl:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_incl,PMPI_GROUP_INCL)(group, n, ranks, newgroup, ierr);
    WRITE_TRACE("%lli:", (long long int) *group);
    WRITE_TRACE("%lli:", (long long int) *n);
  WRITE_TRACE("%p,%i[", (void*) ranks, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) ranks[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *newgroup);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_intersection,MPI_GROUP_INTERSECTION) (int* group1, int* group2, int* newgroup, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_intersection,PMPI_GROUP_INTERSECTION)(group1, group2, newgroup, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_intersection:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_intersection,PMPI_GROUP_INTERSECTION)(group1, group2, newgroup, ierr);
    WRITE_TRACE("%lli:", (long long int) *group1);
    WRITE_TRACE("%lli:", (long long int) *group2);
    WRITE_TRACE("%lli:", (long long int) *newgroup);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_range_excl,MPI_GROUP_RANGE_EXCL) (int* group, int* n, int* ranges, int* newgroup, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_range_excl,PMPI_GROUP_RANGE_EXCL)(group, n, ranges, newgroup, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_range_excl:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_range_excl,PMPI_GROUP_RANGE_EXCL)(group, n, ranges, newgroup, ierr);
    WRITE_TRACE("%lli:", (long long int) *group);
    WRITE_TRACE("%lli:", (long long int) *n);
  WRITE_TRACE("%p,%i[", (void*) ranges, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("[%i,%i,%i];", *(&(ranges[trace_elem_idx])+0), *(&(ranges[trace_elem_idx])+1), *(&(ranges[trace_elem_idx])+2));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *newgroup);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_range_incl,MPI_GROUP_RANGE_INCL) (int* group, int* n, int* ranges, int* newgroup, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_range_incl,PMPI_GROUP_RANGE_INCL)(group, n, ranges, newgroup, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_range_incl:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_range_incl,PMPI_GROUP_RANGE_INCL)(group, n, ranges, newgroup, ierr);
    WRITE_TRACE("%lli:", (long long int) *group);
    WRITE_TRACE("%lli:", (long long int) *n);
  WRITE_TRACE("%p,%i[", (void*) ranges, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("[%i,%i,%i];", *(&(ranges[trace_elem_idx])+0), *(&(ranges[trace_elem_idx])+1), *(&(ranges[trace_elem_idx])+2));
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *newgroup);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_rank,MPI_GROUP_RANK) (int* group, int* rank, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_rank,PMPI_GROUP_RANK)(group, rank, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_rank:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_rank,PMPI_GROUP_RANK)(group, rank, ierr);
    WRITE_TRACE("%lli:", (long long int) *group);
    WRITE_TRACE("%lli:", (long long int) *rank);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_size,MPI_GROUP_SIZE) (int* group, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_size,PMPI_GROUP_SIZE)(group, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_size,PMPI_GROUP_SIZE)(group, size, ierr);
    WRITE_TRACE("%lli:", (long long int) *group);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_translate_ranks,MPI_GROUP_TRANSLATE_RANKS) (int* group1, int* n, int* ranks1, int* group2, int* ranks2, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_translate_ranks,PMPI_GROUP_TRANSLATE_RANKS)(group1, n, ranks1, group2, ranks2, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_translate_ranks:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_translate_ranks,PMPI_GROUP_TRANSLATE_RANKS)(group1, n, ranks1, group2, ranks2, ierr);
    WRITE_TRACE("%lli:", (long long int) *group1);
    WRITE_TRACE("%lli:", (long long int) *n);
  WRITE_TRACE("%p,%i[", (void*) ranks1, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) ranks1[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *group2);
  WRITE_TRACE("%p,%i[", (void*) ranks2, (int) (*n));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*n); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) ranks2[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_group_union,MPI_GROUP_UNION) (int* group1, int* group2, int* newgroup, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_group_union,PMPI_GROUP_UNION)(group1, group2, newgroup, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Group_union:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_group_union,PMPI_GROUP_UNION)(group1, group2, newgroup, ierr);
    WRITE_TRACE("%lli:", (long long int) *group1);
    WRITE_TRACE("%lli:", (long long int) *group2);
    WRITE_TRACE("%lli:", (long long int) *newgroup);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_iallgather,MPI_IALLGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_iallgather,PMPI_IALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Iallgather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_iallgather,PMPI_IALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_iallgatherv,MPI_IALLGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_iallgatherv,PMPI_IALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Iallgatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_iallgatherv,PMPI_IALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) displs[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_iallreduce,MPI_IALLREDUCE) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_iallreduce,PMPI_IALLREDUCE)(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Iallreduce:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_iallreduce,PMPI_IALLREDUCE)(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ialltoall,MPI_IALLTOALL) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ialltoall,PMPI_IALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ialltoall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ialltoall,PMPI_IALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ialltoallv,MPI_IALLTOALLV) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtype, int* recvbuf, int* recvcounts, int* rdispls, int* recvtype, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ialltoallv,PMPI_IALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ialltoallv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ialltoallv,PMPI_IALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) rdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ialltoallw,MPI_IALLTOALLW) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtypes, int* recvbuf, int* recvcounts, int* rdispls, int* recvtypes, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ialltoallw,PMPI_IALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ialltoallw:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ialltoallw,PMPI_IALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sendtypes, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendtypes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) rdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) recvtypes, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvtypes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ibarrier,MPI_IBARRIER) (int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ibarrier,PMPI_IBARRIER)(comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ibarrier:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ibarrier,PMPI_IBARRIER)(comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ibcast,MPI_IBCAST) (int* buffer, int* count, int* datatype, int* root, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ibcast,PMPI_IBCAST)(buffer, count, datatype, root, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ibcast:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ibcast,PMPI_IBCAST)(buffer, count, datatype, root, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buffer);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ibsend,MPI_IBSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ibsend,PMPI_IBSEND)(buf, count, datatype, dest, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ibsend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ibsend,PMPI_IBSEND)(buf, count, datatype, dest, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_iexscan,MPI_IEXSCAN) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_iexscan,PMPI_IEXSCAN)(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Iexscan:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_iexscan,PMPI_IEXSCAN)(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_igather,MPI_IGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_igather,PMPI_IGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Igather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_igather,PMPI_IGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_igatherv,MPI_IGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* root, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_igatherv,PMPI_IGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Igatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_igatherv,PMPI_IGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, request, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) displs[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_improbe,MPI_IMPROBE) (int* source, int* tag, int* comm, int* flag, int* message, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_improbe,PMPI_IMPROBE)(source, tag, comm, flag, message, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Improbe:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_improbe,PMPI_IMPROBE)(source, tag, comm, flag, message, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *source);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *flag);
    WRITE_TRACE("%lli:", (long long int) *message);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_imrecv,MPI_IMRECV) (int* buf, int* count, int* type, int* message, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_imrecv,PMPI_IMRECV)(buf, count, type, message, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Imrecv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_imrecv,PMPI_IMRECV)(buf, count, type, message, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *type);
    WRITE_TRACE("%lli:", (long long int) *message);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ineighbor_allgather,MPI_INEIGHBOR_ALLGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ineighbor_allgather,PMPI_INEIGHBOR_ALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_allgather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ineighbor_allgather,PMPI_INEIGHBOR_ALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ineighbor_allgatherv,MPI_INEIGHBOR_ALLGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ineighbor_allgatherv,PMPI_INEIGHBOR_ALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_allgatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ineighbor_allgatherv,PMPI_INEIGHBOR_ALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count( MPI_Comm_f2c((*comm)),  &ideg,  &odeg,  &wted);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) displs[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ineighbor_alltoall,MPI_INEIGHBOR_ALLTOALL) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ineighbor_alltoall,PMPI_INEIGHBOR_ALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_alltoall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ineighbor_alltoall,PMPI_INEIGHBOR_ALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ineighbor_alltoallv,MPI_INEIGHBOR_ALLTOALLV) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtype, int* recvbuf, int* recvcounts, int* rdispls, int* recvtype, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ineighbor_alltoallv,PMPI_INEIGHBOR_ALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_alltoallv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ineighbor_alltoallv,PMPI_INEIGHBOR_ALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count( MPI_Comm_f2c((*comm)),  &ideg,  &odeg,  &wted);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) rdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ineighbor_alltoallw,MPI_INEIGHBOR_ALLTOALLW) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtypes, int* recvbuf, int* recvcounts, int* rdispls, int* recvtypes, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ineighbor_alltoallw,PMPI_INEIGHBOR_ALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ineighbor_alltoallw:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ineighbor_alltoallw,PMPI_INEIGHBOR_ALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count( MPI_Comm_f2c((*comm)),  &ideg,  &odeg,  &wted);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sendtypes, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendtypes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) rdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) recvtypes, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvtypes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_info_create,MPI_INFO_CREATE) (int* info, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_info_create,PMPI_INFO_CREATE)(info, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_info_create,PMPI_INFO_CREATE)(info, ierr);
    WRITE_TRACE("%lli:", (long long int) *info);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_info_delete,MPI_INFO_DELETE) (int* info, char* key, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_info_delete,PMPI_INFO_DELETE)(info, key, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_delete:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_info_delete,PMPI_INFO_DELETE)(info, key, ierr);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *key);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_info_dup,MPI_INFO_DUP) (int* info, int* newinfo, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_info_dup,PMPI_INFO_DUP)(info, newinfo, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_dup:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_info_dup,PMPI_INFO_DUP)(info, newinfo, ierr);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *newinfo);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_info_free,MPI_INFO_FREE) (int* info, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_info_free,PMPI_INFO_FREE)(info, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *info);
 FortranCInterface_GLOBAL(pmpi_info_free,PMPI_INFO_FREE)(info, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_info_get,MPI_INFO_GET) (int* info, char* key, int* valuelen, char* value, int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_info_get,PMPI_INFO_GET)(info, key, valuelen, value, flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_get:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_info_get,PMPI_INFO_GET)(info, key, valuelen, value, flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *key);
    WRITE_TRACE("%lli:", (long long int) *valuelen);
    WRITE_TRACE("%lli:", (long long int) *value);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_info_get_nkeys,MPI_INFO_GET_NKEYS) (int* info, int* nkeys, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_info_get_nkeys,PMPI_INFO_GET_NKEYS)(info, nkeys, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_get_nkeys:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_info_get_nkeys,PMPI_INFO_GET_NKEYS)(info, nkeys, ierr);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *nkeys);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_info_get_nthkey,MPI_INFO_GET_NTHKEY) (int* info, int* n, char* key, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_info_get_nthkey,PMPI_INFO_GET_NTHKEY)(info, n, key, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_get_nthkey:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_info_get_nthkey,PMPI_INFO_GET_NTHKEY)(info, n, key, ierr);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *n);
    WRITE_TRACE("%lli:", (long long int) *key);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_info_get_valuelen,MPI_INFO_GET_VALUELEN) (int* info, char* key, int* valuelen, int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_info_get_valuelen,PMPI_INFO_GET_VALUELEN)(info, key, valuelen, flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_get_valuelen:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_info_get_valuelen,PMPI_INFO_GET_VALUELEN)(info, key, valuelen, flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *key);
    WRITE_TRACE("%lli:", (long long int) *valuelen);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_info_set,MPI_INFO_SET) (int* info, char* key, char* value, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_info_set,PMPI_INFO_SET)(info, key, value, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Info_set:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_info_set,PMPI_INFO_SET)(info, key, value, ierr);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *key);
    WRITE_TRACE("%lli:", (long long int) *value);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_init,MPI_INIT) (int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_init,PMPI_INIT)(ierr);
    return;
  }
  lap_check();
  double curr_time;
  WRITE_TRACE("%s", "MPI_Init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_init,PMPI_INIT)(ierr);
  curr_time = PMPI_Wtime();
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  WRITE_TRACE("# Current timestamp: %0.2f\n", curr_time * 1e6);
}

void FortranCInterface_GLOBAL(mpi_init_thread,MPI_INIT_THREAD) (int *required, int *provided, int *ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_init_thread,PMPI_INIT_THREAD)(required, provided, ierr);
    return;
  }
  lap_check();
  double curr_time;
  WRITE_TRACE("%s", "MPI_Init_thread:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
  // printf("[DEBUG] MPI_Init_thread\n", required, *provided);
 FortranCInterface_GLOBAL(pmpi_init_thread,PMPI_INIT_THREAD)(required, provided, ierr);
  curr_time = PMPI_Wtime();
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
  WRITE_TRACE("# Current timestamp: %0.2f\n", curr_time * 1e6);
}

void FortranCInterface_GLOBAL(mpi_initialized,MPI_INITIALIZED) (int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_initialized,PMPI_INITIALIZED)(flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Initialized:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_initialized,PMPI_INITIALIZED)(flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_intercomm_create,MPI_INTERCOMM_CREATE) (int* local_comm, int* local_leader, int* bridge_comm, int* remote_leader, int* tag, int* newintercomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_intercomm_create,PMPI_INTERCOMM_CREATE)(local_comm, local_leader, bridge_comm, remote_leader, tag, newintercomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Intercomm_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_intercomm_create,PMPI_INTERCOMM_CREATE)(local_comm, local_leader, bridge_comm, remote_leader, tag, newintercomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *local_comm);
    WRITE_TRACE("%lli:", (long long int) *local_leader);
    WRITE_TRACE("%lli:", (long long int) *bridge_comm);
    WRITE_TRACE("%lli:", (long long int) *remote_leader);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *newintercomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_intercomm_merge,MPI_INTERCOMM_MERGE) (int* intercomm, int* high, int* newintercomm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_intercomm_merge,PMPI_INTERCOMM_MERGE)(intercomm, high, newintercomm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Intercomm_merge:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_intercomm_merge,PMPI_INTERCOMM_MERGE)(intercomm, high, newintercomm, ierr);
    WRITE_TRACE("%lli:", (long long int) *intercomm);
    WRITE_TRACE("%lli:", (long long int) *high);
    WRITE_TRACE("%lli:", (long long int) *newintercomm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_iprobe,MPI_IPROBE) (int* source, int* tag, int* comm, int* flag, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_iprobe,PMPI_IPROBE)(source, tag, comm, flag, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Iprobe:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_iprobe,PMPI_IPROBE)(source, tag, comm, flag, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *source);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *flag);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_irecv,MPI_IRECV) (int* buf, int* count, int* datatype, int* source, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_irecv,PMPI_IRECV)(buf, count, datatype, source, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Irecv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_irecv,PMPI_IRECV)(buf, count, datatype, source, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *source);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lu:", (unsigned long)request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ireduce,MPI_IREDUCE) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* root, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ireduce,PMPI_IREDUCE)(sendbuf, recvbuf, count, datatype, op, root, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ireduce:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ireduce,PMPI_IREDUCE)(sendbuf, recvbuf, count, datatype, op, root, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ireduce_scatter,MPI_IREDUCE_SCATTER) (int* sendbuf, int* recvbuf, int* recvcounts, int* datatype, int* op, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ireduce_scatter,PMPI_IREDUCE_SCATTER)(sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ireduce_scatter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ireduce_scatter,PMPI_IREDUCE_SCATTER)(sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ireduce_scatter_block,MPI_IREDUCE_SCATTER_BLOCK) (int* sendbuf, int* recvbuf, int* recvcount, int* datatype, int* op, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ireduce_scatter_block,PMPI_IREDUCE_SCATTER_BLOCK)(sendbuf, recvbuf, recvcount, datatype, op, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ireduce_scatter_block:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ireduce_scatter_block,PMPI_IREDUCE_SCATTER_BLOCK)(sendbuf, recvbuf, recvcount, datatype, op, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_irsend,MPI_IRSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_irsend,PMPI_IRSEND)(buf, count, datatype, dest, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Irsend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_irsend,PMPI_IRSEND)(buf, count, datatype, dest, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_is_thread_main,MPI_IS_THREAD_MAIN) (int* flag, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_is_thread_main,PMPI_IS_THREAD_MAIN)(flag, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Is_thread_main:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_is_thread_main,PMPI_IS_THREAD_MAIN)(flag, ierr);
    WRITE_TRACE("%lli:", (long long int) *flag);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_iscan,MPI_ISCAN) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_iscan,PMPI_ISCAN)(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Iscan:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_iscan,PMPI_ISCAN)(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_iscatter,MPI_ISCATTER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_iscatter,PMPI_ISCATTER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Iscatter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_iscatter,PMPI_ISCATTER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_iscatterv,MPI_ISCATTERV) (int* sendbuf, int* sendcounts, int* displs, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_iscatterv,PMPI_ISCATTERV)(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Iscatterv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_iscatterv,PMPI_ISCATTERV)(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) (rank == (*root) ? size : 0));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(rank == (*root) ? size : 0); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) (rank == (*root) ? size : 0));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(rank == (*root) ? size : 0); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) displs[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_isend,MPI_ISEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_isend,PMPI_ISEND)(buf, count, datatype, dest, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Isend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_isend,PMPI_ISEND)(buf, count, datatype, dest, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lu:", (unsigned long)request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_issend,MPI_ISSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_issend,PMPI_ISSEND)(buf, count, datatype, dest, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Issend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_issend,PMPI_ISSEND)(buf, count, datatype, dest, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_keyval_create,MPI_KEYVAL_CREATE) (int* copy_fn, int* delete_fn, int* keyval, int* extra_state, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_keyval_create,PMPI_KEYVAL_CREATE)(copy_fn, delete_fn, keyval, extra_state, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Keyval_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_keyval_create,PMPI_KEYVAL_CREATE)(copy_fn, delete_fn, keyval, extra_state, ierr);
    WRITE_TRACE("%lli:", (long long int) *copy_fn);
    WRITE_TRACE("%lli:", (long long int) *delete_fn);
    WRITE_TRACE("%lli:", (long long int) *keyval);
    WRITE_TRACE("%lli:", (long long int) *extra_state);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_keyval_free,MPI_KEYVAL_FREE) (int* keyval, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_keyval_free,PMPI_KEYVAL_FREE)(keyval, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Keyval_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *keyval);
 FortranCInterface_GLOBAL(pmpi_keyval_free,PMPI_KEYVAL_FREE)(keyval, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_lookup_name,MPI_LOOKUP_NAME) (char* service_name, int* info, char* port_name, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_lookup_name,PMPI_LOOKUP_NAME)(service_name, info, port_name, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Lookup_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_lookup_name,PMPI_LOOKUP_NAME)(service_name, info, port_name, ierr);
    WRITE_TRACE("%lli:", (long long int) *service_name);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *port_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_mprobe,MPI_MPROBE) (int* source, int* tag, int* comm, int* message, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_mprobe,PMPI_MPROBE)(source, tag, comm, message, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Mprobe:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_mprobe,PMPI_MPROBE)(source, tag, comm, message, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *source);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *message);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_mrecv,MPI_MRECV) (int* buf, int* count, int* type, int* message, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_mrecv,PMPI_MRECV)(buf, count, type, message, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Mrecv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_mrecv,PMPI_MRECV)(buf, count, type, message, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *type);
    WRITE_TRACE("%lli:", (long long int) *message);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_neighbor_allgather,MPI_NEIGHBOR_ALLGATHER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_neighbor_allgather,PMPI_NEIGHBOR_ALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_allgather:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_neighbor_allgather,PMPI_NEIGHBOR_ALLGATHER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_neighbor_allgatherv,MPI_NEIGHBOR_ALLGATHERV) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcounts, int* displs, int* recvtype, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_neighbor_allgatherv,PMPI_NEIGHBOR_ALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_allgatherv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_neighbor_allgatherv,PMPI_NEIGHBOR_ALLGATHERV)(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count( MPI_Comm_f2c((*comm)),  &ideg,  &odeg,  &wted);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) displs[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_neighbor_alltoall,MPI_NEIGHBOR_ALLTOALL) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_neighbor_alltoall,PMPI_NEIGHBOR_ALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_alltoall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_neighbor_alltoall,PMPI_NEIGHBOR_ALLTOALL)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_neighbor_alltoallv,MPI_NEIGHBOR_ALLTOALLV) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtype, int* recvbuf, int* recvcounts, int* rdispls, int* recvtype, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_neighbor_alltoallv,PMPI_NEIGHBOR_ALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_alltoallv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_neighbor_alltoallv,PMPI_NEIGHBOR_ALLTOALLV)(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count( MPI_Comm_f2c((*comm)),  &ideg,  &odeg,  &wted);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) rdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_neighbor_alltoallw,MPI_NEIGHBOR_ALLTOALLW) (int* sendbuf, int* sendcounts, int* sdispls, int* sendtypes, int* recvbuf, int* recvcounts, int* rdispls, int* recvtypes, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_neighbor_alltoallw,PMPI_NEIGHBOR_ALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Neighbor_alltoallw:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_neighbor_alltoallw,PMPI_NEIGHBOR_ALLTOALLW)(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);
int ideg, odeg, wted; PMPI_Dist_graph_neighbors_count( MPI_Comm_f2c((*comm)),  &ideg,  &odeg,  &wted);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sdispls, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) sendtypes, (int) odeg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<odeg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendtypes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) rdispls, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) rdispls[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) recvtypes, (int) ideg);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<ideg; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvtypes[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_op_commutative,MPI_OP_COMMUTATIVE) (int* op, int* commute, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_op_commutative,PMPI_OP_COMMUTATIVE)(op, commute, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Op_commutative:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_op_commutative,PMPI_OP_COMMUTATIVE)(op, commute, ierr);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *commute);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_op_create,MPI_OP_CREATE) (int* function, int* commute, int* op, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_op_create,PMPI_OP_CREATE)(function, commute, op, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Op_create:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_op_create,PMPI_OP_CREATE)(function, commute, op, ierr);
    WRITE_TRACE("%lli:", (long long int) *function);
    WRITE_TRACE("%lli:", (long long int) *commute);
    WRITE_TRACE("%lli:", (long long int) *op);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_op_free,MPI_OP_FREE) (int* op, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_op_free,PMPI_OP_FREE)(op, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Op_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *op);
 FortranCInterface_GLOBAL(pmpi_op_free,PMPI_OP_FREE)(op, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_open_port,MPI_OPEN_PORT) (int* info, char* port_name, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_open_port,PMPI_OPEN_PORT)(info, port_name, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Open_port:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_open_port,PMPI_OPEN_PORT)(info, port_name, ierr);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *port_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_pack,MPI_PACK) (int* inbuf, int* incount, int* datatype, int* outbuf, int* outsize, int* position, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_pack,PMPI_PACK)(inbuf, incount, datatype, outbuf, outsize, position, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Pack:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_pack,PMPI_PACK)(inbuf, incount, datatype, outbuf, outsize, position, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *inbuf);
    WRITE_TRACE("%lli:", (long long int) *incount);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *outbuf);
    WRITE_TRACE("%lli:", (long long int) *outsize);
    WRITE_TRACE("%lli:", (long long int) *position);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_pack_external,MPI_PACK_EXTERNAL) (char* datarep, int* inbuf, int* incount, int* datatype, int* outbuf, int* outsize, int* position, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_pack_external,PMPI_PACK_EXTERNAL)(datarep, inbuf, incount, datatype, outbuf, outsize, position, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Pack_external:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_pack_external,PMPI_PACK_EXTERNAL)(datarep, inbuf, incount, datatype, outbuf, outsize, position, ierr);
  WRITE_TRACE("%p,%i[", (void*) datarep, (int) strlen(datarep));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<strlen(datarep); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) datarep[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *inbuf);
    WRITE_TRACE("%lli:", (long long int) *incount);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *outbuf);
    WRITE_TRACE("%lli:", (long long int) *outsize);
    WRITE_TRACE("%lli:", (long long int) *position);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_pack_external_size,MPI_PACK_EXTERNAL_SIZE) (char* datarep, int* incount, int* datatype, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_pack_external_size,PMPI_PACK_EXTERNAL_SIZE)(datarep, incount, datatype, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Pack_external_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_pack_external_size,PMPI_PACK_EXTERNAL_SIZE)(datarep, incount, datatype, size, ierr);
  WRITE_TRACE("%p,%i[", (void*) datarep, (int) strlen(datarep));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<strlen(datarep); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) datarep[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *incount);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_pack_size,MPI_PACK_SIZE) (int* incount, int* datatype, int* comm, int* size, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_pack_size,PMPI_PACK_SIZE)(incount, datatype, comm, size, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Pack_size:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_pack_size,PMPI_PACK_SIZE)(incount, datatype, comm, size, ierr);
    WRITE_TRACE("%lli:", (long long int) *incount);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *size);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_pcontrol,MPI_PCONTROL) (int* level) {
  if (*level == 0) { lap_tracing_enabled = 0; lap_elem_tracing_enabled = 0; lap_backtrace_enabled = 0; }
  if (*level == 1) { lap_tracing_enabled = 1; lap_elem_tracing_enabled = 0; lap_backtrace_enabled = 0; }
  if (*level == 2) { lap_tracing_enabled = 1; lap_elem_tracing_enabled = 1; lap_backtrace_enabled = 0; }
  if (*level == 3) { lap_tracing_enabled = 1; lap_elem_tracing_enabled = 1; lap_backtrace_enabled = 1; }
  lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
  /* WRITE_TRACE("  # backtrace [%s]\n", lap_backtrace_buf); */
  WRITE_TRACE("# Current timestamp: %0.2f\n", PMPI_Wtime() * 1e6);
  /* WRITE_TRACE("# pcontrol with value / epoch %i)\n", *level); */
}

void FortranCInterface_GLOBAL(mpi_probe,MPI_PROBE) (int* source, int* tag, int* comm, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_probe,PMPI_PROBE)(source, tag, comm, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Probe:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_probe,PMPI_PROBE)(source, tag, comm, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *source);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_publish_name,MPI_PUBLISH_NAME) (char* service_name, int* info, char* port_name, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_publish_name,PMPI_PUBLISH_NAME)(service_name, info, port_name, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Publish_name:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_publish_name,PMPI_PUBLISH_NAME)(service_name, info, port_name, ierr);
    WRITE_TRACE("%lli:", (long long int) *service_name);
    WRITE_TRACE("%lli:", (long long int) *info);
    WRITE_TRACE("%lli:", (long long int) *port_name);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_put,MPI_PUT) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* win, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_put,PMPI_PUT)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Put:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_put,PMPI_PUT)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *origin_count);
    WRITE_TRACE("%lli:", (long long int) *origin_datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *target_count);
    WRITE_TRACE("%lli:", (long long int) *target_datatype);
    WRITE_TRACE("%lli:", (long long int) *win);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_query_thread,MPI_QUERY_THREAD) (int* provided, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_query_thread,PMPI_QUERY_THREAD)(provided, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Query_thread:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_query_thread,PMPI_QUERY_THREAD)(provided, ierr);
    WRITE_TRACE("%lli:", (long long int) *provided);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_raccumulate,MPI_RACCUMULATE) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* op, int* win, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_raccumulate,PMPI_RACCUMULATE)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Raccumulate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_raccumulate,PMPI_RACCUMULATE)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *origin_count);
    WRITE_TRACE("%lli:", (long long int) *origin_datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *target_count);
    WRITE_TRACE("%lli:", (long long int) *target_datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *win);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_recv,MPI_RECV) (int* buf, int* count, int* datatype, int* source, int* tag, int* comm, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_recv,PMPI_RECV)(buf, count, datatype, source, tag, comm, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Recv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_recv,PMPI_RECV)(buf, count, datatype, source, tag, comm, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *source);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_recv_init,MPI_RECV_INIT) (int* buf, int* count, int* datatype, int* source, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_recv_init,PMPI_RECV_INIT)(buf, count, datatype, source, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Recv_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_recv_init,PMPI_RECV_INIT)(buf, count, datatype, source, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *source);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_reduce,MPI_REDUCE) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* root, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_reduce,PMPI_REDUCE)(sendbuf, recvbuf, count, datatype, op, root, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Reduce:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_reduce,PMPI_REDUCE)(sendbuf, recvbuf, count, datatype, op, root, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_reduce_local,MPI_REDUCE_LOCAL) (int* inbuf, int* inoutbuf, int* count, int* datatype, int* op, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_reduce_local,PMPI_REDUCE_LOCAL)(inbuf, inoutbuf, count, datatype, op, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Reduce_local:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_reduce_local,PMPI_REDUCE_LOCAL)(inbuf, inoutbuf, count, datatype, op, ierr);
    WRITE_TRACE("%lli:", (long long int) *inbuf);
    WRITE_TRACE("%lli:", (long long int) *inoutbuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_reduce_scatter,MPI_REDUCE_SCATTER) (int* sendbuf, int* recvbuf, int* recvcounts, int* datatype, int* op, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_reduce_scatter,PMPI_REDUCE_SCATTER)(sendbuf, recvbuf, recvcounts, datatype, op, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Reduce_scatter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_reduce_scatter,PMPI_REDUCE_SCATTER)(sendbuf, recvbuf, recvcounts, datatype, op, comm, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
  WRITE_TRACE("%p,%i[", (void*) recvcounts, (int) size);
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<size; trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) recvcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_reduce_scatter_block,MPI_REDUCE_SCATTER_BLOCK) (int* sendbuf, int* recvbuf, int* recvcount, int* datatype, int* op, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_reduce_scatter_block,PMPI_REDUCE_SCATTER_BLOCK)(sendbuf, recvbuf, recvcount, datatype, op, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Reduce_scatter_block:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_reduce_scatter_block,PMPI_REDUCE_SCATTER_BLOCK)(sendbuf, recvbuf, recvcount, datatype, op, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_register_datarep,MPI_REGISTER_DATAREP) (char* datarep, int* read_conversion_fn, int* write_conversion_fn, int* dtype_file_extent_fn, int* extra_state, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_register_datarep,PMPI_REGISTER_DATAREP)(datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Register_datarep:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_register_datarep,PMPI_REGISTER_DATAREP)(datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state, ierr);
    WRITE_TRACE("%lli:", (long long int) *datarep);
    WRITE_TRACE("%lli:", (long long int) *read_conversion_fn);
    WRITE_TRACE("%lli:", (long long int) *write_conversion_fn);
    WRITE_TRACE("%lli:", (long long int) *dtype_file_extent_fn);
    WRITE_TRACE("%lli:", (long long int) *extra_state);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_request_free,MPI_REQUEST_FREE) (int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_request_free,PMPI_REQUEST_FREE)(request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Request_free:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
    WRITE_TRACE("%lli:", (long long int) *request);
 FortranCInterface_GLOBAL(pmpi_request_free,PMPI_REQUEST_FREE)(request, ierr);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_request_get_status,MPI_REQUEST_GET_STATUS) (int* request, int* flag, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_request_get_status,PMPI_REQUEST_GET_STATUS)(request, flag, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Request_get_status:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_request_get_status,PMPI_REQUEST_GET_STATUS)(request, flag, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *request);
    WRITE_TRACE("%lli:", (long long int) *flag);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_rget,MPI_RGET) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* win, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_rget,PMPI_RGET)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Rget:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_rget,PMPI_RGET)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *origin_count);
    WRITE_TRACE("%lli:", (long long int) *origin_datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *target_count);
    WRITE_TRACE("%lli:", (long long int) *target_datatype);
    WRITE_TRACE("%lli:", (long long int) *win);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_rget_accumulate,MPI_RGET_ACCUMULATE) (int* origin_addr, int* origin_count, int* origin_datatype, int* result_addr, int* result_count, int* result_datatype, int* target_rank, int* target_disp, int* target_count, int* target_datatype, int* op, int* win, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_rget_accumulate,PMPI_RGET_ACCUMULATE)(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Rget_accumulate:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_rget_accumulate,PMPI_RGET_ACCUMULATE)(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *origin_count);
    WRITE_TRACE("%lli:", (long long int) *origin_datatype);
    WRITE_TRACE("%lli:", (long long int) *result_addr);
    WRITE_TRACE("%lli:", (long long int) *result_count);
    WRITE_TRACE("%lli:", (long long int) *result_datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *target_count);
    WRITE_TRACE("%lli:", (long long int) *target_datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *win);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_rput,MPI_RPUT) (int* origin_addr, int* origin_count, int* origin_datatype, int* target_rank, int* target_disp, int* target_cout, int* target_datatype, int* win, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_rput,PMPI_RPUT)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_cout, target_datatype, win, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Rput:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_rput,PMPI_RPUT)(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_cout, target_datatype, win, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *origin_addr);
    WRITE_TRACE("%lli:", (long long int) *origin_count);
    WRITE_TRACE("%lli:", (long long int) *origin_datatype);
    WRITE_TRACE("%lli:", (long long int) *target_rank);
    WRITE_TRACE("%lli:", (long long int) *target_disp);
    WRITE_TRACE("%lli:", (long long int) *target_cout);
    WRITE_TRACE("%lli:", (long long int) *target_datatype);
    WRITE_TRACE("%lli:", (long long int) *win);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_rsend,MPI_RSEND) (int* ibuf, int* count, int* datatype, int* dest, int* tag, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_rsend,PMPI_RSEND)(ibuf, count, datatype, dest, tag, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Rsend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_rsend,PMPI_RSEND)(ibuf, count, datatype, dest, tag, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *ibuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_rsend_init,MPI_RSEND_INIT) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_rsend_init,PMPI_RSEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Rsend_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_rsend_init,PMPI_RSEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_scan,MPI_SCAN) (int* sendbuf, int* recvbuf, int* count, int* datatype, int* op, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_scan,PMPI_SCAN)(sendbuf, recvbuf, count, datatype, op, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Scan:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_scan,PMPI_SCAN)(sendbuf, recvbuf, count, datatype, op, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *op);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_scatter,MPI_SCATTER) (int* sendbuf, int* sendcount, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_scatter,PMPI_SCATTER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Scatter:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_scatter,PMPI_SCATTER)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_scatterv,MPI_SCATTERV) (int* sendbuf, int* sendcounts, int* displs, int* sendtype, int* recvbuf, int* recvcount, int* recvtype, int* root, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_scatterv,PMPI_SCATTERV)(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Scatterv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_scatterv,PMPI_SCATTERV)(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
int rank, size; PMPI_Comm_size( MPI_Comm_f2c((*comm)),  &size); PMPI_Comm_rank( MPI_Comm_f2c((*comm)),  &rank);
//end of prologs
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
  WRITE_TRACE("%p,%i[", (void*) sendcounts, (int) (rank==(*root) ? size : 0));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(rank==(*root) ? size : 0); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) sendcounts[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%p,%i[", (void*) displs, (int) (rank==(*root) ? size : 0));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(rank==(*root) ? size : 0); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) displs[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *root);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_send,MPI_SEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_send,PMPI_SEND)(buf, count, datatype, dest, tag, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Send:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_send,PMPI_SEND)(buf, count, datatype, dest, tag, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_send_init,MPI_SEND_INIT) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_send_init,PMPI_SEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Send_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_send_init,PMPI_SEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_sendrecv,MPI_SENDRECV) (int* sendbuf, int* sendcount, int* sendtype, int* dest, int* sendtag, int* recvbuf, int* recvcount, int* recvtype, int* source, int* recvtag, int* comm, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_sendrecv,PMPI_SENDRECV)(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Sendrecv:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_sendrecv,PMPI_SENDRECV)(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *sendbuf);
    WRITE_TRACE("%lli:", (long long int) *sendcount);
    WRITE_TRACE("%lli:", (long long int) *sendtype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *sendtag);
    WRITE_TRACE("%lli:", (long long int) *recvbuf);
    WRITE_TRACE("%lli:", (long long int) *recvcount);
    WRITE_TRACE("%lli:", (long long int) *recvtype);
    WRITE_TRACE("%lli:", (long long int) *source);
    WRITE_TRACE("%lli:", (long long int) *recvtag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_sendrecv_replace,MPI_SENDRECV_REPLACE) (int* buf, int* count, int* datatype, int* dest, int* sendtag, int* source, int* recvtag, int* comm, int* status, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_sendrecv_replace,PMPI_SENDRECV_REPLACE)(buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Sendrecv_replace:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_sendrecv_replace,PMPI_SENDRECV_REPLACE)(buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *sendtag);
    WRITE_TRACE("%lli:", (long long int) *source);
    WRITE_TRACE("%lli:", (long long int) *recvtag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *status);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ssend,MPI_SSEND) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ssend,PMPI_SSEND)(buf, count, datatype, dest, tag, comm, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ssend:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ssend,PMPI_SSEND)(buf, count, datatype, dest, tag, comm, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_ssend_init,MPI_SSEND_INIT) (int* buf, int* count, int* datatype, int* dest, int* tag, int* comm, int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_ssend_init,PMPI_SSEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Ssend_init:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_ssend_init,PMPI_SSEND_INIT)(buf, count, datatype, dest, tag, comm, request, ierr);
    WRITE_TRACE("%lli:", (long long int) *buf);
    WRITE_TRACE("%lli:", (long long int) *count);
    WRITE_TRACE("%lli:", (long long int) *datatype);
    WRITE_TRACE("%lli:", (long long int) *dest);
    WRITE_TRACE("%lli:", (long long int) *tag);
    WRITE_TRACE("%lli:", (long long int) *comm);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_start,MPI_START) (int* request, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_start,PMPI_START)(request, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Start:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_start,PMPI_START)(request, ierr);
    WRITE_TRACE("%lli:", (long long int) *request);
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

void FortranCInterface_GLOBAL(mpi_startall,MPI_STARTALL) (int* count, int* array_of_requests, int* ierr) {
  if (lap_tracing_enabled == 0) { 
    int pmpi_retval; FortranCInterface_GLOBAL(pmpi_startall,PMPI_STARTALL)(count, array_of_requests, ierr);
    return;
  }
  lap_check();
  WRITE_TRACE("%s", "MPI_Startall:");
  WRITE_TRACE("%0.2f:", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);
 FortranCInterface_GLOBAL(pmpi_startall,PMPI_STARTALL)(count, array_of_requests, ierr);
    WRITE_TRACE("%lli:", (long long int) *count);
  WRITE_TRACE("%p,%i[", (void*) array_of_requests, (int) (*count));
  if (lap_elem_tracing_enabled == 0) {  } else { 
    for (int trace_elem_idx=0; trace_elem_idx<(*count); trace_elem_idx++) {
      WRITE_TRACE("%lli;", (long long int) array_of_requests[trace_elem_idx]);
  }
  WRITE_TRACE("]%s", ":");
}
  WRITE_TRACE("%0.2f", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);  if (lap_backtrace_enabled) {
    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);
    WRITE_TRACE("  # backtrace [%s]", lap_backtrace_buf);
  }
  WRITE_TRACE("%s", "\n");
}

