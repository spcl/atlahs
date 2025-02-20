! Basic module initializing the MPI communication and handling most
! of the MPI calls (wrapper functions).
!
!
! ICON
!
! ---------------------------------------------------------------
! Copyright (C) 2004-2024, DWD, MPI-M, DKRZ, KIT, ETH, MeteoSwiss
! Contact information: icon-model.org
!
! See AUTHORS.TXT for a list of authors
! See LICENSES/ for license information
! SPDX-License-Identifier: BSD-3-Clause
! ---------------------------------------------------------------
!
!  MPI-Handshake
!  -------------
!
!  To allow the co-existence of external processes ICON proceeds the
!  MPI-Handshake (see https://gitlab.dkrz.de/dkrz-sw/mpi-handshake)
!  with the group name "icon" directly after initializing MPI.  This
!  yields a MPI_Comm where all processes are contained in that also
!  have provided this group name.

!  Furthermore ICON participates in the groups "yac" and/or "comin" if
!  these modules are enabled. The resulting communicators are then
!  used to initialize the respective software component.
!
!  In the following the further split-up of the icon communicator is described.
!
!  ICON communicator split
!  -----------------------
!
!  ICON processors are divided into
!    1.    worker PEs    : majority of MPI tasks, doing the actual work
!    2.    I/O PEs       : dedicated I/O server tasks          (only for parallel_nml::num_io_procs > 0)
!    3.    one test PE   : for verification runs               (only for parallel_nml::p_test_run == .TRUE.)
!    4.    restart PEs   : for asynchronous restart writing    (only for dedicatedRestartProcs > 0)
!    5.    prefetch PEs  : for prefetching of data             (only for parallel_nml::num_prefetch_proc > 0)
!    6.    radar I/O PEs : for some async. tasks of radar      (only for parallel_nml::num_io_procs_radar > 0)
!                          forward operator EMVORADO (asynchr. processing: I/O and postprocessing tasks like
!                          radar composites, bubble generator, superobservations)
!
!  The communicators are split like this:
!
!          0    p_work_pe0    p_io_pe0    p_restart_pe0    p_pref_pe0    p_radario_pe0   process_mpi_all_size
!
!          |         |            |             |               |                |              |
!          V         V            V             V               V                V              V
!
!          +---------------------------------------------------------------------+---------------+
!          !                      process_mpi_all_comm                                           !
!          +---------+------------+-------------+---------------+----------------+---------------+
!          | test PE | worker PEs |   I/O PEs   |  restart PEs  |  prefetch PEs  | radar I/O PEs |
!          +---------+------------+-------------+---------------+----------------+---------------+
!          |    A    |     B      |     C       |      D        |       E        |               | p_comm_work
!          |    A    |     A      |             |               |                |               | p_comm_work_test
!          |    A    |     B      |     B       |               |                | (see the      | p_comm_work_io
!          |    A    |            |     B       |               |                |    Note^^     | p_comm_io (B is worker PE 0 if num_io_procs == 0)
!          |         |     A      |             |      A        |                |      below)   | p_comm_work_restart
!          |         |     A      |             |               |       A        |               | p_comm_work_pref
!          +---------+------------+-------------+---------------+----------------+---------------+
!
!  Note that there are actually two different p_comm_work_io communicators:
!  One that spans the worker AND I/O PEs as the NAME implies, AND one that IS ONLY defined on the test PE.
!  Similarly, there IS another ghost communicator p_comm_io defined on the test PE.
!  This has the consequence that `my_process_is_io()` IS NOT equivalent to `p_comm_io /= MPI_COMM_NULL`
!
!  Process groups with specific main procs, all of these are called from mo_atmo_model:
!    * I/O (mo_name_list_output): name_list_io_main_proc()
!    * restart (mo_async_restart): restart_main_proc()
!    * prefetch (mo_async_latbc): prefetch_main_proc()
!
!
!  List of MPI communicators:
!  --------------------------
!
!       global_mpi_communicator
!
!         description  : MPI communicator spanning all PEs running  (= MPI_COMM_WORLD).
!         size         : global_mpi_size
!         this PE's ID : my_global_mpi_id (= get_my_global_mpi_id())
!
!
!       process_mpi_all_comm  (= get_my_mpi_all_communicator())
!
!         description  : MPI communicator containing all PEs that are running this
!                        model component. Different from global_mpi_communicator,
!                        if master_nml::no_of_models > 1
!         size         : process_mpi_all_size
!         this PE's ID : my_process_mpi_all_id (= get_my_mpi_all_id())
!
!
!       p_comm_work  (=get_my_mpi_work_communicator())
!
!         description  : MPI communicator for work group. On I/O and test PEs this
!                        defaults to process_mpi_all_comm.
!         size         : num_work_procs (on worker PEs: num_work_procs==p_n_work)
!         this PE's ID : p_pe_work (= get_my_mpi_work_id())
!
!       In the case of the NEC hybrid mode (detached PE0), the sub-communicator p_comm_work_only encompasses
!       all true worker PEs (excluding PE0), otherwise, p_comm_work_only is identical to p_comm_work
!
!
!       ... less important MPI communicators ...
!
!       p_comm_work_test (size = 0/1)
!         description  : MPI communicator spanning work group and test PE
!                        in verification mode parallel_nml::p_test_run == .TRUE.
!       p_comm_work_io
!         description  : MPI communicator spanning work group and I/O PEs
!       p_comm_io
!         description  : MPI communicator spanning I/O PEs
!       p_comm_work_2_io
!         description  : Inter(!)communicator work PEs - I/O PEs
!       p_comm_work_pref
!         description  : MPI Communicator spanning work group and prefetch PEs
!       p_comm_work_2_pref
!         description  : Inter(!)communicator work PEs - prefetching PEs
!
!
!  Processor splitting:
!  --------------------
!
!       In order to improve the parallel load balancing, the domains
!       of the first refinement level can be distributed to disjoint
!       processor subsets. This distribution process is weighted by
!       the namelist parameter grid_nml::patch_weight.  Since the
!       processor sets are disjoint, MPI calls happen independently on
!       each PE subset.
!
!       For global operations (sum, min, max in mo_sync), each patch
!       contains additional data members which are initialized in
!       mo_complete_subdivision::set_patch_communicators:
!
!       p_patch % comm
!         description  : MPI communicator for this patch. If no processor
!                        splitting enabled: p_patch%comm==p_comm_work.
!         size         : p_patch%n_proc
!         this PE's ID : p_patch%rank
!
!       p_patch % proc0
!         description  : global id of processor with rank 0 (within the
!                        working set p_comm_work)
!
!       The global communicator which is currently in use is stored by
!       a call to mo_mpi::push_glob_comm in the data structure
!       mo_mpi::glob_comm(0:max_lev), where the level number is equal
!       to 0 for the global grid, 1 for the first generation of child
!       domains etc.
!
!
! Split of process_mpi_all_comm and stdio process:
! ------------------------------------------------
!
!    The process_mpi_all_comm is the whole model communicator and is split
!    to test/work/io/restart, in this order
!
!    The process_mpi_stdio_id is always the 0 process of the process_mpi_all_comm
!    If is p_test, then the testing process is also the stdio process
!
!    The my_process_is_mpi_workroot() is true for the 0 process of the
!    p_comm_work communicator, this communicator does not include the test process,
!    the io and the restart processes. This communicator and and the mpi_workroot
!    process should be used in the case of gather and scatter procedures that do
!    not use the io process (the third part). This will be different from using
!    the stdio process only in the case of p_test
!
!
! ^^A Note on radar I/O PEs:
! --------------------------
!
!   If ANY(run_nml::luse_radarfwo(1:max_dom)) and parallel_nml::num_io_procs_radar > 0,
!   These PEs are appended at the very end of the PE list, but are not used by ICON itself.
!   They are provided exclusively to the radar forward operator EMVORADO for
!   Input/Output of observed and simulated radar data and postprocessing tasks
!   such as computing superobservations for data assimilation,
!   generating radar reflectivity composites for model verification and visualization,
!   or the "warm bubble generator" for triggering missing convective cells.
!   From this set of PEs, EMVORADO creates different communicators internally on its own,
!   triggered by a "CALL init_emvorado_mpi()". These communicators
!   are only used in EMVORADO and its interface (src/data_assimilation/interfaces/) and consist of:
!
!     - a clone of p_comm_work                    : icomm_cart_fwo
!     - a communicator for the radar I/O Pes      : icomm_radario
!     - a combined communicator                   : icomm_radar = icomm_cart_fwo + icomm_radario
!
!   In case EMVORADO runs asynchronously for more than 1 (N) different model domains, i.e.,
!   run_nml::luse_radarfwo(1:N) = .TRUE. and parallel_nml::num_io_procs_radar > 0,
!   and if parallel_nml::num_io_procs_radar > N,
!   there are also separate radar I/O sub-communicators for each model domain:
!
!     - a subset of icomm_radario for each domain : icomm_radario_dom(i), i=1...N
!     - a combined communicator for each domain   : icomm_radar_dom(i) = icomm_cart_fwo + icomm_radario_dom(i), i=1...N
!     - icomm_radario is still available as the superset of all icomm_radario_dom(i)
!     - icomm_radar is still available as icomm_cart_fwo + icomm_radario
!
!   This enables to run the radar I/O and postprocessing for each domain
!   separately and in parallel, minimizing communication latency on the worker
!   procs.
!
!   Although ICON does not need to access these communicators explicitly,
!   it checks at a few places if the PE does belong to icomm_radar or icomm_radario.
!   For this purpose, the following is provided PUBLIC below:
!
!   process_mpi_radario_size        : equals parallel_nml::num_io_procs_radar
!   process_mpi_all_radarioroot_id  : ID of root of icomm_radario in the global communicator (= p_radario_pe0)
!   my_process_is_radar()           : returns .TRUE. if PE belongs to icomm_radar
!   my_process_is_radario()         : returns .TRUE. if PE belongs to icomm_radario
!   my_process_is_mpi_radarioroot() : returns .TRUE. if PE is root of icomm_radario
!

!This IS a small helper to avoid a full #ifdef ... #ELSE ... #endif sequence where we
!can just replace the MPI symbol with something constant.
#ifdef NOMPI
#define MERGE_HAVE_MPI(mpiVariant, noMpiVariant) noMpiVariant
#else
#define MERGE_HAVE_MPI(mpiVariant, noMpiVariant) mpiVariant
#endif

MODULE mo_mpi

  ! Comment: Please use basic WRITE to nerr for messaging in the whole
  !          MPI package to achieve proper output.

  USE, INTRINSIC :: iso_c_binding, ONLY: c_char, c_signed_char, c_int
  
#ifndef NOMPI
  USE mpi
#endif

#ifdef _OPENMP
  USE omp_lib, ONLY: omp_get_max_threads, omp_set_num_threads
#endif

  USE mo_kind, ONLY: i4, i8, dp, sp, wp
  USE mo_io_units,       ONLY: nerr
  USE mo_impl_constants, ONLY: pio_type_async, pio_type_cdipio
  USE mtime,             ONLY: datetime, max_datetime_str_len, datetimeToString, &
    &                          newDatetime, deallocateDatetime
#ifdef HAVE_CDI_PIO
  USE mo_cdi_pio_interface, ONLY: nml_io_cdi_pio_conf_handle
#endif

#ifdef USE_NCCL
  USE nccl
  USE openacc
#endif

#ifdef __STANDALONE
  INTERFACE
    SUBROUTINE exit(iret) BIND(C,name='exit')
      IMPORT :: c_int
      INTEGER(c_int), VALUE :: iret
    END SUBROUTINE exit
  END INTERFACE
#else
  USE mo_util_system, ONLY: util_exit
#endif

  USE mo_master_control, ONLY: get_my_process_type, hamocc_process, ocean_process, process_exists, &
       &                       my_process_is_hamocc, my_process_is_ocean

  USE mo_coupling, ONLY: init_coupler, finalize_coupler
#ifdef HAVE_YAXT
  USE yaxt,                   ONLY: xt_initialize, xt_initialized
#endif
  USE mo_exception,           ONLY: init_logger

#ifndef __NO_ICON_COMIN__
  USE comin_host_interface,   ONLY: mpi_handshake,                &
    &                               comin_callback_context_call,  &
    &                               EP_FINISH, COMIN_DOMAIN_OUTSIDE_LOOP
#endif

  IMPLICIT NONE

  PRIVATE                          ! all declarations are private

#ifdef HAVE_CDI_PIO
  INCLUDE 'cdipio.inc'
#endif

  ! start/stop methods
  PUBLIC :: start_mpi, stop_mpi, abort_mpi

#ifndef NOMPI
  ! print mpi error
  PUBLIC :: handle_mpi_error
#endif

  ! split the global communicator to _process_mpi_communicator
  PUBLIC :: split_global_mpi_communicator
  !The given communicator will be the all communicator for this component
!   PUBLIC :: set_process_mpi_communicator
  ! Sets the test, work, i/o and prefetch communicators
  PUBLIC :: set_mpi_work_communicators
  ! set other parameters
  PUBLIC :: set_process_mpi_name
  ! generates a intercomm between the work PEs of two model components
  PUBLIC :: get_mpi_work_intercomm

  PUBLIC :: push_glob_comm, pop_glob_comm, get_glob_proc0

  ! Logical functions
  PUBLIC :: run_is_global_mpi_parallel
  PUBLIC :: my_process_is_stdio, my_process_is_mpi_parallel, my_process_is_mpi_all_parallel
  PUBLIC :: my_process_is_mpi_seq, my_process_is_mpi_test, my_process_is_mpi_workroot
  PUBLIC :: my_process_is_mpi_ioroot
  PUBLIC :: my_process_is_mpi_all_seq, my_process_is_io
  PUBLIC :: my_process_is_global_root
  PUBLIC :: my_process_is_mpi_prefroot, my_process_is_pref
  PUBLIC :: my_process_is_restart
  PUBLIC :: my_process_is_work, my_process_is_work_only

  ! get parameters
  PUBLIC :: get_my_global_mpi_communicator   ! essentially a copy of MPI_COMM_WORLD
  PUBLIC :: get_my_mpi_all_communicator   ! the communicator for the specific component, ie the process_mpi_all_comm
  PUBLIC :: get_my_mpi_all_comm_size   ! this is the the size of the communicator for the specific component
  PUBLIC :: get_my_mpi_work_communicator   ! the communicator for the workers of this component
  PUBLIC :: get_my_mpi_work_comm_size   ! this is the the size of the workers
  PUBLIC :: get_hamocc_ocean_mpi_communicator 

  PUBLIC :: get_mpi_all_workroot_id, get_my_global_mpi_id, get_my_mpi_all_id
  PUBLIC :: get_mpi_all_ioroot_id
  PUBLIC :: get_my_mpi_work_id, get_mpi_prefroot_id
  PUBLIC :: default_comm_type, null_comm_type


  ! some public communicators
  PUBLIC :: process_mpi_all_comm
  PUBLIC :: process_mpi_all_test_id, process_mpi_all_workroot_id, &
    &       process_mpi_all_ioroot_id, &
    &       process_mpi_all_prefroot_id, p_comm_work_pref_compute_pe0

  PUBLIC :: p_comm_work, p_comm_work_test, p_comm_work_2_test, p_comm_work_only
  PUBLIC :: p_comm_work_2_io, p_comm_work_io, p_comm_dio_io, &
    &       p_comm_io, p_comm_work_pref, p_comm_work_2_pref
  !restart communicators
  PUBLIC :: p_comm_work_2_restart, p_comm_work_restart
  PUBLIC :: p_communicator_a, p_communicator_b, p_communicator_d

#ifndef __NO_ICON_COMIN__
  PUBLIC :: p_comm_comin
#endif

  PUBLIC :: process_mpi_io_size, process_mpi_restart_size, process_mpi_pref_size

  PUBLIC :: process_mpi_all_size
  PUBLIC :: process_mpi_radario_size, process_mpi_all_radarioroot_id
  PUBLIC :: my_process_is_radar, my_process_is_radario, my_process_is_mpi_radarioroot

  PUBLIC :: process_mpi_stdio_id
  PUBLIC :: process_mpi_root_id
  PUBLIC :: process_work_io0
  PUBLIC :: process_work_pref0

  ! Main communication methods
  PUBLIC :: global_mpi_barrier
  PUBLIC :: work_mpi_barrier

  PUBLIC :: p_comm_size
  PUBLIC :: p_comm_rank
  PUBLIC :: p_send, p_recv, p_sendrecv, p_bcast, p_barrier
! PUBLIC :: p_bcast_achar
  PUBLIC :: p_get_bcast_role
  PUBLIC :: p_isend, p_irecv, p_wait, p_wait_any,         &
    &       p_irecv_packed, p_send_packed, p_recv_packed, &
    &       p_bcast_packed,                               &
    &       p_pack_int, p_pack_bool, p_pack_real,         &
    &       p_pack_int_1d, p_pack_real_1d,                &
    &       p_pack_string, p_pack_real_2d,                &
    &       p_pack_size_int, p_pack_size_bool,            &
    &       p_pack_size_real_dp, p_pack_size_string,         &
    &       p_unpack_int, p_unpack_bool, p_unpack_real,   &
    &       p_unpack_int_1d, p_unpack_real_1d,            &
    &       p_unpack_string, p_unpack_real_2d, p_test
  PUBLIC :: p_max, p_min, p_lor, p_sum, p_global_sum, p_field_sum
  PUBLIC :: p_scatter, p_gather
  PUBLIC :: p_probe
  PUBLIC :: p_gatherv, p_allgather, p_allgatherv
  PUBLIC :: p_scatterv
  PUBLIC :: p_allreduce_max
  PUBLIC :: p_reduce
  PUBLIC :: p_allreduce
  PUBLIC :: p_commit_type_struct
  PUBLIC :: p_alltoall
  PUBLIC :: p_alltoallv
  PUBLIC :: p_alltoallv_p2p
  PUBLIC :: p_clear_request
  PUBLIC :: p_mpi_wtime
  PUBLIC :: p_comm_is_intercomm
  PUBLIC :: p_comm_remote_size
  PUBLIC :: get_mpi_comm_world_ranks

  PUBLIC :: p_isEqual

  !----------- to be removed -----------------------------------------
  PUBLIC :: p_pe, p_io
  PUBLIC :: num_test_procs, num_work_procs,     &
       &    p_work_pe0, p_io_pe0,               &
       &    p_n_work, p_pe_work, &
       &    p_pref_pe0
  PUBLIC :: p_pe_work_only
  !--------------------------------------------------------------------

#ifndef NOMPI
  PUBLIC :: MPI_INTEGER, MPI_STATUS_SIZE, MPI_SUCCESS, &
            MPI_INFO_NULL, MPI_ADDRESS_KIND, &
            MPI_UNDEFINED, MPI_IN_PLACE, MPI_OP_NULL, &
            MPI_DATATYPE_NULL
  PUBLIC :: MPI_2INTEGER
#endif
  PUBLIC :: MPI_ANY_SOURCE, MPI_COMM_NULL, MPI_COMM_SELF
  PUBLIC :: MPI_REQUEST_NULL

  ! real data type matching real type of MPI implementation
  PUBLIC :: p_real_dp, p_real_sp, p_real
  PUBLIC :: p_int
  PUBLIC :: p_int_i4
  PUBLIC :: p_int_i8
  PUBLIC :: p_bool
  PUBLIC :: p_address_kind
  PUBLIC :: p_real_dp_byte, p_real_sp_byte, p_int_byte
  PUBLIC :: p_int_i4_byte, p_int_i8_byte
  PUBLIC :: p_mpi_comm_null
  
  ! mpi reduction operators
  PUBLIC :: mpi_lor, mpi_land, mpi_sum, mpi_min, mpi_max, &
       mpi_minloc, mpi_maxloc
#ifdef NOMPI
  INTEGER, PARAMETER :: mpi_lor = 1, mpi_land = 2, mpi_sum = 3, &
       mpi_min = 4, mpi_max = 5, mpi_minloc = 6, mpi_maxloc = 7
#endif

  ! NCCL-related routines. Typically no-op when NCCL is not active.
  PUBLIC :: push_gpu_comm_queue, pop_gpu_comm_queue, get_comm_acc_queue, &
            acc_wait_comms, comm_group_start, comm_group_end

  PUBLIC ::get_mpi_time,ellapsed_mpi_time

  TYPE t_work_root_process
    INTEGER :: comp_id
    INTEGER :: global_mpi_id
  END TYPE t_work_root_process

  TYPE (t_work_root_process), ALLOCATABLE :: p_work_root_processes(:)

  ! general run time information

#ifndef NOMPI
 ! MPI call inherent variables
  INTEGER :: p_error                     ! MPI error number
  INTEGER :: p_status(MPI_STATUS_SIZE)   ! standard information of MPI_RECV

  INTEGER, ALLOCATABLE, SAVE :: p_request(:) ! request values for non blocking calls
  INTEGER :: p_irequest ! the first p_irequest values of p_request are in use
  INTEGER :: p_mrequest ! actual size of p_request
  INTEGER, PARAMETER :: p_request_alloc_size = 4096
  INTEGER, PARAMETER :: p_address_kind = MPI_ADDRESS_KIND
  ! this is the global communicator
  INTEGER :: global_mpi_communicator = mpi_comm_world ! replaces MPI_COMM_WORLD
#else
  INTEGER, PARAMETER :: p_address_kind = i8    ! should not get touched at all
  INTEGER, PARAMETER :: MPI_COMM_NULL  = 0
  INTEGER, PARAMETER :: MPI_COMM_SELF  = 1
  ! dummy arguments for function calls:
  INTEGER, PARAMETER :: MPI_ANY_SOURCE = 0
  INTEGER, PARAMETER :: mpi_request_null = 0
  ! this is the global communicator
  INTEGER, PARAMETER :: global_mpi_communicator = 0  ! replaces MPI_COMM_WORLD
#endif

  INTEGER :: hamocc_ocean_mpi_communicator

  ! public parallel run information
  CHARACTER(len=64) :: global_mpi_name
  CHARACTER(len=64) :: process_mpi_name
  ! Do not change the following parameters
  INTEGER, PARAMETER :: process_mpi_stdio_id = 0
  INTEGER, PARAMETER :: process_mpi_root_id = 0
  INTEGER, PARAMETER :: default_comm_type = 1
  INTEGER, PARAMETER :: null_comm_type = 0

  ! communicator sets
  INTEGER :: global_mpi_size          ! total number of processes in global world
  INTEGER :: my_global_mpi_id         ! process id in global world
  LOGICAL :: is_global_mpi_parallel

  ! this is the communicator for the whole component (atmo/ocean/etc)
  INTEGER :: process_mpi_all_comm     ! communicator in the whole model-component
  INTEGER :: process_mpi_all_size     ! total number of processes in the whole model-component
  INTEGER :: my_process_mpi_all_id        ! the id in the whole model communicator
  INTEGER :: process_mpi_all_workroot_id  ! the root process in component
  INTEGER :: process_mpi_all_ioroot_id    ! the first I/O process
  INTEGER :: process_mpi_all_radarioroot_id    ! the first radar I/O process
  INTEGER :: process_mpi_all_test_id  ! the test process in component
  INTEGER :: process_work_io0
  INTEGER :: process_mpi_all_prefroot_id ! the id of the first prefetch process
  INTEGER :: p_comm_work_pref_compute_pe0 ! the ID for Communicator spanning work group and prefetch PEs
  INTEGER :: process_work_pref0           ! ID of prefetch PE0 within p_comm_work_pref communicator

  LOGICAL :: process_is_mpi_parallel
  LOGICAL :: process_is_stdio

  ! this is the local work communicator (computation, i/o, etc)
!   INTEGER :: process_mpi_local_comm     ! communicator in the work group
!   INTEGER :: process_mpi_local_size     ! total number of processes in the whole model-component
!   INTEGER :: my_process_mpi_local_id

  INTEGER :: my_mpi_function  ! test, work, i/o, restart_output or prefetch
  INTEGER, PARAMETER :: test_mpi_process = 1
  INTEGER, PARAMETER :: work_mpi_process = 2
  INTEGER, PARAMETER :: io_mpi_process = 3
  INTEGER, PARAMETER :: restart_mpi_process = 4
  INTEGER, PARAMETER :: pref_mpi_process = 5
  INTEGER, PARAMETER :: radario_mpi_process = 6

  !------------------------------------------------------------
  ! Processor distribution:
  ! num_test_procs:      0 or 1
  ! num_work_procs:      number of procs running in parallel on the model
  ! process_mpi_io_size: number of procs for I/O
  ! num_restart_procs:   number of procs used for writing restart files
  ! num_prefetch_proc:      number of procs used for prefetching of data
  ! num_test_procs + num_work_procs + process_mpi_io_size + num_restart_procs + num_prefetch_proc = process_mpi_all_size
  INTEGER :: num_test_procs
  INTEGER :: num_work_procs
  INTEGER :: p_radario_pe0
  INTEGER :: process_mpi_radario_size
  INTEGER :: process_mpi_io_size
  INTEGER :: process_mpi_restart_size
  INTEGER :: process_mpi_pref_size

  ! Note: p_work_pe0, p_io_pe0 are identical on all PEs

  INTEGER :: p_work_pe0    ! Number of workgroup PE 0 within all PEs
  INTEGER :: p_workonly_pe0 ! Number of workgroup PE 0 excluding detached PE0
  INTEGER :: p_io_pe0      ! Number of I/O PE 0 within all PEs (process_mpi_all_size if no I/O PEs)
  INTEGER :: p_pref_pe0    ! Number of the prefetching PE 0 within all PEs (process_mpi_all_size
  !                          if no prefetching PEs)

  ! Note: p_n_work, p_pe_work are NOT identical on all PEs

  ! p_n_work: Number of PEs working together:
  ! - num_work_procs    for non-verification runs
  ! - num_work_procs    for verification runs on pes != p_test_pe
  ! - num_test_procs    for verification runs on p_test_pe
  ! - num_io_procs      always on I/O pes
  ! - num_restart_procs on restart PEs

  INTEGER :: p_n_work
  INTEGER :: p_pe_work        ! PE number within work group
  INTEGER :: p_pe_work_only   ! PE number within work group excluding detached stdio-PE (NEC hybrid mode), otherwise same as p_pe_work


  ! MPI communicators
  INTEGER :: p_comm_work           ! Communicator for work group
  INTEGER :: p_comm_work_only      ! Communicator for work group excluding detached stdio-PE (NEC hybrid mode), otherwise same as p_comm_work
  INTEGER :: p_comm_work_test      ! Communicator spanning work group and test PE
  INTEGER :: p_comm_work_2_test
  INTEGER :: p_comm_work_io        ! Communicator spanning work group and I/O PEs
  INTEGER :: p_comm_dio_io         ! Communicator spanning detached std I/O and I/O PEs
  INTEGER :: p_comm_io             ! Communicator spanning the I/O PEs
  INTEGER :: p_comm_work_2_io      ! Inter(!)communicator work PEs - I/O PEs
  INTEGER :: p_comm_work_restart   ! Communicator spanning work group and Restart Output PEs
  INTEGER :: p_comm_work_2_restart ! Inter(!)communicator work PEs - Restart PEs
  INTEGER :: p_comm_work_pref           ! Communicator spanning work group and prefetch PEs
  INTEGER :: p_comm_work_2_pref    ! Inter(!)communicator work PEs - prefetching PEs

  INTEGER :: p_communicator_a ! for Set A
  INTEGER :: p_communicator_b ! for Set B
  INTEGER :: p_communicator_d ! for debug node

  INTEGER :: p_pe     = 0     ! this is the PE number of this task
  INTEGER :: p_io     = 0     ! PE number of PE handling IO

#ifndef __NO_ICON_COMIN__
  INTEGER :: p_comm_comin
#endif

! non blocking calls

  ! module intrinsic names

!!#ifndef NOMPI
!!LK  INTEGER :: iope                  ! PE able to do IO
!!#endif

#ifdef DEBUG
  INTEGER :: nbcast                ! counter for broadcasts for debugging
#endif

  ! MPI native transfer types

  INTEGER :: p_int     = 0
  INTEGER :: p_real    = 0
  INTEGER :: p_bool    = 0
  INTEGER :: p_char    = 0

  ! MPI transfer types

  INTEGER :: p_real_dp = 0
  INTEGER :: p_real_sp = 0
  INTEGER :: p_int_i4  = 0
  INTEGER :: p_int_i8  = 0

  ! MPI native transfer types byte size

  INTEGER :: p_int_byte     = 0
  INTEGER :: p_real_byte    = 0
!  INTEGER :: p_bool_byte    = 0
!  INTEGER :: p_char_byte    = 0

  ! MPI transfer types byte size

  INTEGER :: p_real_dp_byte = 0
  INTEGER :: p_real_sp_byte = 0
  INTEGER :: p_int_i4_byte  = 0
  INTEGER :: p_int_i8_byte  = 0

  INTEGER :: p_mpi_comm_null = -32766
  
  ! Flag if processor splitting is active
  LOGICAL, PUBLIC :: proc_split = .FALSE.
  LOGICAL, PUBLIC :: i_am_accel_node = .FALSE.

#ifdef USE_NCCL
  INTEGER, PARAMETER :: gpu_comm_queue_depth = 32
  INTEGER :: gpu_comm_queue(gpu_comm_queue_depth)
  INTEGER :: gpu_comm_queue_id = 0
  LOGICAL :: nccl_active = .FALSE. 
  TYPE(ncclComm) :: nccl_comm
  TYPE(ncclUniqueId) :: nccl_id
  PUBLIC :: push_gpu_comm_queue, pop_gpu_comm_queue
#else
  LOGICAL, PARAMETER :: nccl_active = .FALSE.
#endif


  ! communicator stack for global sums
  INTEGER, PARAMETER :: max_lev = 10 ! 2 is sufficient
  INTEGER, PUBLIC :: comm_lev = 0, glob_comm(0:max_lev), comm_proc0(0:max_lev)

  ! Storage for buffered non-blocking point-to-point communication. This is
  ! especially needed for communication between I/O servers. These tasks may
  ! be completely asynchronous and therefore ISENDs may be launched before a
  ! corresponding IRECVs has been issued.
  INTEGER :: mpi_buffer(10000)


  ! define generic interfaces to allow proper compiling with picky compilers
  ! like NAG f95 for clean argument checking and shortening the call sequence.

  INTERFACE p_send
     MODULE PROCEDURE p_send_char
     MODULE PROCEDURE p_send_real
     MODULE PROCEDURE p_send_sreal
     MODULE PROCEDURE p_send_int
     MODULE PROCEDURE p_send_bool
     MODULE PROCEDURE p_send_real_1d
     MODULE PROCEDURE p_send_sreal_1d
     MODULE PROCEDURE p_send_int_1d
     MODULE PROCEDURE p_send_bool_1d
     MODULE PROCEDURE p_send_char_1d
     MODULE PROCEDURE p_send_real_2d
     MODULE PROCEDURE p_send_int_2d
     MODULE PROCEDURE p_send_bool_2d
     MODULE PROCEDURE p_send_real_3d
     MODULE PROCEDURE p_send_int_3d
     MODULE PROCEDURE p_send_bool_3d
     MODULE PROCEDURE p_send_real_4d
     MODULE PROCEDURE p_send_int_4d
     MODULE PROCEDURE p_send_bool_4d
     MODULE PROCEDURE p_send_real_5d
  END INTERFACE

  INTERFACE p_isend
     MODULE PROCEDURE p_isend_char
     MODULE PROCEDURE p_isend_real
     MODULE PROCEDURE p_isend_sreal
     MODULE PROCEDURE p_isend_int
     MODULE PROCEDURE p_isend_bool
     MODULE PROCEDURE p_isend_char_1d
     MODULE PROCEDURE p_isend_real_1d
     MODULE PROCEDURE p_isend_sreal_1d
     MODULE PROCEDURE p_isend_int_1d
     MODULE PROCEDURE p_isend_bool_1d
     MODULE PROCEDURE p_isend_real_2d
     MODULE PROCEDURE p_isend_sreal_2d
     MODULE PROCEDURE p_isend_int_2d
     MODULE PROCEDURE p_isend_bool_2d
     MODULE PROCEDURE p_isend_real_3d
     MODULE PROCEDURE p_isend_int_3d
     MODULE PROCEDURE p_isend_bool_3d
     MODULE PROCEDURE p_isend_real_4d
     MODULE PROCEDURE p_isend_int_4d
     MODULE PROCEDURE p_isend_bool_4d
     MODULE PROCEDURE p_isend_real_5d
  END INTERFACE

  INTERFACE p_recv
     MODULE PROCEDURE p_recv_char
     MODULE PROCEDURE p_recv_real
     MODULE PROCEDURE p_recv_sreal
     MODULE PROCEDURE p_recv_int
     MODULE PROCEDURE p_recv_bool
     MODULE PROCEDURE p_recv_real_1d
     MODULE PROCEDURE p_recv_sreal_1d
     MODULE PROCEDURE p_recv_int_1d
     MODULE PROCEDURE p_recv_bool_1d
     MODULE PROCEDURE p_recv_char_1d
     MODULE PROCEDURE p_recv_real_2d
     MODULE PROCEDURE p_recv_int_2d
     MODULE PROCEDURE p_recv_bool_2d
     MODULE PROCEDURE p_recv_real_3d
     MODULE PROCEDURE p_recv_int_3d
     MODULE PROCEDURE p_recv_bool_3d
     MODULE PROCEDURE p_recv_real_4d
     MODULE PROCEDURE p_recv_int_4d
     MODULE PROCEDURE p_recv_bool_4d
     MODULE PROCEDURE p_recv_real_5d
  END INTERFACE

  INTERFACE p_irecv
     MODULE PROCEDURE p_irecv_char
     MODULE PROCEDURE p_irecv_real
     MODULE PROCEDURE p_irecv_sreal
     MODULE PROCEDURE p_irecv_int
     MODULE PROCEDURE p_irecv_bool
     MODULE PROCEDURE p_irecv_char_1d
     MODULE PROCEDURE p_irecv_real_1d
     MODULE PROCEDURE p_irecv_sreal_1d
     MODULE PROCEDURE p_irecv_int_1d
     MODULE PROCEDURE p_irecv_bool_1d
     MODULE PROCEDURE p_irecv_real_2d
     MODULE PROCEDURE p_irecv_sreal_2d
     MODULE PROCEDURE p_irecv_int_2d
     MODULE PROCEDURE p_irecv_bool_2d
     MODULE PROCEDURE p_irecv_real_3d
     MODULE PROCEDURE p_irecv_int_3d
     MODULE PROCEDURE p_irecv_bool_3d
     MODULE PROCEDURE p_irecv_real_4d
     MODULE PROCEDURE p_irecv_int_4d
     MODULE PROCEDURE p_irecv_bool_4d
  END INTERFACE p_irecv

  INTERFACE p_wait
    MODULE PROCEDURE p_wait
    MODULE PROCEDURE p_wait_1
    MODULE PROCEDURE p_wait_n
    MODULE PROCEDURE p_wait_n_stati
  END INTERFACE p_wait

  INTERFACE p_clear_request
    MODULE PROCEDURE p_clear_request
    MODULE PROCEDURE p_clear_requests
  END INTERFACE p_clear_request

  INTERFACE p_sendrecv
     MODULE PROCEDURE p_sendrecv_real_1d
     MODULE PROCEDURE p_sendrecv_real_2d
     MODULE PROCEDURE p_sendrecv_real_3d
     MODULE PROCEDURE p_sendrecv_real_4d
     MODULE PROCEDURE p_sendrecv_char_array
  END INTERFACE

  INTERFACE p_send_packed
    MODULE PROCEDURE p_send_packed
    MODULE PROCEDURE p_send_packed_2d
  END INTERFACE p_send_packed

  INTERFACE p_bcast
     MODULE PROCEDURE p_bcast_real
     MODULE PROCEDURE p_bcast_real_single
     MODULE PROCEDURE p_bcast_int_i4
     MODULE PROCEDURE p_bcast_int_i8
     MODULE PROCEDURE p_bcast_bool
     MODULE PROCEDURE p_bcast_real_1d
     MODULE PROCEDURE p_bcast_real_1d_single
     MODULE PROCEDURE p_bcast_int_1d
     MODULE PROCEDURE p_bcast_int_i8_1d
     MODULE PROCEDURE p_bcast_bool_1d
     MODULE PROCEDURE p_bcast_real_2d
     MODULE PROCEDURE p_bcast_real_2d_single
     MODULE PROCEDURE p_bcast_int_2d
     MODULE PROCEDURE p_bcast_bool_2d
     MODULE PROCEDURE p_bcast_real_3d
     MODULE PROCEDURE p_bcast_int_3d
     MODULE PROCEDURE p_bcast_bool_3d
     MODULE PROCEDURE p_bcast_real_4d
     MODULE PROCEDURE p_bcast_int_4d
     MODULE PROCEDURE p_bcast_bool_4d
     MODULE PROCEDURE p_bcast_real_5d
     MODULE PROCEDURE p_bcast_int_7d
     MODULE PROCEDURE p_bcast_real_7d
     MODULE PROCEDURE p_bcast_char
     MODULE PROCEDURE p_bcast_cchar
     MODULE PROCEDURE p_bcast_char_1d
     MODULE PROCEDURE p_bcast_datetime
     MODULE PROCEDURE p_bcast_char_2d
  END INTERFACE

  INTERFACE p_scatter
     MODULE PROCEDURE p_scatter_real_1d1d
     MODULE PROCEDURE p_scatter_real_2d1d
     MODULE PROCEDURE p_scatter_sp_1d1d
     MODULE PROCEDURE p_scatter_sp_2d1d
     MODULE PROCEDURE p_scatter_int_1d1d
     MODULE PROCEDURE p_scatter_int_2d1d
  END INTERFACE

  INTERFACE p_gather
     MODULE PROCEDURE p_gather_real_0d1d
     MODULE PROCEDURE p_gather_real_1d2d
     MODULE PROCEDURE p_gather_real_2d3d
     MODULE PROCEDURE p_gather_real_5d6d
     MODULE PROCEDURE p_gather_real_1d1d
     MODULE PROCEDURE p_gather_int_0d1d
     MODULE PROCEDURE p_gather_int_1d1d
     MODULE PROCEDURE p_gather_int_1d2d
     MODULE PROCEDURE p_gather_int_2d3d
     MODULE PROCEDURE p_gather_char_0d1d
     MODULE PROCEDURE p_gather_bool_0d1d
  END INTERFACE

  INTERFACE p_allgather
     MODULE PROCEDURE p_allgather_int_0d1d
     MODULE PROCEDURE p_allgather_int_1d2d
  END INTERFACE

  INTERFACE p_allgatherv
     MODULE PROCEDURE p_allgatherv_real_1d
     MODULE PROCEDURE p_allgatherv_int_1d
     MODULE PROCEDURE p_allgatherv_int_1d_contiguous
  END INTERFACE

  INTERFACE p_scatterv
    MODULE PROCEDURE p_scatterv_real1D2D
    MODULE PROCEDURE p_scatterv_real1D1D
    MODULE PROCEDURE p_scatterv_int_1d1d
    MODULE PROCEDURE p_scatterv_single1D1D
  END INTERFACE

  INTERFACE p_gatherv
    MODULE PROCEDURE p_gatherv_int
    MODULE PROCEDURE p_gatherv_real2D1D
    MODULE PROCEDURE p_gatherv_real3D1D
    MODULE PROCEDURE p_gatherv_int2D1D
    MODULE PROCEDURE p_gatherv_real2D2D
    MODULE PROCEDURE p_gatherv_sreal2D2D
    MODULE PROCEDURE p_gatherv_int2D2D
  END INTERFACE

  INTERFACE p_max
     MODULE PROCEDURE p_max_0d
     MODULE PROCEDURE p_max_int_0d
     MODULE PROCEDURE p_max_1d
     MODULE PROCEDURE p_max_int_1d
     MODULE PROCEDURE p_max_2d
     MODULE PROCEDURE p_max_3d
     MODULE PROCEDURE p_max_0d_sp
     MODULE PROCEDURE p_max_1d_sp
     MODULE PROCEDURE p_max_2d_sp
     MODULE PROCEDURE p_max_3d_sp
  END INTERFACE

  INTERFACE p_min
     MODULE PROCEDURE p_min_0d
     MODULE PROCEDURE p_min_int_0d
     MODULE PROCEDURE p_min_1d
     MODULE PROCEDURE p_min_int_1d
     MODULE PROCEDURE p_min_2d
     MODULE PROCEDURE p_min_3d
  END INTERFACE

  INTERFACE p_lor
    MODULE PROCEDURE p_lor_0d
  END INTERFACE p_lor

  INTERFACE p_sum
     MODULE PROCEDURE p_sum_sp_0d
     MODULE PROCEDURE p_sum_sp_1d
     MODULE PROCEDURE p_sum_dp_0d
     MODULE PROCEDURE p_sum_dp_1d
     MODULE PROCEDURE p_sum_dp_2d
     MODULE PROCEDURE p_sum_dp_3d
     MODULE PROCEDURE p_sum_i8_1d
     MODULE PROCEDURE p_sum_i_1d
     MODULE PROCEDURE p_sum_i_0d
  END INTERFACE

  INTERFACE p_global_sum
     MODULE PROCEDURE p_global_sum_1d
  END INTERFACE

  INTERFACE p_field_sum
     MODULE PROCEDURE p_field_sum_1d
     MODULE PROCEDURE p_field_sum_2d
  END INTERFACE

  !> generic interface for MPI communication calls
  INTERFACE p_allreduce_max
    MODULE PROCEDURE p_allreduce_max_int_1d
  END INTERFACE

  INTERFACE p_reduce
    MODULE PROCEDURE p_reduce_i8_0d
    MODULE PROCEDURE p_reduce_i4_0d
  END INTERFACE p_reduce

  INTERFACE p_allreduce
    MODULE PROCEDURE p_allreduce_bool_0d
    MODULE PROCEDURE p_allreduce_int4_0d
    MODULE PROCEDURE p_allreduce_int8_0d
  END INTERFACE p_allreduce

  INTERFACE p_alltoall
    MODULE PROCEDURE p_alltoall_int
  END INTERFACE

  INTERFACE p_alltoallv
    MODULE PROCEDURE p_alltoallv_int
    MODULE PROCEDURE p_alltoallv_int_i8_1d
    MODULE PROCEDURE p_alltoallv_real_2d
    MODULE PROCEDURE p_alltoallv_sreal_2d
    MODULE PROCEDURE p_alltoallv_int_2d
  END INTERFACE

  INTERFACE p_alltoallv_p2p
    MODULE PROCEDURE p_alltoallv_p2p_real_2d
    MODULE PROCEDURE p_alltoallv_p2p_int_2d
  END INTERFACE

  INTERFACE p_isEqual
    MODULE PROCEDURE p_isEqual_int
    MODULE PROCEDURE p_isEqual_charArray
  END INTERFACE

  INTERFACE p_minmax_common
    MODULE PROCEDURE p_minmax_common
    MODULE PROCEDURE p_minmax_common_sp
  END INTERFACE p_minmax_common

  CHARACTER(*), PARAMETER :: modname = 'mo_mpi'

  CHARACTER(len=256) :: message_text = ''
  

CONTAINS

#ifndef NOMPI
  SUBROUTINE handle_mpi_error(ierror, routine, line, mpi_routine)
    INTEGER, INTENT(in) :: ierror, line
    CHARACTER(*), INTENT(in) :: routine, mpi_routine

    INTEGER :: msg_len, ierror_
    CHARACTER(len=mpi_max_error_string) :: msg

    CALL mpi_error_string(ierror, msg, msg_len, ierror_)
    WRITE (nerr, '(3a,i0,a,i0,4a)') 'In routine ', routine, ' at line ', line, &
         ' error ', ierror, ' occurred when calling ', mpi_routine, &
         ': ', msg(1:msg_len)
    CALL abort_mpi
  END SUBROUTINE handle_mpi_error
#endif

  !------------------------------------------------------------------------------
  ! NCCL routines
  !------------------------------------------------------------------------------

#ifdef USE_NCCL

  SUBROUTINE push_gpu_comm_queue(queue)
    INTEGER, INTENT(IN) :: queue
    CHARACTER(len=*), PARAMETER :: routine = modname//'::push_gpu_comm_queue'

    gpu_comm_queue_id = gpu_comm_queue_id+1
    IF (gpu_comm_queue_id > gpu_comm_queue_depth) THEN
      CALL finish(routine, 'GPU stream queue for the communications is too deep')
    END IF

    gpu_comm_queue(gpu_comm_queue_id) = queue
  END SUBROUTINE

  SUBROUTINE pop_gpu_comm_queue()
    CHARACTER(len=*), PARAMETER :: routine = modname//'::pop_gpu_comm_queue'

    gpu_comm_queue_id = gpu_comm_queue_id-1
    IF (gpu_comm_queue_id <= 0) THEN
      CALL finish(routine, 'Trying to pop from the empty GPU stream queue')
    END IF
  END SUBROUTINE

  SUBROUTINE NCCL_CHECK(result)
    TYPE(ncclResult), INTENT(IN) :: result
    CHARACTER(len=*), PARAMETER :: routine = modname//'::NCCL_CHECK'

    IF (result /= ncclSuccess) THEN
      CALL finish(routine, ncclGetErrorString(result))
    END IF
  END SUBROUTINE

  FUNCTION get_comm_stream()
    INTEGER(cuda_stream_kind) :: get_comm_stream
    get_comm_stream = acc_get_cuda_stream(get_comm_acc_queue())
  END FUNCTION

  FUNCTION get_comm_acc_queue()
    INTEGER :: get_comm_acc_queue
    get_comm_acc_queue = gpu_comm_queue(gpu_comm_queue_id)
  END FUNCTION

  SUBROUTINE acc_wait_comms(queue)
    INTEGER, INTENT(IN) :: queue
    ! no-op
  END SUBROUTINE

  SUBROUTINE comm_group_start()
    nccl_active = .TRUE.
    CALL NCCL_CHECK(ncclGroupStart())
  END SUBROUTINE

  SUBROUTINE comm_group_end()
    CALL NCCL_CHECK(ncclGroupEnd())
    nccl_active = .FALSE.
  END SUBROUTINE

#else

  SUBROUTINE push_gpu_comm_queue(queue)
    INTEGER, INTENT(IN) :: queue
    ! no-op
  END SUBROUTINE

  SUBROUTINE pop_gpu_comm_queue()
    ! no-op
  END SUBROUTINE

  FUNCTION get_comm_acc_queue()
    INTEGER :: get_comm_acc_queue
    get_comm_acc_queue = 1
  END FUNCTION

  SUBROUTINE acc_wait_comms(queue)
    INTEGER, INTENT(IN) :: queue
    !$ACC WAIT(queue)
  END SUBROUTINE

  SUBROUTINE comm_group_start()
    ! no-op
  END SUBROUTINE

  SUBROUTINE comm_group_end()
    ! no-op
  END SUBROUTINE

#endif

  !------------------------------------------------------------------------------
  REAL(dp) FUNCTION get_mpi_time()
    get_mpi_time = MERGE_HAVE_MPI(MPI_Wtime(), 0.0_dp)
  END FUNCTION get_mpi_time
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  REAL(dp) FUNCTION ellapsed_mpi_time(start_time)
    REAL(dp), INTENT(in) :: start_time
    ellapsed_mpi_time = MERGE_HAVE_MPI(MPI_Wtime() - start_time, 0.0_dp)
  END FUNCTION ellapsed_mpi_time
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  SUBROUTINE set_process_mpi_name(name)
    CHARACTER(len=*), INTENT(in) ::name

    process_mpi_name = name

  END SUBROUTINE set_process_mpi_name
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  ! Rene: this has become obsolete and should be erased.
  ! It is used to idendify globally the proccess that caused an error
  INTEGER FUNCTION get_my_global_mpi_id()
    get_my_global_mpi_id = my_global_mpi_id
  END FUNCTION get_my_global_mpi_id
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_my_global_mpi_communicator()
    get_my_global_mpi_communicator = global_mpi_communicator
  END FUNCTION get_my_global_mpi_communicator
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_hamocc_ocean_mpi_communicator()
    get_hamocc_ocean_mpi_communicator = hamocc_ocean_mpi_communicator
  END FUNCTION get_hamocc_ocean_mpi_communicator
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_my_mpi_all_communicator()
    get_my_mpi_all_communicator = process_mpi_all_comm
  END FUNCTION get_my_mpi_all_communicator
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_my_mpi_all_comm_size()
    get_my_mpi_all_comm_size = process_mpi_all_size
  END FUNCTION get_my_mpi_all_comm_size
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_my_mpi_all_id()
    get_my_mpi_all_id = my_process_mpi_all_id
  END FUNCTION get_my_mpi_all_id
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_mpi_all_workroot_id()
    get_mpi_all_workroot_id = process_mpi_all_workroot_id
  END FUNCTION get_mpi_all_workroot_id
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_mpi_all_ioroot_id()
    get_mpi_all_ioroot_id = process_mpi_all_ioroot_id
  END FUNCTION get_mpi_all_ioroot_id
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_mpi_prefroot_id()
    get_mpi_prefroot_id = process_mpi_all_prefroot_id
  END FUNCTION get_mpi_prefroot_id
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_my_mpi_work_communicator()
    get_my_mpi_work_communicator = p_comm_work
  END FUNCTION get_my_mpi_work_communicator
  !------------------------------------------------------------------------------

!------------------------------------------------------------------------------
  INTEGER FUNCTION get_my_mpi_work_comm_size()
    get_my_mpi_work_comm_size = p_n_work
  END FUNCTION get_my_mpi_work_comm_size
  !------------------------------------------------------------------------------
  !------------------------------------------------------------------------------
  INTEGER FUNCTION get_my_mpi_work_id()
    get_my_mpi_work_id = p_pe_work
  END FUNCTION get_my_mpi_work_id
  !------------------------------------------------------------------------------



  !------------------------------------------------------------------------------
  LOGICAL FUNCTION my_process_is_stdio()
    my_process_is_stdio = MERGE_HAVE_MPI(process_is_stdio, .TRUE.)
  END FUNCTION my_process_is_stdio
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  LOGICAL FUNCTION my_process_is_io()
    my_process_is_io = (my_mpi_function == io_mpi_process)
  END FUNCTION my_process_is_io
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  LOGICAL FUNCTION my_process_is_radar()
    my_process_is_radar = ( my_process_is_work() .OR. my_process_is_radario() )
  END FUNCTION my_process_is_radar
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  LOGICAL FUNCTION my_process_is_radario()
    my_process_is_radario = (my_mpi_function == radario_mpi_process)
  END FUNCTION my_process_is_radario
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  LOGICAL FUNCTION my_process_is_mpi_radarioroot()
    my_process_is_mpi_radarioroot = (my_process_is_radario() .AND. &
     &                   (my_process_mpi_all_id == process_mpi_all_radarioroot_id))
  END FUNCTION my_process_is_mpi_radarioroot
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  LOGICAL FUNCTION my_process_is_mpi_ioroot()
    my_process_is_mpi_ioroot = (my_process_mpi_all_id == process_mpi_all_ioroot_id)
  END FUNCTION my_process_is_mpi_ioroot
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  LOGICAL FUNCTION my_process_is_global_root()
    my_process_is_global_root = (my_global_mpi_id == process_mpi_stdio_id) ! == 0
  END FUNCTION my_process_is_global_root
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  LOGICAL FUNCTION my_process_is_restart()
     my_process_is_restart = (my_mpi_function == restart_mpi_process)
  END FUNCTION my_process_is_restart
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  LOGICAL FUNCTION my_process_is_work()
     my_process_is_work = (my_mpi_function == work_mpi_process)
  END FUNCTION my_process_is_work
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  LOGICAL FUNCTION my_process_is_work_only()
     my_process_is_work_only = (my_process_is_work() .AND. p_pe >= p_workonly_pe0)
  END FUNCTION my_process_is_work_only
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  LOGICAL FUNCTION my_process_is_pref()
    my_process_is_pref = (my_mpi_function == pref_mpi_process)
  END FUNCTION my_process_is_pref
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  LOGICAL FUNCTION my_process_is_mpi_prefroot()
    my_process_is_mpi_prefroot = (my_process_is_pref() .AND. &
     &                   (my_process_mpi_all_id == process_mpi_all_prefroot_id))
  END FUNCTION my_process_is_mpi_prefroot
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  LOGICAL FUNCTION my_process_is_mpi_test()
    my_process_is_mpi_test = (my_mpi_function == test_mpi_process)
  END FUNCTION my_process_is_mpi_test
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  ! If is mpi parallel and not a test process
  !! Note: mpi i/o processes do not count in mpi parallel work
  !!        Only computational processes are checked for running in parallel
  LOGICAL FUNCTION my_process_is_mpi_parallel()
    my_process_is_mpi_parallel = process_is_mpi_parallel
  END FUNCTION my_process_is_mpi_parallel
  !------------------------------------------------------------------------------


  !------------------------------------------------------------------------------
  !>
  ! If is mpi all parallel
  LOGICAL FUNCTION my_process_is_mpi_all_parallel()
    my_process_is_mpi_all_parallel = (process_mpi_all_size > 1)
  END FUNCTION my_process_is_mpi_all_parallel
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  ! If is mpi all sequential
  LOGICAL FUNCTION my_process_is_mpi_all_seq()
    my_process_is_mpi_all_seq = (process_mpi_all_size <= 1)
  END FUNCTION my_process_is_mpi_all_seq
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  !! If is not mpi work parallel or this is a test process
  !! returns true
  !! Note: mpi i/o processes do not count is mpi parallel work
  !!        Only computational processes are checked for running in parallel
  LOGICAL FUNCTION my_process_is_mpi_seq()
    my_process_is_mpi_seq = .NOT. process_is_mpi_parallel
  END FUNCTION my_process_is_mpi_seq
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  LOGICAL FUNCTION my_process_is_mpi_workroot()
!     my_process_is_mpi_workroot = (my_process_mpi_all_id == process_mpi_all_workroot_id)
    my_process_is_mpi_workroot = (p_pe_work == process_mpi_root_id)
  END FUNCTION my_process_is_mpi_workroot
  !------------------------------------------------------------------------------


  !------------------------------------------------------------------------------
  !>
  ! If is not mpi parallel or is a test process
  LOGICAL FUNCTION run_is_global_mpi_parallel()
    run_is_global_mpi_parallel = is_global_mpi_parallel
  END FUNCTION run_is_global_mpi_parallel
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  SUBROUTINE print_info_stderr (name, text)
    CHARACTER (len=*), INTENT(in) :: name
    CHARACTER (len=*), INTENT(in) :: text

    IF (my_process_is_stdio() ) THEN
      WRITE (nerr,'(a,a,a,a)') " ", TRIM(name), ": ", TRIM(text)
    ENDIF

  END SUBROUTINE print_info_stderr
  !------------------------------------------------------------------------------


  !------------------------------------------------------------------------------
  SUBROUTINE finish (name, text)
    CHARACTER (len=*), INTENT(in) :: name
    CHARACTER (len=*), INTENT(in) :: text

    WRITE (nerr,'(i4.4,a,a,a,a)') my_global_mpi_id, ": ", TRIM(name), ": ", TRIM(text)
    CALL abort_mpi

  END SUBROUTINE finish
  !------------------------------------------------------------------------------


  !-------------------------------------------------------------------------
  !
  !> Pushes the communicator and proc0 onto the communicator stack.
  !! The communicator stack is needed for global sums if the processor
  !! set is split among different 1st level patches.

  SUBROUTINE push_glob_comm(comm, proc0)
    INTEGER, INTENT(IN) :: comm, proc0

    ! Safety check
    IF(comm_lev>=max_lev) &
      CALL finish('push_glob_comm','max_lev exceeded')

    comm_lev = comm_lev+1
    glob_comm(comm_lev) = comm
    comm_proc0(comm_lev) = proc0

  END SUBROUTINE push_glob_comm

  !-------------------------------------------------------------------------
  !
  !> Pops one level of the communicator stack

  SUBROUTINE pop_glob_comm()

    ! Safety check
    IF(comm_lev<=0) &
      CALL finish('pop_glob_comm','stack empty')

    comm_lev = comm_lev-1

  END SUBROUTINE pop_glob_comm


  !-------------------------------------------------------------------------
  !
  !> @return current top of communicator stack

  FUNCTION get_glob_proc0()
    INTEGER :: get_glob_proc0
    get_glob_proc0 = comm_proc0(comm_lev)
  END FUNCTION get_glob_proc0


  !------------------------------------------------------------------------------
  !-------------------------------------------------------------------------
  ! Warning: The dummy argument num_restart_procs IS NOT identical to the namelist PARAMETER anymore,
  !          rather, it IS the number of *dedicated* restart processes.
  !          We don't care about work processes here that are reused as restart writing processes.
  SUBROUTINE set_mpi_work_communicators(p_test_run, l_test_openmp, num_io_procs,               &
    &                                   num_restart_procs, my_comp_id, num_prefetch_proc,      &
    &                                   num_test_pe, pio_type,                                 &
    &                                   num_io_procs_radar, radar_flag_doms_model, num_dio_procs)

    LOGICAL, INTENT(INOUT)           :: p_test_run, l_test_openmp
    INTEGER, INTENT(INOUT)           :: num_io_procs
    INTEGER, INTENT(INOUT)           :: num_restart_procs
    INTEGER, INTENT(IN)              :: my_comp_id
    INTEGER, INTENT(IN),    OPTIONAL :: num_prefetch_proc, num_test_pe
    INTEGER, INTENT(IN),    OPTIONAL :: pio_type
    INTEGER, INTENT(INOUT), OPTIONAL :: num_io_procs_radar
    LOGICAL, INTENT(IN),    OPTIONAL :: radar_flag_doms_model(:)
    INTEGER, INTENT(IN),    OPTIONAL :: num_dio_procs

!   !local variables
    INTEGER :: my_color, remote_leader, peer_comm, p_error, global_dup_comm
    INTEGER :: tmp_common_intracom, tmp_work_intracom, my_common_intracom_mpi_id, my_work_intracom_mpi_id
    INTEGER :: hamocc_root, ocean_root
    LOGICAL :: I_am_receiver, I_am_sender
    INTEGER :: my_function_comm
    CHARACTER(*), PARAMETER :: method_name = "set_mpi_work_communicators"
    INTEGER :: grp_process_mpi_all_comm, grp_comm_work_io, input_ranks(1), &
               translated_ranks(1), grp_comm_work_pref
    INTEGER :: sizeof_prefetch_processes, pio_type_

    INTEGER :: num_radario_procs
    LOGICAL :: lhave_radar


#ifdef HAVE_CDI_PIO
    INTEGER :: my_cdi_pio_role, grib_mode_for_cdi_pio
#endif
    INTEGER :: p_restart_pe0 ! Number of Restart Output PE 0 within all PEs
                             ! (process_mpi_all_size if no restart PEs)
    INTEGER :: num_component, i
    INTEGER, ALLOCATABLE :: root_buffer(:)
    INTEGER :: comp_id
    CHARACTER(len=1000) :: message_text = ''
    CHARACTER(len=*), PARAMETER :: &
         routine = modname//'::set_mpi_work_communicators'

    INTEGER :: my_arch, p
    INTEGER, ALLOCATABLE :: glb_arch(:)
    LOGICAL :: arch_mismatch

    IF (PRESENT(num_prefetch_proc)) THEN
      sizeof_prefetch_processes = num_prefetch_proc
    ELSE
      sizeof_prefetch_processes = 0
    ENDIF

    comp_id = my_comp_id

    IF (PRESENT(pio_type)) THEN
      pio_type_ = pio_type
    ELSE
      pio_type_ = pio_type_async
    END IF

    IF (PRESENT(num_io_procs_radar) .AND. PRESENT(radar_flag_doms_model)) THEN
      lhave_radar = .TRUE.
      IF (.NOT.ANY(radar_flag_doms_model)) THEN
        CALL print_info_stderr(method_name, &
             & 'num_io_procs_radar has no effect if there are no radar-active domains!')
        CALL print_info_stderr(method_name, &
             & '--> num_io_procs_radar set to 0')
        num_io_procs_radar = 0
      END IF
    ELSE
      lhave_radar = .FALSE.
    END IF

! check l_test_openmp
#ifndef _OPENMP
    IF (l_test_openmp) THEN
      CALL print_info_stderr(method_name, &
        & 'l_test_openmp has no effect if the model is compiled without OpenMP support')
      CALL print_info_stderr(method_name, &
        & '--> l_test_openmp set to .FALSE.')
      l_test_openmp = .FALSE.
    END IF
#endif

    ! check p_test_run, num_io_procs and num_restart_procs
#ifdef NOMPI
    ! Unconditionally set p_test_run to .FALSE. and num_io_procs to 0,
    ! all other variables are already set correctly
    IF (p_test_run) THEN
      CALL print_info_stderr(method_name, &
       & 'p_test_run has no effect if the model is compiled with the NOMPI compiler directive')
      CALL print_info_stderr(method_name, &
       & '--> p_test_run set to .FALSE.')
      p_test_run = .FALSE.
    END IF
    IF (num_io_procs /= 0) THEN
      CALL print_info_stderr(method_name, &
       & 'num_io_procs has no effect if the model is compiled with the NOMPI compiler directive')
      CALL print_info_stderr(method_name, &
       & '--> num_io_procs set to 0')
      num_io_procs = 0
    END IF

    IF (lhave_radar) THEN
      IF (num_io_procs_radar /= 0) THEN
        CALL print_info_stderr(method_name, &
             & 'num_io_procs_radar has no effect if the model is compiled with the NOMPI compiler directive')
        CALL print_info_stderr(method_name, &
             & '--> num_io_procs_radar set to 0')
        num_io_procs_radar = 0
      ENDIF
    ENDIF

    IF (num_restart_procs /= 0) THEN
      CALL print_info_stderr(method_name, &
       & 'num_restart_procs has no effect if the model is compiled with the NOMPI compiler directive')
      CALL print_info_stderr(method_name, &
       & '--> num_restart_procs set to 0')
      num_restart_procs = 0
    END IF
    IF (sizeof_prefetch_processes /= 0) THEN
      CALL print_info_stderr(method_name, &
      & 'sizeof_prefetch_processes has no effect if the model is compiled with the NOMPI compiler directive')
      CALL print_info_stderr(method_name, &
      & '--> sizeof_prefetch_processes set to 0')
      sizeof_prefetch_processes = 0
    END IF


    ! set the sequential values
    p_work_pe0 = 0
    num_io_procs = 0
    sizeof_prefetch_processes = 0
    num_work_procs = 1
    p_comm_io = mpi_comm_self

    num_radario_procs = 0
    p_radario_pe0 = 0

#else

    ! A run on 1 PE is never a verification run,
    ! correct this if the user should set it differently
    IF (process_mpi_all_size < 2) THEN
      IF (p_test_run) THEN
        CALL print_info_stderr(method_name, &
            & 'p_test_run has no effect in seq run')
        CALL print_info_stderr(method_name, &
            & '--> p_test_run set to .FALSE.')
        p_test_run = .FALSE.
      ENDIF
      IF (num_io_procs > 0) THEN
        CALL print_info_stderr(method_name, &
            & 'num_io_procs cannot be > 0 in seq run')
        CALL print_info_stderr(method_name, &
            & '--> num_io_procs set to 0')
        num_io_procs = 0
      ENDIF
      IF (lhave_radar) THEN
        IF (num_io_procs_radar > 0) THEN
          CALL print_info_stderr(method_name, &
               & 'num_io_procs_radar cannot be > 0 in seq run')
          CALL print_info_stderr(method_name, &
               & '--> num_io_procs_radar set to 0')
          num_io_procs_radar = 0
        ENDIF
      ENDIF
      IF (num_restart_procs > 0) THEN
        CALL print_info_stderr(method_name, &
            & 'num_restart_procs cannot be > 0 in seq run')
        CALL print_info_stderr(method_name, &
            & '--> num_restart_procs set to 0')
        num_restart_procs = 0
      ENDIF
      IF (sizeof_prefetch_processes > 0) THEN
        CALL print_info_stderr(method_name, &
            & 'sizeof_prefetch_processes cannot be > 0 in seq run')
        CALL print_info_stderr(method_name, &
            & '--> sizeof_prefetch_processes set to 0')
        sizeof_prefetch_processes = 0
      ENDIF
    ENDIF
    IF(num_io_procs < 0) num_io_procs = 0              ! for safety only
    IF (lhave_radar) THEN
      IF (num_io_procs_radar < 0) num_io_procs_radar = 0  ! for safety only
    ENDIF
    IF(num_restart_procs < 0) num_restart_procs = 0    ! for safety only
    IF(sizeof_prefetch_processes < 0) sizeof_prefetch_processes = 0 ! for safety only
    ! -----------------------------------------
    ! Set if test
    IF(p_test_run) THEN
      IF (PRESENT(num_test_pe)) THEN
        num_test_procs = MERGE(num_test_pe, 1, num_test_pe > 1)
      ELSE
        num_test_procs = 1
      END IF
    ELSE
      num_test_procs = 0
    ENDIF
    IF (lhave_radar) THEN
      num_radario_procs = num_io_procs_radar
    ELSE
      num_radario_procs = 0
    END IF

    ! -----------------------------------------
    ! how many work processors?
    num_work_procs = process_mpi_all_size - num_test_procs - num_io_procs - num_restart_procs - sizeof_prefetch_processes &
                        - num_radario_procs

    ! Check if there are sufficient PEs at all
    IF(num_work_procs < 1) THEN
      WRITE (message_text, &
             '(a,7("  |  ",a,i0))' &
             ) 'ERROR in processor config: ', &
             & 'total number of procs: ', process_mpi_all_size, &
             & 'num_test_procs: ', num_test_procs, &
             & 'num_work_procs: ',num_work_procs, &
             & 'num_io_procs: ',num_io_procs, &
             & 'num_io_procs_radar: ',num_radario_procs, &
             & 'num_restart_procs: ',num_restart_procs, &
             & 'sizeof_prefetch_process: ',sizeof_prefetch_processes
      CALL print_info_stderr(method_name, TRIM(message_text))
      CALL finish(method_name, &
      & 'not enough processors for given values of p_test_run/num_io_procs/num_restart_procs/'// &
      &    'sizeof_prefetch_processes/num_radario_procs')
    ELSE IF (p_test_run .AND. num_work_procs == 1) THEN
#ifdef _OPENACC
      PRINT *, "Testing 1 CPU against 1 GPU"
#else
      CALL finish(method_name, &
      & 'running p_test_run with only 1 work processor does not make sense')
#endif
    ENDIF

    WRITE(message_text,'(5(a,i0))') 'Number of procs for test: ',num_test_procs, &
      & ', work: ',num_work_procs, &
      & ', I/O: ',num_io_procs, &
      & ', Restart: ',num_restart_procs, &
      & ', Prefetching: ',sizeof_prefetch_processes
    CALL print_info_stderr(method_name, message_text)

    ! Everything seems ok. Proceed to setup the communicators and ids
    ! Set up p_work_pe0, p_io_pe0, p_restart_pe0, p_pref_pe0
    ! which are identical on all PEs
    p_work_pe0    = num_test_procs
    p_io_pe0      = num_test_procs + num_work_procs
    p_restart_pe0 = num_test_procs + num_work_procs + num_io_procs
    p_pref_pe0    = num_test_procs + num_work_procs + num_io_procs + num_restart_procs
    ! preliminary, will be re-computed below to match the requirements from EMVORADO:
    p_radario_pe0 = num_test_procs + num_work_procs + num_io_procs + num_restart_procs + sizeof_prefetch_processes

    ! Print the process layout
    WRITE (message_text, *) "0 <= ", num_test_procs, " test procs < "                          , &
                          & p_work_pe0,    " <= ", num_work_procs, " work procs < "            , &
                          & p_io_pe0,      " <= ", num_io_procs, " io procs < "                , &
                          & p_restart_pe0, " <= ", num_restart_procs, " restart procs < "      , &
                          & p_pref_pe0,    " <= ", sizeof_prefetch_processes, " pref procs < " , &
                          & p_radario_pe0, " <= ", num_radario_procs, " radario procs < "     , &
                          & process_mpi_all_size
    CALL print_info_stderr(method_name, message_text)

    ! Set up p_n_work and p_pe_work which are NOT identical on all PEs
    IF(p_pe < p_work_pe0) THEN
      ! Test PE (if present)
      p_n_work  = num_test_procs ! PE in verification work group
      p_pe_work = p_pe           ! PE number within work group
    ELSE IF(p_pe < p_io_pe0) THEN
      ! Work PE
      p_n_work  = num_work_procs
      p_pe_work = p_pe - num_test_procs
    ELSE IF(p_pe < p_restart_pe0) THEN
      ! I/O PE (if present)
      p_n_work  = num_io_procs
      p_pe_work = p_pe - num_test_procs - num_work_procs
    ELSE IF(p_pe < p_pref_pe0) THEN
      ! Restart PE (if present)
      p_n_work  = num_restart_procs
      p_pe_work = p_pe - num_test_procs - num_work_procs - num_io_procs
    ELSE IF (p_pe < p_radario_pe0) THEN
      p_n_work  = sizeof_prefetch_processes
      p_pe_work = p_pe - num_test_procs - num_work_procs - num_io_procs - num_restart_procs
    ELSE
      p_n_work  = num_radario_procs
      p_pe_work = p_pe - num_test_procs - num_work_procs - num_io_procs - num_restart_procs - sizeof_prefetch_processes
    ENDIF

    ! Set communicators
    ! =================

    ! Split communicator process_mpi_all_comm between test/work/io/restart/prefetching
    ! to get  which is the communicator for
    ! usage WITHIN every group of the 5 different types
    IF(p_pe < p_work_pe0) THEN
      my_mpi_function = test_mpi_process
    ELSE IF(p_pe < p_io_pe0) THEN
      my_mpi_function = work_mpi_process
    ELSE IF(p_pe < p_restart_pe0) THEN
      my_mpi_function = io_mpi_process
    ELSE IF(p_pe < p_pref_pe0) THEN
      my_mpi_function = restart_mpi_process
    ELSE IF (p_pe < p_radario_pe0) THEN
      my_mpi_function = pref_mpi_process
    ELSE
      my_mpi_function = radario_mpi_process
    ENDIF

    ! create intra-communicators for
    ! * test ranks and work ranks (later
    !   split into p_comm_work for the respective groups)
    ! * io ranks
    ! * restart ranks and
    ! * prefetch ranks
    my_color = MERGE(work_mpi_process, my_mpi_function, &
      &                   my_mpi_function == test_mpi_process &
      &              .OR. my_mpi_function == work_mpi_process)
    CALL mpi_comm_split(process_mpi_all_comm, my_color, p_pe, &
         my_function_comm, p_error)

    IF (p_test_run .AND. my_color == work_mpi_process) THEN
      p_comm_work_test = my_function_comm
      my_color = MERGE(1, 2, my_mpi_function == test_mpi_process)
      CALL mpi_comm_split(p_comm_work_test, my_color, p_pe, &
           p_comm_work, p_error)
      my_function_comm = p_comm_work
    ELSE
      p_comm_work = my_function_comm
      ! If not a test run, p_comm_work_test must not be used at all
      p_comm_work_test = MPI_COMM_NULL
    ENDIF

    ! Apply communicator splitting in case of NEC hybrid mode
    p_comm_work_only = p_comm_work
    p_workonly_pe0   = p_work_pe0
    p_pe_work_only   = p_pe_work

    IF (PRESENT(num_dio_procs)) THEN
       IF (my_mpi_function == work_mpi_process .AND. num_dio_procs > 0) THEN
          ! set first process of p_comm_work_only
          p_workonly_pe0 = p_work_pe0 + num_dio_procs
          IF (p_pe > p_work_pe0 + num_dio_procs - 1) THEN
             ! give color to real work processes
             my_color = 1
          ELSE
             ! give color to detached stdio-PEs
             my_color = 2
          END IF
          CALL MPI_COMM_SPLIT(p_comm_work, my_color, p_pe, p_comm_work_only, p_error)
          CALL MPI_COMM_RANK (p_comm_work_only, p_pe_work_only, p_error)
       ENDIF
    ENDIF

    ! Create p_comm_work_io, the communicator spanning work group and I/O PEs
    IF (num_io_procs > 0) THEN
      my_color = MERGE(2, &
        &              MERGE(1, mpi_undefined, &
        &                    my_mpi_function == test_mpi_process), &
        &       my_mpi_function == work_mpi_process &
           .OR. my_mpi_function == io_mpi_process)
      CALL mpi_comm_split(process_mpi_all_comm, my_color, p_pe, &
           p_comm_work_io, p_error)

      ! Check whether all detached std I/O PEs and I/O PEs run on same architecture
#ifdef __NEC__
      my_arch = 1 ! NEC-SX
#else
      my_arch = 0
#endif

      IF (p_comm_work_io /= MPI_COMM_NULL .AND. PRESENT(num_dio_procs)) THEN
        ! gather all architectures on PE0
        ALLOCATE(glb_arch(num_work_procs + num_io_procs))
        CALL p_gather(my_arch, glb_arch, 0, p_comm_work_io)

        ! compare process architectures with PE0 architecture
        IF (p_pe == p_work_pe0) THEN

          arch_mismatch = .FALSE.
          ! check all detached std I/O PEs
          IF (PRESENT(num_dio_procs)) THEN
            DO p = 1, num_dio_procs
              IF (my_arch /= glb_arch(p)) arch_mismatch = .TRUE.
            END DO
          END IF
          ! check all I/O PEs
          DO p = num_work_procs + 1, num_work_procs + num_io_procs
            IF (my_arch /= glb_arch(p)) arch_mismatch = .TRUE.
          END DO
          DEALLOCATE(glb_arch)

          ! if mismatch found, abort
          IF (arch_mismatch) THEN
            CALL finish(routine, "All detached std I/O PEs and I/O PEs must run on same architecture!")
          END IF
        END IF
      END IF

      IF (pio_type_ == pio_type_async &
        & .AND. (     my_mpi_function == work_mpi_process &
        &        .OR. my_mpi_function == io_mpi_process)) THEN
        my_color = MERGE(1, mpi_undefined, my_mpi_function == io_mpi_process)
        CALL mpi_comm_split(p_comm_work_io, my_color, p_pe, p_comm_io, p_error)
      ELSE
        p_comm_io = mpi_comm_null
      END IF
      IF (pio_type_ == pio_type_cdipio &
        & .AND. (     my_mpi_function == work_mpi_process &
        &        .OR. my_mpi_function == io_mpi_process)) THEN
#ifdef HAVE_CDI_PIO
        grib_mode_for_cdi_pio = pio_mpi_fw_at_all
        nml_io_cdi_pio_conf_handle = cdiPioConfCreate()
        ! todo: cdiPioCSRLastN needs to match assignment of mpi function
        my_cdi_pio_role = cdiPioCSRLastN(p_comm_work_io, &
          &                              grib_mode_for_cdi_pio, &
          &                              num_io_procs)
        CALL cdiPioConfSetIOMode(nml_io_cdi_pio_conf_handle, &
          &                      grib_mode_for_cdi_pio)
        CALL cdiPioConfSetCSRole(nml_io_cdi_pio_conf_handle, my_cdi_pio_role)
        CALL cdiPioConfSetBatchedRMA(nml_io_cdi_pio_conf_handle, .FALSE.)
#else
        CALL finish(routine, "ICON was compiled without CDI-PIO")
#endif
      END IF
    ELSE IF (     my_mpi_function == work_mpi_process &
      &      .OR. my_mpi_function == test_mpi_process) THEN
      p_comm_work_io = my_function_comm
      p_comm_io = MERGE(mpi_comm_self, mpi_comm_null, p_pe_work == 0)
    ELSE
      p_comm_io = mpi_comm_null
      p_comm_work_io = mpi_comm_null
    END IF

    ! translate the rank "p_io_pe0" to the communicator "p_comm_work_io":
    IF (p_comm_work_io /= mpi_comm_null) THEN
      CALL MPI_Comm_group(process_mpi_all_comm, grp_process_mpi_all_comm, p_error)
      CALL MPI_Comm_group(p_comm_work_io,       grp_comm_work_io,         p_error)
      IF (num_io_procs > 0) THEN
        input_ranks(1) = p_io_pe0
      ELSE IF (p_test_run .AND. my_mpi_function == test_mpi_process) THEN
        input_ranks(1) = 0
      ELSE
        input_ranks(1) = p_work_pe0
      END IF
      CALL MPI_group_translate_ranks(grp_process_mpi_all_comm, 1, input_ranks, &
        &                            grp_comm_work_io, translated_ranks, p_error)
      process_work_io0 = translated_ranks(1)
      CALL MPI_group_free(grp_process_mpi_all_comm, p_error)
      CALL MPI_group_free(grp_comm_work_io, p_error)
    ELSE
      process_work_io0 = -1
    END IF

    ! Set p_comm_work_restart, the communicator spanning work group and Restart Ouput PEs
    my_color = MERGE(1, MPI_UNDEFINED, &
      &                   my_mpi_function == work_mpi_process &
      &              .OR. my_mpi_function == restart_mpi_process)
    CALL mpi_comm_split(process_mpi_all_comm, my_color, p_pe, &
      p_comm_work_restart, p_error)

    ! Set p_comm_work_pref, the communicator spanning work group and prefetching PEs
    IF (sizeof_prefetch_processes > 0) THEN
      my_color = MERGE(1, MPI_UNDEFINED, &
        &                   my_mpi_function == work_mpi_process &
        &              .OR. my_mpi_function == pref_mpi_process)
      CALL mpi_comm_split(process_mpi_all_comm, my_color, p_pe, &
           p_comm_work_pref, p_error)
    ELSE
      ! If no prefetching PEs are present, p_comm_inp_pref must not be used at all
      p_comm_work_pref = mpi_comm_null
    ENDIF

    ! translate the rank "p_pref_pe0" to the communicator "p_comm_work_pref":
    IF (p_comm_work_pref /= MPI_COMM_NULL) THEN
      CALL MPI_Comm_group(process_mpi_all_comm, grp_process_mpi_all_comm, p_error)
      CALL MPI_Comm_group(p_comm_work_pref,     grp_comm_work_pref,       p_error)
      IF (sizeof_prefetch_processes > 0) THEN
        input_ranks(1) = p_pref_pe0
      ELSE IF (p_test_run .AND. (p_pe < p_work_pe0)) THEN
        input_ranks(1) = 0
      ELSE
        input_ranks(1) = p_work_pe0
      END IF
      CALL MPI_group_translate_ranks(grp_process_mpi_all_comm, 1, input_ranks, &
        &                            grp_comm_work_pref, translated_ranks, p_error)
      process_work_pref0 = translated_ranks(1)
      CALL MPI_group_free(grp_process_mpi_all_comm, p_error)
      CALL MPI_group_free(grp_comm_work_pref, p_error)
    END IF

    ! Create Intercommunicator work PEs - I/O PEs

    ! From MPI-Report:
    ! Advice to users: We recommend using a dedicated peer communicator, such as a
    ! duplicate of MPI_COMM_WORLD, to avoid trouble with peer communicators.

    ! No idea what these troubles may be in reality, but let us follow the advice

    CALL MPI_Comm_dup(process_mpi_all_comm, peer_comm, p_error)

    IF (num_io_procs > 0 .AND. (     my_mpi_function == work_mpi_process &
      &                         .OR. my_mpi_function == io_mpi_process) &
      &                  .AND. pio_type_ ==  pio_type_async) THEN
      remote_leader &
        = MERGE(p_io_pe0, p_work_pe0, my_mpi_function == work_mpi_process)
      CALL mpi_intercomm_create(my_function_comm, 0, peer_comm, remote_leader, &
        & 1, p_comm_work_2_io, p_error)
    ELSE
      ! No Intercommunicator for test PE or for all, when no IO PEs are defined
      p_comm_work_2_io = MPI_COMM_NULL
    ENDIF

    IF (num_test_procs > 0 .AND. (     my_mpi_function == work_mpi_process &
      &                           .OR. my_mpi_function == test_mpi_process)) THEN
      remote_leader &
        = MERGE(0, p_work_pe0, my_mpi_function == work_mpi_process)
      CALL mpi_intercomm_create(my_function_comm, 0, peer_comm, remote_leader, &
        & 1, p_comm_work_2_test, p_error)
    ELSE
      ! No Intercommunicator when no test PEs are requested or the process is
      ! of neither work nor test function
      p_comm_work_2_test = MPI_COMM_NULL
    ENDIF

    ! Perform the same as above, but create the inter-communicators between
    ! the worker PEs and the restart PEs.
    IF (num_restart_procs > 0 &
      & .AND. (     my_mpi_function == work_mpi_process &
      &        .OR. my_mpi_function == restart_mpi_process)) THEN
      remote_leader &
        = MERGE(p_restart_pe0, p_work_pe0, my_mpi_function == work_mpi_process)
      CALL mpi_intercomm_create(my_function_comm, 0, peer_comm, remote_leader, &
        & 2, p_comm_work_2_restart, p_error)
    ELSE
      ! No Intercommunicator for Test PE or for all, when no restart PEs
      p_comm_work_2_restart = MPI_COMM_NULL
    ENDIF

    ! Perform the same as above, but create the inter-communicator between
    ! the worker PEs and the prefetching PEs.
    IF (sizeof_prefetch_processes > 0 &
      & .AND. (     my_mpi_function == work_mpi_process &
      &        .OR. my_mpi_function == pref_mpi_process)) THEN
      remote_leader &
           = MERGE(p_pref_pe0, p_work_pe0, my_mpi_function == work_mpi_process)
      CALL mpi_intercomm_create(my_function_comm, 0, peer_comm, remote_leader, &
           & 3, p_comm_work_2_pref, p_error)
    ELSE
      ! No Intercommunicator for Test PE or for all, when no prefetch PEs
      p_comm_work_2_pref = MPI_COMM_NULL
    ENDIF

    CALL mpi_comm_free(peer_comm, p_error)
    ! if OpenMP is used, the test PE uses only 1 thread in order to check
    ! the correctness of the OpenMP implementation
    ! Currently the I/O PEs are also single threaded!
#ifdef _OPENMP
    IF (l_test_openmp .AND. my_mpi_function == test_mpi_process) &
         CALL OMP_SET_NUM_THREADS(1)
    IF (p_pe >= p_io_pe0 .AND. p_pe < p_radario_pe0) CALL OMP_SET_NUM_THREADS(1)
#endif

    IF (lhave_radar) THEN
      IF (num_radario_procs > 0) THEN
        p_radario_pe0 = num_test_procs + num_work_procs + num_io_procs + num_restart_procs + sizeof_prefetch_processes
      ELSE
        p_radario_pe0 = p_work_pe0
      END IF
    END IF

#endif

    ! fill some derived variables
    process_mpi_all_workroot_id     = p_work_pe0
    process_mpi_all_ioroot_id       = p_io_pe0
    process_mpi_all_test_id         = p_work_pe0-1
    process_mpi_restart_size        = num_restart_procs
    process_mpi_io_size             = num_io_procs
    process_mpi_radario_size        = num_radario_procs
    process_mpi_all_radarioroot_id  = p_radario_pe0
    process_mpi_all_prefroot_id     = p_pref_pe0
    process_mpi_pref_size           = sizeof_prefetch_processes
    p_comm_work_pref_compute_pe0    = p_work_pe0

    ! In case of test run, only the test process is stdio
    process_is_stdio = (my_process_mpi_all_id == process_mpi_stdio_id)
    process_is_mpi_parallel = p_n_work > 1

#ifdef NOMPI
    p_work_root_processes(1)%comp_id = comp_id
    p_work_root_processes(1)%global_mpi_id = my_global_mpi_id
#else
    ! gather information of work root processes from all components
    CALL MPI_Comm_dup(global_mpi_communicator, global_dup_comm, p_error)

    num_component = SIZE(p_work_root_processes)
    ALLOCATE(root_buffer(2 * num_component))
    IF (my_global_mpi_id == 0) THEN

      IF (my_process_mpi_all_id == 0) THEN
        root_buffer(1) = comp_id
        root_buffer(2) = my_global_mpi_id
      END IF

      DO i = MERGE(1, 0, my_process_mpi_all_id == 0), num_component - 1

        CALL MPI_RECV(root_buffer(2*i+1:2*i+2), 2, p_int, MPI_ANY_SOURCE, 0, &
                      global_dup_comm, p_status, p_error)
      END DO

    ELSE IF (my_process_mpi_all_id == 0) THEN

      root_buffer(1) = comp_id
      root_buffer(2) = my_global_mpi_id
      CALL MPI_SEND(root_buffer, 2, p_int, 0, 0, global_dup_comm, p_error)

    END IF

    CALL MPI_BCAST(root_buffer, 2 * num_component, p_int, 0, global_dup_comm, &
                   p_error)
    CALL MPI_COMM_FREE(global_dup_comm, p_error)

    DO i = 1, num_component
      p_work_root_processes(i)%comp_id = root_buffer(2*i-1)
      p_work_root_processes(i)%global_mpi_id = root_buffer(2*i)
    END DO

    DEALLOCATE(root_buffer)
#endif

! if there is a hamocc proccess, create hamocc-ocean communicator
#ifndef NOMPI
      IF (process_exists(hamocc_process)) THEN
        ALLOCATE(root_buffer(2))

        IF ((get_my_process_type() == hamocc_process .or. get_my_process_type() == ocean_process) &
            .AND. my_mpi_function == work_mpi_process) THEN
          my_color = 1 
        ELSE
          my_color = 0
        ENDIF
        ! create a intracommunicator between hamocc and ocean
        CALL mpi_comm_split(global_mpi_communicator, my_color, my_global_mpi_id, &
          tmp_common_intracom, p_error)
          
        IF (my_color == 0)  THEN
          ! we will not use this communicator if not hamocc and ocean
          CALL MPI_COMM_FREE(tmp_common_intracom, p_error)
          hamocc_ocean_mpi_communicator = MPI_COMM_NULL 
        ELSE
          !------------------------------------------------
          ! we neet to create an intercommunicator 
          !   split again to the compoment communicators
          CALL MPI_COMM_RANK (tmp_common_intracom, my_common_intracom_mpi_id, p_error)
          my_color = get_my_process_type()
          CALL mpi_comm_split(tmp_common_intracom, my_color, my_common_intracom_mpi_id, &
            tmp_work_intracom, p_error)
          CALL MPI_COMM_RANK (tmp_work_intracom, my_work_intracom_mpi_id, p_error)        
        
          ! get the my_common_intracom_mpi_id of the roots to proc 0
          !   first get the hamocc root_buffer
          I_am_sender = .false.
          I_am_receiver = .false.
          IF (my_process_is_hamocc() .AND. my_work_intracom_mpi_id == 0) THEN
            root_buffer(1) = my_common_intracom_mpi_id
            hamocc_root = my_common_intracom_mpi_id
            I_am_sender = .true.
          ENDIF
          IF (my_common_intracom_mpi_id == 0) I_am_receiver = .true.
          IF (I_am_receiver .and. I_am_sender) THEN
            ! do nothing 
            I_am_receiver = .false.
            I_am_sender   = .false.
          ENDIF
          !  sent the hamocc root_buffer
          IF (I_am_sender) THEN
            CALL MPI_SEND(root_buffer, 1, p_int, 0, 0, tmp_common_intracom, p_error)
          ENDIF
          IF (I_am_receiver) THEN
            root_buffer(1) = -1        
            CALL MPI_RECV(root_buffer, 1, p_int, MPI_ANY_SOURCE, 0, &
              & tmp_common_intracom, p_status, p_error)
            hamocc_root = root_buffer(1)          
          ENDIF        
          
          ! now get the ocean root_buffer
          I_am_sender = .false.
          I_am_receiver = .false.
          IF (my_process_is_ocean() .AND. my_work_intracom_mpi_id == 0) THEN
            root_buffer(1) = my_common_intracom_mpi_id
            ocean_root = my_common_intracom_mpi_id
            I_am_sender = .true.
          ENDIF
          IF (my_common_intracom_mpi_id == 0) I_am_receiver = .true.
          IF (I_am_receiver .and. I_am_sender) THEN
            ! do nothing 
            I_am_receiver = .false.
            I_am_sender   = .false.
          ENDIF
          ! sent the hamocc root_buffer
          IF (I_am_sender) THEN
            CALL MPI_SEND(root_buffer, 1, p_int, 0, 0, tmp_common_intracom, p_error)
          ENDIF
          IF (I_am_receiver) THEN
            root_buffer(1) = -1        
            CALL MPI_RECV(root_buffer, 1, p_int, MPI_ANY_SOURCE, 0, &
              & tmp_common_intracom, p_status, p_error)
            ocean_root = root_buffer(1)          
          ENDIF        
        
          ! ok, now broadcast the ocean and hamocc roots
          IF (my_common_intracom_mpi_id == 0) THEN
            root_buffer(1) = ocean_root
            root_buffer(2) = hamocc_root          
            CALL MPI_BCAST(root_buffer, 2, p_int, 0, tmp_common_intracom, p_error)
          ELSE
            CALL MPI_BCAST(root_buffer, 2, p_int, 0, tmp_common_intracom, p_error)
            ocean_root = root_buffer(1)
            hamocc_root = root_buffer(2)
          ENDIF
          
          ! now we are ready to create the intercommunicator between haomcc and ocean
          IF (my_process_is_ocean()) THEN
            CALL MPI_Intercomm_create(tmp_work_intracom, 0, tmp_common_intracom, &
              & hamocc_root, 0, hamocc_ocean_mpi_communicator, p_error)
          ELSE
              ! this is a hamocc process
            CALL MPI_Intercomm_create(tmp_work_intracom, 0, tmp_common_intracom, &
                & ocean_root, 0, hamocc_ocean_mpi_communicator, p_error)
                                    
          ENDIF
          
          CALL MPI_COMM_FREE(tmp_work_intracom, p_error)
          CALL MPI_COMM_FREE(tmp_common_intracom, p_error)

          ! -- done -- 
        ENDIF
        
        DEALLOCATE(root_buffer)
       
      ENDIF ! hamocc process exists 
#endif
    
#ifdef HAVE_YAXT
!   initialize here yaxt for all the processors
      IF (.NOT. xt_initialized()) CALL xt_initialize(global_mpi_communicator)
#endif

    ! still to be filled
!     process_mpi_local_comm  = process_mpi_all_comm
!     process_mpi_local_size  = process_mpi_all_size
!     my_process_mpi_local_id = my_process_mpi_all_id

#ifdef USE_NCCL
      IF (my_process_is_work()) THEN
        IF (my_process_is_mpi_workroot()) THEN
          CALL NCCL_CHECK(ncclGetUniqueId(nccl_id))
        END IF

        CALL MPI_BCAST(nccl_id, int(sizeof(ncclUniqueId), 4), MPI_BYTE, &
          process_mpi_root_id, get_my_mpi_work_communicator(), p_error)

        CALL NCCL_CHECK(ncclCommInitRank(nccl_comm, get_my_mpi_work_comm_size(), &
          nccl_id, get_my_mpi_work_id()))

        CALL push_gpu_comm_queue(1)
      END IF
#endif
   

#ifdef DEBUG
      WRITE (nerr,'(a,a,i5)') method_name, ' p_pe=',            p_pe
      WRITE (nerr,'(a,a,i5)') method_name, ' p_work_pe0=',      p_work_pe0
      WRITE (nerr,'(a,a,i5)') method_name, ' p_io_pe0=',        p_io_pe0
      WRITE (nerr,'(a,a,i5)') method_name, ' p_restart_pe0=',   p_restart_pe0
      WRITE (nerr,'(a,a,i5)') method_name, ' p_radario_pe0=',   p_radario_pe0
      WRITE (nerr,'(a,a,i5)') method_name, ' p_comm_work=',     p_comm_work
      WRITE (nerr,'(a,a,i5)') method_name, ' p_comm_work_2_restart=', &
        & p_comm_work_2_restart
      WRITE (nerr,'(a,a,i5)') method_name, ' my_mpi_function=', my_mpi_function
#endif

  END SUBROUTINE set_mpi_work_communicators
  !-------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  INTEGER FUNCTION get_mpi_work_intercomm(other_comp_id)

    INTEGER, INTENT(IN) :: other_comp_id
    CHARACTER(len=*), PARAMETER :: method_name = 'get_mpi_work_intercomm'

    INTEGER :: peer_comm, num_component, other_comp_root_global_mpi_id, i, &
               intercomm

#ifdef NOMPI
    get_mpi_work_intercomm = -1
#else

    IF (my_mpi_function /= work_mpi_process) THEN
      CALL finish(method_name, ' Process type is not work_mpi_process.')
    END IF
    num_component = SIZE(p_work_root_processes)
    other_comp_root_global_mpi_id = -1

    DO i = 1, num_component
  !    write(0,*) "get_mpi_work_intercomm:", i, p_work_root_processes(i)%comp_id
      IF (p_work_root_processes(i)%comp_id == other_comp_id) THEN
        other_comp_root_global_mpi_id = p_work_root_processes(i)%global_mpi_id
      END IF
    END DO

    IF (other_comp_root_global_mpi_id == -1) THEN
      CALL finish(method_name, ' Other component not found.')
    END IF

    ! Perform the same as above, but create the intra-communicators between
    ! the worker PEs and the prefetching PEs.
    CALL MPI_COMM_DUP(global_mpi_communicator, peer_comm, p_error)
    CALL MPI_Intercomm_create(p_comm_work, 0, peer_comm, &
                              other_comp_root_global_mpi_id, 0, intercomm, &
                              p_error)
    CALL MPI_COMM_FREE(peer_comm, p_error)

    get_mpi_work_intercomm = intercomm
#endif

  END FUNCTION get_mpi_work_intercomm
  !-------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  SUBROUTINE set_default_mpi_work_variables()

    ! fill some derived variables
    process_is_stdio = (my_process_mpi_all_id == process_mpi_stdio_id)
    process_is_mpi_parallel = (process_mpi_all_size > 1)
    my_mpi_function = work_mpi_process

    process_mpi_all_test_id     = -1
    process_mpi_all_workroot_id = 0
    process_mpi_io_size         = 0
    process_mpi_restart_size    = 0
    process_mpi_pref_size       = 0
    process_mpi_radario_size    = 0
    p_comm_work_pref_compute_pe0 = 0

!     process_mpi_local_comm  = process_mpi_all_comm
!     process_mpi_local_size  = process_mpi_all_size
!     my_process_mpi_local_id = my_process_mpi_all_id

    ! set some of the old variables
    ! should be removed once the old variables are cleaned
    p_pe           = my_process_mpi_all_id
    p_io           = 0
    num_test_procs = 0
    num_work_procs = process_mpi_all_size
    p_work_pe0     = 0
    p_io_pe0       = process_mpi_all_size    ! Number of I/O PE 0 within all PEs (process_mpi_all_size if no I/O PEs)
    p_radario_pe0  = process_mpi_all_size
    ! Number of prefetching PE 0 within all PEs (process_mpi_all_size if no prefetching PEs)
    p_pref_pe0     = process_mpi_all_size
    p_n_work       = process_mpi_all_size
    p_pe_work      = my_process_mpi_all_id

    p_comm_work             = process_mpi_all_comm
    p_comm_work_io          = MPI_COMM_NULL
    p_comm_work_test        = MPI_COMM_NULL
    p_comm_work_2_io        = MPI_COMM_NULL
    p_comm_work_restart     = MPI_COMM_NULL
    p_comm_work_2_restart   = MPI_COMM_NULL
    p_comm_io               = MPI_COMM_NULL
    P_comm_work_pref        = MPI_COMM_NULL
    P_comm_work_2_pref      = MPI_COMM_NULL

    ! print some info
    IF ( .NOT. process_is_mpi_parallel) THEN
      WRITE (nerr,'(/,a,/)')  ' Single processor run.'
    ELSEIF (process_is_stdio) THEN
      WRITE (nerr,'(/,a,a,a,i0,a,/)') ' ', TRIM(process_mpi_name), &
        ' runs on ', process_mpi_all_size, ' mpi processes.'
    END IF

  END SUBROUTINE set_default_mpi_work_variables
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  !! Splits the global communicator into this component's communicator
  !! Should be called before the component configuration
  SUBROUTINE split_global_mpi_communicator(component_no, num_components)
    INTEGER, INTENT(in) :: component_no, num_components

    INTEGER :: new_communicator
    LOGICAL             :: l_mpi_is_initialised
    CHARACTER(len=*), PARAMETER :: &
         routine = modname//'::split_process_mpi_communicator'
#ifdef NOMPI
    ALLOCATE(p_work_root_processes(1))
    RETURN
#else
    !--------------------------------------------
    ! check if mpi is initialized
    CALL MPI_INITIALIZED(l_mpi_is_initialised, p_error)
    IF (p_error /= MPI_SUCCESS) THEN
      ! assume MPI is not running
      WRITE (nerr,'(a,a)') routine, ' MPI_INITITIALIZED failed.'
      WRITE (nerr,'(a,i4)') ' Error =  ', p_error
#ifdef __STANDALONE
      CALL exit(p_error)
#else
      CALL util_exit(p_error)
#endif
    END IF
    !--------------------------------------------
    ! split global_mpi_communicator
    CALL MPI_Comm_split(global_mpi_communicator, component_no, my_global_mpi_id, &
      & new_communicator, p_error)
    IF (p_error /= MPI_SUCCESS) THEN
      WRITE (message_text,'(a,i4)') ' MPI_Comm_split failed: error =  ', p_error
      CALL finish(routine, message_text)
    END IF
    CALL set_process_mpi_communicator(new_communicator)
    ALLOCATE(p_work_root_processes(num_components))
#endif

  END SUBROUTINE split_global_mpi_communicator
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  !>
  !! Set this component's communicator
  !! The new communicator id is duplicated from the input communicator
  !! Should be called before the component configuration
  SUBROUTINE set_process_mpi_communicator(new_communicator)
    INTEGER, INTENT(in) :: new_communicator

    LOGICAL             :: l_mpi_is_initialised
    CHARACTER(len=*), PARAMETER :: &
         routine = modname//'::set_process_mpi_communicator'

#ifdef NOMPI
    process_mpi_all_comm    = new_communicator
    process_mpi_all_size    = 1
    my_process_mpi_all_id   = 0

#else
    ! since here we define the process all communicator
    ! the work communicator is identical to the all communicator
    ! and the no test or i/o processes are present
    CALL MPI_INITIALIZED(l_mpi_is_initialised, p_error)

    IF (p_error /= MPI_SUCCESS) THEN
      WRITE (nerr,'(a,a)') routine, ' MPI_INITITIALIZED failed.'
      WRITE (nerr,'(a,i4)') ' Error =  ', p_error
#ifdef __STANDALONE
      CALL exit(p_error)
#else
      CALL util_exit(p_error)
#endif
    END IF

    IF ( .NOT. l_mpi_is_initialised ) THEN
       WRITE (nerr,'(a,a)') routine, &
         & ' MPI_Init or start_mpi needs to be called first.'
#ifdef __STANDALONE
      CALL exit(p_error)
#else
      CALL util_exit(p_error)
#endif
    ENDIF

    IF ( process_mpi_all_comm /= MPI_COMM_NULL) THEN
      ! free original communicator
      CALL MPI_COMM_FREE(process_mpi_all_comm, p_error)
      IF (p_error /= MPI_SUCCESS) THEN
        WRITE (nerr,'(a,a)') routine, &
          & ' MPI_COMM_FREE failed. start_mpi needs to be called before.'
        WRITE (nerr,'(a,i4)') ' Error =  ', p_error
        CALL abort_mpi
      END IF
    ENDIF
    ! assign MPI communicator generated elsewhere to process_mpi_all_comm

    CALL MPI_COMM_DUP(new_communicator, process_mpi_all_comm, p_error)
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,a)') routine,&
         & ' MPI_COMM_DUP failed for process_mpi_all_comm.'
       WRITE (nerr,'(a,i4)') ' Error =  ', p_error
       CALL abort_mpi
    END IF

    ! get local PE identification
    CALL MPI_COMM_RANK (process_mpi_all_comm, my_process_mpi_all_id, p_error)

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,a)') routine, ' MPI_COMM_RANK failed.'
       WRITE (nerr,'(a,i4)') ' Error =  ', p_error
       CALL abort_mpi
    ELSE
#ifdef DEBUG
       WRITE (nerr,'(a,a,i4,a)') routine, ' my_process_mpi_all_id ', &
         & my_process_mpi_all_id, ' started.'
#endif
    END IF


    CALL MPI_COMM_SIZE (process_mpi_all_comm, process_mpi_all_size, p_error)

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,a,i4,a)') routine, ' PE: ',&
         & my_process_mpi_all_id, ' MPI_COMM_SIZE failed.'
       WRITE (nerr,'(a,i4)') ' Error =  ', p_error
       CALL abort_mpi
    END IF


!     CALL MPI_COMM_DUP(process_mpi_all_comm,p_comm_work,p_error)
!     IF (p_error /= MPI_SUCCESS) THEN
!        WRITE (nerr,'(a,a)') routine, ' MPI_COMM_DUP failed for p_comm_work.'
!        WRITE (nerr,'(a,i4)') ' Error =  ', p_error
!        CALL abort_mpi
!     END IF
!
!     CALL MPI_COMM_DUP(process_mpi_all_comm,p_comm_work_test,p_error)
!     IF (p_error /= MPI_SUCCESS) THEN
!        WRITE (nerr,'(a,a)') routine, ' MPI_COMM_DUP failed for p_comm_work_test.'
!        WRITE (nerr,'(a,i4)') ' Error =  ', p_error
!        CALL abort_mpi
!     END IF

#endif

    CALL set_default_mpi_work_variables()

  END SUBROUTINE set_process_mpi_communicator
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  SUBROUTINE start_mpi(global_name)

!#ifdef _OPENMP
!    USE mo_util_string, ONLY: toupper
!#endif

    CHARACTER(len=*), INTENT(in), OPTIONAL :: global_name

    ! variables are required for determing I/O size in bytes of the defined
    ! KIND types for assigning the right MPI data types with the used kinds
    INTEGER     :: iig = 0
    INTEGER(i4) :: ii4 = 0_i4
    INTEGER(i8) :: ii8 = 0_i8
    REAL        :: rrg = 0.0
    REAL(sp)    :: rsp = 0.0_sp
    REAL(dp)    :: rdp = 0.0_dp
    INTEGER :: version, subversion

    CHARACTER(len=132) :: yname

    CHARACTER(len=256) :: group_names(3) = ["icon ", "yac  ", "comin"]
    INTEGER            :: group_comms(3)

    ! variables used for determing the OpenMP threads
    ! suitable as well for coupled models
#if (defined _OPENMP)
!    CHARACTER(len=32) :: env_name
!    CHARACTER(len=32) :: thread_num
!    INTEGER :: threads
    INTEGER :: global_no_of_threads
#ifndef NOMPI
    INTEGER :: provided
#endif
#ifdef __SX__
    EXTERNAL :: getenv
#endif
#endif

    CHARACTER(len=*), PARAMETER :: routine = modname//'::start_mpi'


! #ifdef _OPENMP
!     global_no_of_threads = 1
! #endif


    ! start MPI
#ifndef NOMPI
#ifdef _OPENMP
#ifdef __MULTIPLE_MPI_THREADS
    CALL MPI_INIT_THREAD(MPI_THREAD_MULTIPLE,provided,p_error)
#else
    CALL MPI_INIT_THREAD(MPI_THREAD_FUNNELED,provided,p_error)
#endif
#else
    CALL MPI_INIT (p_error)
#endif
    ! ------------------------------------
    ! Latency injection experiment
    ! ------------------------------------
    CALL MPI_PCONTROL(0)
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,a)') routine, ' MPI_INIT failed.'
       WRITE (nerr,'(a,i4)') ' Error =  ', p_error
#ifdef __STANDALONE
      CALL exit(p_error)
#else
      CALL util_exit(p_error)
#endif
    END IF

#ifdef _OPENMP
    ! Check if MPI_INIT_THREAD returned at least MPI_THREAD_FUNNELED in "provided"
#ifdef __MULTIPLE_MPI_THREADS
    IF (provided < MPI_THREAD_MULTIPLE) THEN
       WRITE (nerr,'(a,a)') routine, &
         & ' MPI_INIT_THREAD did not return desired level of thread support'
       WRITE (nerr,'(a,i0)') " provided: ", provided
       WRITE (nerr,'(a,i0)') " required: ", MPI_THREAD_MULTIPLE
       CALL MPI_Finalize(p_error)
#ifdef __STANDALONE
      CALL exit(iexit)
#else
      CALL util_exit(iexit)
#endif       
    END IF
#else
    IF (provided < MPI_THREAD_FUNNELED) THEN
       WRITE (nerr,'(a,a)') routine, &
         & ' MPI_INIT_THREAD did not return desired level of thread support'
       WRITE (nerr,'(a,i0)') " provided: ", provided
       WRITE (nerr,'(a,i0)') " required: ", MPI_THREAD_FUNNELED
     END IF
#endif
#endif

    CALL MPI_BUFFER_ATTACH(mpi_buffer, SIZE(mpi_buffer), p_error)
    IF (p_error /= 0) THEN
       WRITE (0,*) "Error in MPI_BUFFER_ATTACH."
#ifdef __STANDALONE
      CALL exit(p_error)
#else
      CALL util_exit(p_error)
#endif       
    END IF

#ifndef __NO_ICON_COMIN__
    ! ----------------------------------------------------------
    ! UNDER DEVELOPMENT (ICON ComIn)
    !
    ! MPI handshake: ICON PEs register themselves for groups
    !                `icon`, `yac`, `comin`
    ! ----------------------------------------------------------
    CALL mpi_handshake(MPI_COMM_WORLD, group_names, group_comms)
    ! COMIN TODO: use yac communicator from handshake
    global_mpi_communicator = group_comms(1)
    CALL init_coupler(opt_yac_comm = group_comms(2))
    p_comm_comin = group_comms(3)
#else
    CALL init_coupler(opt_world_comm = global_mpi_communicator)
#endif

    process_mpi_all_comm = MPI_COMM_NULL
    IF (PRESENT(global_name)) THEN
      yname = global_name
    ELSE
      yname = '(unnamed)'
    END IF
    global_mpi_name = yname


    ! get local PE identification
    CALL MPI_COMM_RANK (global_mpi_communicator, my_global_mpi_id, p_error)

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,a)') routine, ' MPI_COMM_RANK failed.'
       WRITE (nerr,'(a,i4)') ' Error =  ', p_error
       CALL abort_mpi
    END IF

    ! Information ...
    IF (my_global_mpi_id == 0) THEN
       WRITE (nerr,'(/,a,a,a)') ' ', &
            TRIM(yname), ' MPI interface runtime information:'
    END IF

#ifdef DEBUG
    WRITE (nerr,'(a,a,i4,a)') routine, ' PE ', my_global_mpi_id, ' started.'
#endif

    ! get number of available PEs
    CALL MPI_COMM_SIZE (global_mpi_communicator, global_mpi_size, p_error)

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,a,i4,a)') routine ,' PE: ', my_global_mpi_id, &
       &  ' MPI_COMM_SIZE failed.'
       WRITE (nerr,'(a,i4)') ' Error =  ', p_error
       CALL abort_mpi
    END IF

    ! for non blocking calls
    p_mrequest = p_request_alloc_size
    ALLOCATE(p_request(p_mrequest))
    p_irequest = 0

    ! lets check the available MPI version
    CALL MPI_GET_VERSION (version, subversion, p_error)

    IF (p_error /= MPI_SUCCESS) THEN
      WRITE (nerr,'(a,a)') routine , ' MPI_GET_VERSION failed.'
      WRITE (nerr,'(a,i4)') ' Error =  ', p_error
      CALL abort_mpi
    END IF

    IF (my_global_mpi_id == 0) THEN
      WRITE (nerr,'(a,a,a,i0,a1,i0)') " ", routine, &
           ' Used MPI version: ', version, '.', subversion
    END IF

    IF (my_global_mpi_id == 0) THEN
      WRITE (nerr,'(a,a,a,a,a,i0,a)') " ", routine, " ", &
        & TRIM(yname), ': Globally run on ',&
        & global_mpi_size, ' mpi processes.'
    END IF

   ! due to a possible circular dependency with mo_machine and other
    ! modules, we determine here locally the I/O size of the different
    ! kind types (assume 8 bit/byte. This is than used for determing
    ! the right MPI send/receive type parameters.
    CALL MPI_SIZEOF(iig, p_int_byte, p_error)
    CALL MPI_SIZEOF(ii4, p_int_i4_byte, p_error)
    CALL MPI_SIZEOF(ii8, p_int_i8_byte, p_error)
    CALL MPI_SIZEOF(rrg, p_real_byte, p_error)
    CALL MPI_SIZEOF(rsp, p_real_sp_byte, p_error)
    CALL MPI_SIZEOF(rdp, p_real_dp_byte, p_error)

    p_int     = MPI_INTEGER
    p_bool    = MPI_LOGICAL
    p_char    = MPI_CHARACTER

    CALL MPI_TYPE_MATCH_SIZE(MPI_TYPECLASS_REAL, p_real_sp_byte, p_real_sp, p_error)
    CALL MPI_TYPE_MATCH_SIZE(MPI_TYPECLASS_REAL, p_real_dp_byte, p_real_dp, p_error)
    CALL MPI_TYPE_MATCH_SIZE(MPI_TYPECLASS_INTEGER, p_int_i4_byte, p_int_i4, p_error)
    CALL MPI_TYPE_MATCH_SIZE(MPI_TYPECLASS_INTEGER, p_int_i8_byte, p_int_i8, p_error)
    IF (wp == dp) THEN
      p_real = p_real_dp
    ELSE IF (wp == sp) THEN
      p_real = p_real_sp
    ELSE
      p_real = mpi_datatype_null
    END IF

    p_mpi_comm_null = MPI_COMM_NULL

#ifdef DEBUG
    WRITE (nerr,'(/,a)')    ' MPI transfer sizes [bytes]:'
    WRITE (nerr,'(a,i4)') '  INTEGER generic:', p_int_byte
    WRITE (nerr,'(a,i4)') '  INTEGER 4 byte :', p_int_i4_byte
    WRITE (nerr,'(a,i4)') '  INTEGER 8 byte :', p_int_i8_byte
    WRITE (nerr,'(a,i4)') '  REAL generic   :', p_real_byte
    WRITE (nerr,'(a,i4)') '  REAL single    :', p_real_sp_byte
    WRITE (nerr,'(a,i4)') '  REAL double    :', p_real_dp_byte
#endif

! MPI ends here
#else

    WRITE (nerr,'(a,a)')  routine, ' No MPI: Single processor run.'
    ! set defaults for sequential run
    global_mpi_size  = 1        ! total number of processes in global world
    my_global_mpi_id = 0        ! process id in global world
    is_global_mpi_parallel = .false.
    process_mpi_name = 'uknown'
    process_mpi_all_comm = MPI_COMM_NULL
    p_mpi_comm_null = MPI_COMM_NULL

#endif


    is_global_mpi_parallel = global_mpi_size > 1

#ifdef _OPENMP
    ! The number of threads, if varying, will be defined via
    ! namelists
    global_no_of_threads = omp_get_max_threads()
#ifndef NOMPI
    ! Make number of threads from environment available to all model PEs
!     CALL MPI_BCAST (global_no_of_threads, 1, MPI_INTEGER, 0, global_mpi_communicator, p_error)
#endif

    IF (my_global_mpi_id == 0) THEN

      IF (is_global_mpi_parallel) THEN
        WRITE (nerr,'(a,a,a)') " ", routine, &
          & ': Running globally hybrid OpenMP-MPI mode.'
      ELSE
        WRITE (nerr,'(a,a,a)') " ", routine,': Running globally OpenMP mode.'
      ENDIF
      WRITE (nerr,'(a,a, a, i0)') " ", routine, &
        & ' global_no_of_threads is ', global_no_of_threads
    ENDIF

#endif


    ! The number of threads, if varying, will be defined via
    ! namelists
! #ifdef _OPENMP
!     ! Expect that PE 0 did got the information of OMP_NUM_THREADS.
!     ! That might be wrong in the coupled case when the model is
!     ! started via MPI dynamic process creation. So we have to check
!     ! the environment variable too.
!     IF (my_global_mpi_id == 0) THEN
!
!       IF (is_global_mpi_parallel) THEN
!         WRITE (nerr,'(/,a)') ' Running globally hybrid OpenMP-MPI mode.'
!       ELSE
!         WRITE (nerr,'(/,a)') ' Running globally OpenMP mode.'
!       ENDIF
!
!       env_name = toupper(TRIM(yname)) // '_THREADS'
! #ifdef __SX__
!       CALL getenv(TRIM(env_name), thread_num)
!
!       IF (thread_num /= ' ') THEN
! #else
!       CALL get_environment_variable(name=TRIM(env_name), value=thread_num, &
!           status=istat)
!       IF (istat == 0) THEN
! #endif
!          READ(thread_num,*) global_no_of_threads
!        ELSE
!          WRITE (nerr,'(1x,a,/)') ' Global number of OpenMP threads not given!'
!          WRITE (nerr,'(1x,a,a,a,/,1x,a,a,a)') &
!              ' Environment variable ', TRIM(env_name), ' either not set,', &
!              ' or not available to ', TRIM(yname), ' root PE.'
!          global_no_of_threads = omp_get_max_threads()
!        ENDIF
!     ENDIF ! (my_global_mpi_id == 0)
!
! #ifndef NOMPI
!     ! Make number of threads from environment available to all model PEs
!     CALL MPI_BCAST (global_no_of_threads, 1, MPI_INTEGER, 0, global_mpi_communicator, p_error)
! #endif
!     ! Inform on OpenMP thread usage
! !     CALL OMP_SET_NUM_THREADS(threads)
! !     threads = OMP_GET_MAX_THREADS()
!
!     IF (my_global_mpi_id == 0) THEN
!        WRITE (nerr,*)
!        WRITE (nerr,'(1x,a,i3)') ' global_no_of_threads is ', global_no_of_threads
!     ENDIF
! #endif


    ! by default, the global communicator is the process communicator
    CALL set_process_mpi_name(global_mpi_name)
    CALL set_process_mpi_communicator(global_mpi_communicator)

    CALL init_logger(proc_id=get_my_global_mpi_id(), &
                     l_write_output=my_process_is_stdio(), &
                     nerr_unit=nerr, &
                     l_extra_output=(proc_split .AND. comm_lev > 0 .AND. get_glob_proc0() == p_pe_work),&
                     extra_info_prefix='PROC SPLIT', &
                     callback_abort=abort_mpi)

  END SUBROUTINE start_mpi
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  SUBROUTINE stop_mpi

    INTEGER :: iexit = 0    
    ! finish MPI and clean up all PEs

    CALL finalize_coupler()

#ifndef NOMPI
    ! to prevent abort due to unfinished communication
    CALL p_barrier(process_mpi_all_comm)
    CALL MPI_PCONTROL(1)
    CALL MPI_FINALIZE (p_error)

    IF (p_error /= MPI_SUCCESS) THEN
      iexit = 1
      WRITE (nerr,'(a)') ' MPI_FINALIZE failed.'
      WRITE (nerr,'(a,i4)') ' Error = ', p_error
      CALL abort_mpi
    END IF
    process_is_mpi_parallel = .FALSE.
    DEALLOCATE(p_request)
#endif
#ifdef __STANDALONE
    CALL exit(iexit)
#else
    CALL util_exit(iexit)
#endif
  END SUBROUTINE stop_mpi
  !------------------------------------------------------------------------------

  !------------------------------------------------------------------------------
  SUBROUTINE abort_mpi

    ! this routine should be used instead of abort, util_abort() or
    ! STOP or any other exit call in all routines for proper clean up
    ! of all PEs

#ifndef __NO_ICON_COMIN__
    CALL comin_callback_context_call(EP_FINISH, COMIN_DOMAIN_OUTSIDE_LOOP)
#endif

#ifndef NOMPI
    CALL MPI_ABORT (MPI_COMM_WORLD, 0, p_error)

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a)') ' MPI_ABORT failed.'
       WRITE (nerr,'(a,i4)') ' Error =  ', p_error
     END IF
#endif
#ifndef __STANDALONE
     CALL util_exit(1)
#else
     CALL exit(1)
#endif

  END SUBROUTINE abort_mpi
  !------------------------------------------------------------------------------

  ! communicator set up
  SUBROUTINE p_set_communicator (nproca, nprocb, mapmesh, debug_parallel)

    INTEGER, INTENT(in) :: nproca, nprocb
    INTEGER, INTENT(in) :: mapmesh(0:,0:)
    INTEGER, INTENT(in) :: debug_parallel

#ifndef NOMPI
    INTEGER :: all_debug_pes(SIZE(mapmesh))

    INTEGER :: group_world, group_a, group_b, group_d
    INTEGER :: p_communicator_tmp

    INTEGER :: n, members

    INTEGER :: ranks(1) = 0

    ! first set global group

    CALL MPI_COMM_GROUP (process_mpi_all_comm, group_world, p_error)

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, &
         & ' MPI_COMM_GROUP failed.'
       WRITE (nerr,'(a,i4)') ' Error =  ', p_error
       CALL abort_mpi
    END IF

    ! communicator is process_mpi_all_comm

    IF (debug_parallel >= 0 ) THEN

       CALL MPI_GROUP_INCL (group_world, 1, ranks, group_d, p_error)

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, &
            &' MPI_GROUP_INCL failed.'
          WRITE (nerr,'(a,i4)') ' Error =  ', p_error
          CALL abort_mpi
       END IF

       CALL MPI_COMM_CREATE (process_mpi_all_comm, group_d, p_communicator_tmp, &
            p_error)

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, &
            &' MPI_COMM_CREATE failed.'
          WRITE (nerr,'(a,i4)') ' Error =  ', p_error
          CALL abort_mpi
       END IF

       IF (my_process_mpi_all_id == 0) p_communicator_d = p_communicator_tmp

       DO n = 1, SIZE(mapmesh)
          all_debug_pes(n) = n
       END DO

       CALL MPI_GROUP_INCL (group_world, SIZE(mapmesh), all_debug_pes, &
            group_d, p_error)

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, &
            & ' MPI_GROUP_INCL failed.'
          WRITE (nerr,'(a,i4)') ' Error =  ', p_error
          CALL abort_mpi
       END IF

       CALL MPI_COMM_CREATE (process_mpi_all_comm, group_d, p_communicator_tmp, &
            p_error)

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, &
            & ' MPI_COMM_CREATE failed.'
          WRITE (nerr,'(a,i4)') ' Error =  ', p_error
          CALL abort_mpi
       END IF

       IF (my_process_mpi_all_id /= 0) p_communicator_d = p_communicator_tmp

    ELSE
       p_communicator_d = process_mpi_all_comm
    END IF

    DO n = 0, nproca-1
       members = nprocb
       CALL MPI_GROUP_INCL (group_world, members, mapmesh(:,n), group_a, &
            p_error)

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, &
            & ' MPI_GROUP_INCL failed.'
          WRITE (nerr,'(a,i4)') ' Error =  ', p_error
          CALL abort_mpi
       END IF

       CALL MPI_COMM_CREATE (process_mpi_all_comm, group_a, p_communicator_tmp, &
            p_error)

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, &
            & ' MPI_COMM_CREATE failed.'
          WRITE (nerr,'(a,i4)') ' Error =  ', p_error
          CALL abort_mpi
       END IF
       IF(p_communicator_tmp/=MPI_COMM_NULL) &
         p_communicator_a = p_communicator_tmp

    END DO

    ! create groups for set Bs

    DO n = 0, nprocb-1
       members = nproca
       CALL MPI_GROUP_INCL (group_world, members, mapmesh(n,:), group_b, &
            p_error)

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, &
            & ' MPI_GROUP_INCL failed.'
          WRITE (nerr,'(a,i4)') ' Error =  ', p_error
          CALL abort_mpi
       END IF

       CALL MPI_COMM_CREATE (process_mpi_all_comm, group_b, p_communicator_tmp, &
            p_error)

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, &
            & ' MPI_COMM_CREATE failed.'
          WRITE (nerr,'(a,i4)') ' Error =  ', p_error
          CALL abort_mpi
       END IF
       IF(p_communicator_tmp/=MPI_COMM_NULL) &
         p_communicator_b = p_communicator_tmp

    END DO

    CALL MPI_BARRIER (process_mpi_all_comm, p_error)

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' PE: ', my_process_mpi_all_id, '&
         & MPI_BARRIER failed.'
       WRITE (nerr,'(a,i4)') ' Error =  ', p_error
       CALL abort_mpi
    END IF

    IF (debug_parallel >= 0 .AND. my_process_mpi_all_id == 0) THEN
      p_communicator_a = p_communicator_d
      p_communicator_b = p_communicator_d
    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,3i8)') &
         'p_set_communicator on PE ', my_process_mpi_all_id, ': ', &
         p_communicator_d, &
         p_communicator_a, &
         p_communicator_b
#endif
#endif
  END SUBROUTINE p_set_communicator

!=========================================================================

  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Comm_size()
  !---------------------------------------------------------------------------------------------------------------------------------
  FUNCTION p_comm_size(communicator)
    INTEGER :: p_comm_size
    INTEGER, INTENT(IN) :: communicator

    CHARACTER(LEN=*), PARAMETER :: routine = modname//'::p_comm_size'
    INTEGER :: ierr

#ifndef NOMPI
    CALL MPI_Comm_size(communicator, p_comm_size, ierr)
    IF(ierr /= MPI_SUCCESS) CALL finish(routine, 'Error in MPI_COMM_SIZE operation!')
#else
    p_comm_size = 1
#endif
  END FUNCTION p_comm_size

  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Comm_rank()
  !---------------------------------------------------------------------------------------------------------------------------------
  FUNCTION p_comm_rank(communicator)
    INTEGER :: p_comm_rank
    INTEGER, INTENT(IN) :: communicator

    CHARACTER(LEN=*), PARAMETER :: routine = modname//'::p_comm_rank'
    INTEGER :: ierr

#ifndef NOMPI
    CALL MPI_COMM_RANK(communicator, p_comm_rank, ierr)
    IF(ierr /= MPI_SUCCESS) CALL finish(routine, 'Error in MPI_COMM_RANK operation!')
#else
    p_comm_rank = 0
#endif
  END FUNCTION p_comm_rank


!=========================================================================

  ! send implementation
  SUBROUTINE p_send_real (t_buffer, p_destination, p_tag, p_count, comm, use_g2g)

    REAL (dp), INTENT(in) :: t_buffer
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, p_tag, &
           &        p_comm, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, p_tag, &
           &        p_comm, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         &        p_comm, p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_real


  ! send implementation

  SUBROUTINE p_send_sreal (t_buffer, p_destination, p_tag, p_count, comm, use_g2g)

    REAL (sp), INTENT(in) :: t_buffer
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_send(t_buffer, icount, p_real_sp, p_destination, p_tag, &
           &        p_comm, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_send(t_buffer, icount, p_real_sp, p_destination, p_tag, &
           &        p_comm, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_send(t_buffer, icount, p_real_sp, p_destination, p_tag, &
         &        p_comm, p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_sreal


  SUBROUTINE p_send_real_1d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (dp), INTENT(in) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF


    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF

    CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         &        p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_real_1d

  SUBROUTINE p_send_sreal_1d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (sp), INTENT(in) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_real_sp, p_destination, p_tag, &
         &        p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_sreal_1d

  SUBROUTINE p_send_real_2d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (dp), INTENT(in) :: t_buffer(:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_real_2d

  SUBROUTINE p_send_real_3d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (dp), INTENT(in) :: t_buffer(:,:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_real_3d

  SUBROUTINE p_send_real_4d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (dp), INTENT(in) :: t_buffer(:,:,:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_real_4d

  SUBROUTINE p_send_real_5d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (dp), INTENT(in) :: t_buffer(:,:,:,:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_real_5d

  SUBROUTINE p_send_int (t_buffer, p_destination, p_tag, p_count, comm, use_g2g)

    INTEGER, INTENT(in) :: t_buffer
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_send(t_buffer, icount, p_int, p_destination, p_tag, &
           &        p_comm, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_send(t_buffer, icount, p_int, p_destination, p_tag, &
           &        p_comm, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_send(t_buffer, icount, p_int, p_destination, p_tag, &
         &        p_comm, p_error)
    !$ACC END HOST_DATA
#endif


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_int

  SUBROUTINE p_send_int_1d (t_buffer, p_destination, p_tag, p_count, comm)

    INTEGER, INTENT(in) :: t_buffer(:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_int, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_int_1d

  SUBROUTINE p_send_int_2d (t_buffer, p_destination, p_tag, p_count, comm)

    INTEGER, INTENT(in) :: t_buffer(:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_int, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_int_2d

  SUBROUTINE p_send_int_3d (t_buffer, p_destination, p_tag, p_count, comm)

    INTEGER, INTENT(in) :: t_buffer(:,:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_int, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_int_3d

  SUBROUTINE p_send_int_4d (t_buffer, p_destination, p_tag, p_count, comm)

    INTEGER, INTENT(in) :: t_buffer(:,:,:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_int, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_int_4d


  SUBROUTINE p_send_bool (t_buffer, p_destination, p_tag, p_count, comm, use_g2g)

    LOGICAL, INTENT(in) :: t_buffer
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_send(t_buffer, icount, p_bool, p_destination, p_tag, &
           &        p_comm, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_send(t_buffer, icount, p_bool, p_destination, p_tag, &
           &        p_comm, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_send(t_buffer, icount, p_bool, p_destination, p_tag, &
         p_comm, p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_bool

  SUBROUTINE p_send_bool_1d (t_buffer, p_destination, p_tag, p_count, comm)

    LOGICAL, INTENT(in) :: t_buffer(:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_bool, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_bool_1d


  SUBROUTINE p_send_char_1d (t_buffer, p_destination, p_tag, p_count, comm)

    CHARACTER(LEN=*), INTENT(in) :: t_buffer(:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF
    icount = icount * LEN(t_buffer(1))



    CALL mpi_send(t_buffer, icount, p_char, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_char_1d


  SUBROUTINE p_send_bool_2d (t_buffer, p_destination, p_tag, p_count, comm)

    LOGICAL, INTENT(in) :: t_buffer(:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_bool, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_bool_2d

  SUBROUTINE p_send_bool_3d (t_buffer, p_destination, p_tag, p_count, comm)

    LOGICAL, INTENT(in) :: t_buffer(:,:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_bool, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_bool_3d

  SUBROUTINE p_send_bool_4d (t_buffer, p_destination, p_tag, p_count, comm)

    LOGICAL, INTENT(in) :: t_buffer(:,:,:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_send(t_buffer, icount, p_bool, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_bool_4d

  SUBROUTINE p_send_char (t_buffer, p_destination, p_tag, p_count, comm)

    CHARACTER(len=*),  INTENT(in) :: t_buffer
    INTEGER,           INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = LEN(t_buffer)
    END IF



    CALL mpi_send(t_buffer, icount, p_char, p_destination, p_tag, &
         p_comm, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_char

! non-blocking sends

  SUBROUTINE p_inc_request
    INTEGER, ALLOCATABLE :: tmp(:)

#ifndef NOMPI
    p_irequest = p_irequest + 1
    IF (p_irequest > p_mrequest) THEN
      ALLOCATE(tmp(p_mrequest))
      tmp(:) = p_request(:)
      DEALLOCATE(p_request)
      ALLOCATE(p_request(p_mrequest+p_request_alloc_size))
      p_request(1:p_mrequest) = tmp(:)
      p_mrequest = p_mrequest+p_request_alloc_size
      DEALLOCATE(tmp)
    ENDIF
#endif

  END SUBROUTINE p_inc_request

#ifdef USE_NCCL
  SUBROUTINE p_isend_nccl_real (t_buffer, p_destination, p_count)
    REAL (dp), INTENT(inout) :: t_buffer
    INTEGER,   INTENT(in) :: p_destination
    INTEGER,   INTENT(in) :: p_count

    !print *, "Send from ", my_global_mpi_id, " to ", p_destination, " size ", p_count

    CALL NCCL_CHECK(ncclSend( acc_deviceptr(t_buffer), int(p_count, cuda_count_kind), ncclFloat64, &
                              p_destination, nccl_comm, int(get_comm_stream(), cuda_stream_kind) ))
  END SUBROUTINE
#endif

  SUBROUTINE p_isend_real (t_buffer, p_destination, p_tag, p_count, comm, request, use_g2g)

    REAL (dp), INTENT(inout) :: t_buffer
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    INTEGER,   INTENT(INOUT), OPTIONAL :: request
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount, out_request

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

    IF (nccl_active .and. my_process_is_work() .and. comm == p_comm_work .and. loc_use_g2g) THEN
#ifdef USE_NCCL
      CALL p_isend_nccl_real(t_buffer, p_destination, p_count)
#endif
    ELSE
! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
      IF (loc_use_g2g) THEN
        !$ACC HOST_DATA USE_DEVICE(t_buffer)
        CALL mpi_isend(t_buffer, icount, p_real_dp, p_destination, p_tag, &
             &         p_comm, out_request, p_error)
        !$ACC END HOST_DATA
      ELSE
        CALL mpi_isend(t_buffer, icount, p_real_dp, p_destination, p_tag, &
             &         p_comm, out_request, p_error)
      END IF
#else
      !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
      CALL mpi_isend(t_buffer, icount, p_real_dp, p_destination, p_tag, &
           &         p_comm, out_request, p_error)
      !$ACC END HOST_DATA
#endif
                
      IF (PRESENT(request)) THEN
        request               = out_request
      ELSE
        CALL p_inc_request
        p_request(p_irequest) = out_request
      END IF
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_real


  SUBROUTINE p_isend_sreal (t_buffer, p_destination, p_tag, p_count, comm, request, use_g2g)

    REAL (sp), INTENT(inout) :: t_buffer
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    INTEGER,   INTENT(INOUT), OPTIONAL :: request
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, out_request, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_isend(t_buffer, icount, p_real_sp, p_destination, p_tag, &
           &         p_comm, out_request, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_isend(t_buffer, icount, p_real_sp, p_destination, p_tag, &
           &         p_comm, out_request, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_isend(t_buffer, icount, p_real_sp, p_destination, p_tag, &
         &         p_comm, out_request, p_error)
    !$ACC END HOST_DATA
#endif

    IF (PRESENT(request)) THEN
      request               = out_request
    ELSE
      CALL p_inc_request
      p_request(p_irequest) = out_request
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_sreal


  SUBROUTINE p_isend_real_1d (t_buffer, p_destination, p_tag, p_count, comm, request)

    REAL (dp), INTENT(inout) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    INTEGER,   INTENT(INOUT), OPTIONAL :: request
#ifndef NOMPI
    INTEGER :: p_comm, icount, out_request

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_isend(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         &         p_comm, out_request, p_error)


    IF (PRESENT(request)) THEN
      request               = out_request
    ELSE
      CALL p_inc_request
      p_request(p_irequest) = out_request
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_real_1d


  SUBROUTINE p_isend_sreal_1d (t_buffer, p_destination, p_tag, p_count, comm, request)

    REAL (sp), INTENT(inout) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    INTEGER,   INTENT(INOUT), OPTIONAL :: request
#ifndef NOMPI
    INTEGER :: p_comm, icount, out_request

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_isend(t_buffer, icount, p_real_sp, p_destination, p_tag, &
      &            p_comm, out_request, p_error)


    IF (PRESENT(request)) THEN
      request               = out_request
    ELSE
      CALL p_inc_request
      p_request(p_irequest) = out_request
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_sreal_1d


  SUBROUTINE p_isend_real_2d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (dp), INTENT(inout) :: t_buffer(:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_real_dp, p_destination, p_tag, &
            p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_real_2d


  SUBROUTINE p_isend_sreal_2d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (sp), INTENT(inout) :: t_buffer(:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_real_sp, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_sreal_2d


  SUBROUTINE p_isend_real_3d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (dp), INTENT(inout) :: t_buffer(:,:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_real_3d

  SUBROUTINE p_isend_real_4d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (dp), INTENT(inout) :: t_buffer(:,:,:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_real_4d

  SUBROUTINE p_isend_real_5d (t_buffer, p_destination, p_tag, p_count, comm)

    REAL (dp), INTENT(inout) :: t_buffer(:,:,:,:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_real_dp, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_real_5d

  SUBROUTINE p_isend_int(t_buffer, p_destination, p_tag, p_count, comm, request, use_g2g)

    INTEGER, INTENT(in) :: t_buffer
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    INTEGER,   INTENT(INOUT), OPTIONAL :: request
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g

#ifndef NOMPI
    INTEGER :: p_comm, out_request, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_isend(t_buffer, icount, p_int, p_destination, p_tag, &
           &         p_comm, out_request, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_isend(t_buffer, icount, p_int, p_destination, p_tag, &
           &         p_comm, out_request, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_isend(t_buffer, icount, p_int, p_destination, p_tag, &
         &         p_comm, out_request, p_error)
    !$ACC END HOST_DATA
#endif

    IF (PRESENT(request)) THEN
      request               = out_request
    ELSE
      CALL p_inc_request
      p_request(p_irequest) = out_request
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_int

  SUBROUTINE p_isend_int_1d(t_buffer, p_destination, p_tag, p_count, comm, request)

    INTEGER, INTENT(inout) :: t_buffer(:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    INTEGER,   INTENT(INOUT), OPTIONAL :: request

#ifndef NOMPI
    INTEGER :: p_comm, out_request, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_isend(t_buffer, icount, p_int, p_destination, p_tag, &
         &         p_comm, out_request, p_error)


    IF (PRESENT(request)) THEN
      request               = out_request
    ELSE
      CALL p_inc_request
      p_request(p_irequest) = out_request
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_int_1d

  SUBROUTINE p_isend_int_2d (t_buffer, p_destination, p_tag, p_count, comm)

    INTEGER, INTENT(inout) :: t_buffer(:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_int, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_int_2d

  SUBROUTINE p_isend_int_3d (t_buffer, p_destination, p_tag, p_count, comm)

    INTEGER, INTENT(inout) :: t_buffer(:,:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_int, p_destination, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_int_3d

  SUBROUTINE p_isend_int_4d (t_buffer, p_destination, p_tag, p_count, comm)

    INTEGER, INTENT(inout) :: t_buffer(:,:,:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_int, p_destination, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_int_4d


  SUBROUTINE p_isend_bool (t_buffer, p_destination, p_tag, p_count, comm, use_g2g)

    LOGICAL, INTENT(inout) :: t_buffer
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_isend(t_buffer, icount, p_bool, p_destination, p_tag, &
           &         p_comm, p_request(p_irequest), p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_isend(t_buffer, icount, p_bool, p_destination, p_tag, &
           &         p_comm, p_request(p_irequest), p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_bool, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_bool

  SUBROUTINE p_isend_bool_1d (t_buffer, p_destination, p_tag, p_count, comm)

    LOGICAL, INTENT(inout) :: t_buffer(:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_bool, p_destination, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_bool_1d

  SUBROUTINE p_isend_bool_2d (t_buffer, p_destination, p_tag, p_count, comm)

    LOGICAL, INTENT(inout) :: t_buffer(:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_bool, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_bool_2d

  SUBROUTINE p_isend_bool_3d (t_buffer, p_destination, p_tag, p_count, comm)

    LOGICAL, INTENT(inout) :: t_buffer(:,:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_bool, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_bool_3d

  SUBROUTINE p_isend_bool_4d (t_buffer, p_destination, p_tag, p_count, comm)

    LOGICAL, INTENT(inout) :: t_buffer(:,:,:,:)
    INTEGER, INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_bool, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_bool_4d

  SUBROUTINE p_isend_char (t_buffer, p_destination, p_tag, p_count, comm)

    CHARACTER(len=*),  INTENT(inout) :: t_buffer
    INTEGER,           INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = LEN(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_char, p_destination, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_isend_char

  ! recv implementation

  SUBROUTINE p_recv_real (t_buffer, p_source, p_tag, p_count, comm, use_g2g)

    REAL (dp), INTENT(out) :: t_buffer
    INTEGER,   INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_recv(t_buffer, icount, p_real_dp, p_source, p_tag, &
           &        p_comm, p_status, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_recv(t_buffer, icount, p_real_dp, p_source, p_tag, &
           &        p_comm, p_status, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_recv(t_buffer, icount, p_real_dp, p_source, p_tag, &
         &        p_comm, p_status, p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_real

  ! recv implementation

  SUBROUTINE p_recv_sreal (t_buffer, p_source, p_tag, p_count, comm, use_g2g)

    REAL (sp), INTENT(out) :: t_buffer
    INTEGER,   INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_recv(t_buffer, icount, p_real_sp, p_source, p_tag, &
           &        p_comm, p_status, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_recv(t_buffer, icount, p_real_sp, p_source, p_tag, &
           &        p_comm, p_status, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_recv(t_buffer, icount, p_real_sp, p_source, p_tag, &
         &        p_comm, p_status, p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_sreal

  SUBROUTINE p_isend_char_1d(t_buffer, p_destination, p_tag, p_count, comm)
    CHARACTER(len=*),  INTENT(inout) :: t_buffer(:)
    INTEGER,           INTENT(in) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF
    IF (icount > 0) icount = icount * LEN(t_buffer(1))

    CALL p_inc_request
    CALL mpi_isend(t_buffer, icount, p_char, p_destination, p_tag, &
         p_comm, p_request(p_irequest), p_error)

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif
  END SUBROUTINE p_isend_char_1d

  SUBROUTINE p_recv_real_1d (t_buffer, p_source, p_tag, p_count, comm, displs)

    REAL (dp), INTENT(out) :: t_buffer(:)
    INTEGER,   INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm, displs
#ifndef NOMPI
    INTEGER :: p_comm, icount, idispls

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    icount = SIZE(t_buffer)
    IF (PRESENT(p_count))  icount = p_count

    idispls = 1
    IF (PRESENT(displs)) idispls = displs


    CALL mpi_recv(t_buffer(idispls), icount, p_real_dp, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_real_1d

  SUBROUTINE p_recv_sreal_1d (t_buffer, p_source, p_tag, p_count, comm, displs)

    REAL (sp), INTENT(out) :: t_buffer(:)
    INTEGER,   INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm, displs
#ifndef NOMPI
    INTEGER :: p_comm, icount, idispls

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF
    IF (PRESENT(displs)) THEN
      idispls = displs
    ELSE
      idispls = 1
    END IF


    CALL mpi_recv(t_buffer(idispls), icount, p_real_sp, p_source, p_tag, &
         &        p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_sreal_1d

  SUBROUTINE p_recv_real_2d (t_buffer, p_source, p_tag, p_count, comm)

    REAL (dp), INTENT(out) :: t_buffer(:,:)
    INTEGER,   INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_real_dp, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_real_2d

  SUBROUTINE p_recv_real_3d (t_buffer, p_source, p_tag, p_count, comm)

    REAL (dp), INTENT(out) :: t_buffer(:,:,:)
    INTEGER,   INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_real_dp, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_real_3d

  SUBROUTINE p_recv_real_4d (t_buffer, p_source, p_tag, p_count, comm)

    REAL (dp), INTENT(out) :: t_buffer(:,:,:,:)
    INTEGER,   INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_real_dp, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_real_4d

  SUBROUTINE p_recv_real_5d (t_buffer, p_source, p_tag, p_count, comm)

    REAL (dp), INTENT(out) :: t_buffer(:,:,:,:,:)
    INTEGER,   INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_real_dp, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_real_5d

  SUBROUTINE p_recv_int (t_buffer, p_source, p_tag, p_count, comm, use_g2g)

    INTEGER, INTENT(out) :: t_buffer
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_recv(t_buffer, icount, p_int, p_source, p_tag, &
           &        p_comm, p_status, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_recv(t_buffer, icount, p_int, p_source, p_tag, &
           &        p_comm, p_status, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_recv(t_buffer, icount, p_int, p_source, p_tag, &
         &        p_comm, p_status, p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_int

  SUBROUTINE p_recv_int_1d (t_buffer, p_source, p_tag, p_count, comm)

    INTEGER, INTENT(out) :: t_buffer(:)
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_int, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_int_1d

  SUBROUTINE p_recv_int_2d (t_buffer, p_source, p_tag, p_count, comm)

    INTEGER, INTENT(out) :: t_buffer(:,:)
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_int, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_int_2d

  SUBROUTINE p_recv_int_3d (t_buffer, p_source, p_tag, p_count, comm)

    INTEGER, INTENT(out) :: t_buffer(:,:,:)
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_int, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_int_3d

  SUBROUTINE p_recv_int_4d (t_buffer, p_source, p_tag, p_count, comm)

    INTEGER, INTENT(out) :: t_buffer(:,:,:,:)
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_int, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_int_4d


  SUBROUTINE p_recv_bool (t_buffer, p_source, p_tag, p_count, comm, use_g2g)

    LOGICAL, INTENT(out) :: t_buffer
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_recv(t_buffer, icount, p_bool, p_source, p_tag, &
           &        p_comm, p_status, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_recv(t_buffer, icount, p_bool, p_source, p_tag, &
           &        p_comm, p_status, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_recv(t_buffer, icount, p_bool, p_source, p_tag, &
         &        p_comm, p_status, p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_bool

  SUBROUTINE p_recv_bool_1d (t_buffer, p_source, p_tag, p_count, comm)

    LOGICAL, INTENT(out) :: t_buffer(:)
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_bool, p_source, p_tag, &
         p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_bool_1d


  SUBROUTINE p_recv_char_1d (t_buffer, p_source, p_tag, p_count, comm)

    CHARACTER(LEN=*), INTENT(out) :: t_buffer(:)
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF
    icount = icount * LEN(t_buffer(1))


    CALL mpi_recv(t_buffer, icount, p_char, p_source, p_tag, &
         p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_char_1d


  SUBROUTINE p_recv_bool_2d (t_buffer, p_source, p_tag, p_count, comm)

    LOGICAL, INTENT(out) :: t_buffer(:,:)
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_bool, p_source, p_tag, &
         p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_bool_2d

  SUBROUTINE p_recv_bool_3d (t_buffer, p_source, p_tag, p_count, comm)

    LOGICAL, INTENT(out) :: t_buffer(:,:,:)
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_bool, p_source, p_tag, &
      &           p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_bool_3d

  SUBROUTINE p_recv_bool_4d (t_buffer, p_source, p_tag, p_count, comm)

    LOGICAL, INTENT(out) :: t_buffer(:,:,:,:)
    INTEGER, INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL mpi_recv(t_buffer, icount, p_bool, p_source, p_tag, &
         p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_recv_bool_4d

  SUBROUTINE p_recv_char (t_buffer, p_source, p_tag, p_count, comm)

    CHARACTER(len=*),  INTENT(out) :: t_buffer
    INTEGER,           INTENT(in)  :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
#ifdef DEBUG
    CHARACTER(LEN=*), PARAMETER :: routine = modname//'::p_recv_char'
#endif
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = LEN(t_buffer)
    END IF



    CALL mpi_recv(t_buffer, icount, p_char, p_source, p_tag, &
         p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish (routine, 'MPI function call failed')
#endif
#endif

  END SUBROUTINE p_recv_char

  ! non-blocking receives

  !================================================================================================
  ! CHARACTER SECTION -----------------------------------------------------------------------------
  !
  SUBROUTINE p_irecv_char (t_buffer, p_source, p_tag, p_count, comm)

    CHARACTER(len=*), INTENT(inout) :: t_buffer
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = LEN(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_char, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_char

  SUBROUTINE p_irecv_char_1d(t_buffer, p_source, p_tag, p_count, comm, request)
    CHARACTER(len=*), INTENT(inout) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    INTEGER, OPTIONAL, INTENT(out) :: request
#ifndef NOMPI
    INTEGER :: p_comm, icount, out_request

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF
    IF (icount > 0) icount = icount * LEN(t_buffer(1))

    CALL mpi_irecv(t_buffer, icount, p_char, p_source, p_tag, &
         p_comm, out_request, p_error)

    IF (PRESENT(request)) THEN
      request               = out_request
    ELSE
      CALL p_inc_request
      p_request(p_irequest) = out_request
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif
  END SUBROUTINE p_irecv_char_1d

  !================================================================================================
  ! REAL SECTION ----------------------------------------------------------------------------------
  !
#ifdef USE_NCCL
  SUBROUTINE p_irecv_nccl_real (t_buffer, p_source, p_count)
    REAL (dp), INTENT(inout) :: t_buffer
    INTEGER,   INTENT(in) :: p_source
    INTEGER,   INTENT(in) :: p_count

    !print *, "Recv on ", my_global_mpi_id, " from ", p_source, " size ", p_count

    CALL NCCL_CHECK(ncclRecv( acc_deviceptr(t_buffer), int(p_count, cuda_count_kind), ncclFloat64, &
                              p_source, nccl_comm, int(get_comm_stream(), cuda_stream_kind) ))

  END SUBROUTINE p_irecv_nccl_real
#endif

  SUBROUTINE p_irecv_real (t_buffer, p_source, p_tag, p_count, comm, use_g2g)

    REAL(dp),  INTENT(inout) :: t_buffer
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

    IF (nccl_active .and. my_process_is_work() .and. comm == p_comm_work .and. loc_use_g2g) THEN
#ifdef USE_NCCL
      CALL p_irecv_nccl_real(t_buffer, p_source, p_count)
#endif
    ELSE
      CALL p_inc_request
! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
      IF (loc_use_g2g) THEN
        !$ACC HOST_DATA USE_DEVICE(t_buffer)
        CALL mpi_irecv(t_buffer, icount, p_real_dp, p_source, p_tag, &
             &         p_comm, p_request(p_irequest), p_error)
        !$ACC END HOST_DATA
      ELSE
        CALL mpi_irecv(t_buffer, icount, p_real_dp, p_source, p_tag, &
             &         p_comm, p_request(p_irequest), p_error)
      END IF
#else
      !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
      CALL mpi_irecv(t_buffer, icount, p_real_dp, p_source, p_tag, &
           &         p_comm, p_request(p_irequest), p_error)
      !$ACC END HOST_DATA
#endif
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_real


  SUBROUTINE p_irecv_sreal (t_buffer, p_source, p_tag, p_count, comm, use_g2g)

    REAL(sp),  INTENT(inout) :: t_buffer
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

    CALL p_inc_request
! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_irecv(t_buffer, icount, p_real_sp, p_source, p_tag, &
           &         p_comm, p_request(p_irequest), p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_irecv(t_buffer, icount, p_real_sp, p_source, p_tag, &
           &         p_comm, p_request(p_irequest), p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_irecv(t_buffer, icount, p_real_sp, p_source, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_sreal

  SUBROUTINE p_irecv_real_1d (t_buffer, p_source, p_tag, p_count, comm)

    REAL(dp),  INTENT(inout) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_real_dp, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_real_1d


  SUBROUTINE p_irecv_sreal_1d (t_buffer, p_source, p_tag, p_count, comm)

    REAL(sp),  INTENT(inout) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_real_sp, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_sreal_1d


  SUBROUTINE p_irecv_real_2d (t_buffer, p_source, p_tag, p_count, comm)

    REAL(dp),  INTENT(inout) :: t_buffer(:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_real_dp, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_real_2d


  SUBROUTINE p_irecv_sreal_2d (t_buffer, p_source, p_tag, p_count, comm)

    REAL(sp),  INTENT(inout) :: t_buffer(:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_real_sp, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_sreal_2d


  SUBROUTINE p_irecv_real_3d (t_buffer, p_source, p_tag, p_count, comm)

    REAL(dp),  INTENT(inout) :: t_buffer(:,:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_real_dp, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_real_3d

  SUBROUTINE p_irecv_real_4d (t_buffer, p_source, p_tag, p_count, comm)

    REAL(dp),  INTENT(inout) :: t_buffer(:,:,:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_real_dp, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_real_4d
  !================================================================================================
  ! INTEGER SECTION -------------------------------------------------------------------------------
  !
  SUBROUTINE p_irecv_int (t_buffer, p_source, p_tag, p_count, comm, request, use_g2g)

    INTEGER, INTENT(inout) :: t_buffer
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    INTEGER, OPTIONAL, INTENT(INOUT) :: request
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount, out_request

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_irecv(t_buffer, icount, p_int, p_source, p_tag, &
           &         p_comm, out_request, p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_irecv(t_buffer, icount, p_int, p_source, p_tag, &
           &         p_comm, out_request, p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_irecv(t_buffer, icount, p_int, p_source, p_tag, &
         &         p_comm, out_request, p_error)
    !$ACC END HOST_DATA
#endif

    IF (PRESENT(request)) THEN
      request               = out_request
    ELSE
      CALL p_inc_request
      p_request(p_irequest) = out_request
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_int

  SUBROUTINE p_irecv_int_1d(t_buffer, p_source, p_tag, p_count, comm, &
    &                       request)

    INTEGER, INTENT(inout) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    INTEGER, OPTIONAL, INTENT(out) :: request
#ifndef NOMPI
    INTEGER :: p_comm, icount, out_request

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    ENDIF


    CALL mpi_irecv(t_buffer, icount, p_int, p_source, p_tag, &
         p_comm, out_request, p_error)

    IF (PRESENT(request)) THEN
      request               = out_request
    ELSE
      CALL p_inc_request
      p_request(p_irequest) = out_request
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_int_1d

  SUBROUTINE p_irecv_int_2d (t_buffer, p_source, p_tag, p_count, comm)

    INTEGER, INTENT(inout) :: t_buffer(:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    ENDIF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_int, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_int_2d

  SUBROUTINE p_irecv_int_3d (t_buffer, p_source, p_tag, p_count, comm)

    INTEGER, INTENT(inout) :: t_buffer(:,:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_int, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_int_3d

  SUBROUTINE p_irecv_int_4d (t_buffer, p_source, p_tag, p_count, comm)

    INTEGER, INTENT(inout) :: t_buffer(:,:,:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_int, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_int_4d

  !================================================================================================
  ! LOGICAL SECTION -------------------------------------------------------------------------------
  !
  SUBROUTINE p_irecv_bool (t_buffer, p_source, p_tag, p_count, comm, use_g2g)

    LOGICAL, INTENT(inout) :: t_buffer
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = 1
    END IF
    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .false.
    END IF

    CALL p_inc_request
! ACCWA (Cray Fortran <= 16.0.1.1) : ACC IF generate wrong assembly which segfaults CAST-32453
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 16
    IF (loc_use_g2g) THEN
      !$ACC HOST_DATA USE_DEVICE(t_buffer)
      CALL mpi_irecv(t_buffer, icount, p_bool, p_source, p_tag, &
           &         p_comm, p_request(p_irequest), p_error)
      !$ACC END HOST_DATA
    ELSE
      CALL mpi_irecv(t_buffer, icount, p_bool, p_source, p_tag, &
           &         p_comm, p_request(p_irequest), p_error)
    END IF
#else
    !$ACC HOST_DATA USE_DEVICE(t_buffer) IF(loc_use_g2g)
    CALL mpi_irecv(t_buffer, icount, p_bool, p_source, p_tag, &
         &         p_comm, p_request(p_irequest), p_error)
    !$ACC END HOST_DATA
#endif

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_bool

  SUBROUTINE p_irecv_bool_1d (t_buffer, p_source, p_tag, p_count, comm)

    LOGICAL, INTENT(inout) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm
#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_bool, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_bool_1d

  SUBROUTINE p_irecv_bool_2d (t_buffer, p_source, p_tag, p_count, comm)

    LOGICAL, INTENT(inout) :: t_buffer(:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_bool, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_bool_2d

  SUBROUTINE p_irecv_bool_3d (t_buffer, p_source, p_tag, p_count, comm)

    LOGICAL, INTENT(inout) :: t_buffer(:,:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_bool, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_bool_3d

  SUBROUTINE p_irecv_bool_4d (t_buffer, p_source, p_tag, p_count, comm)

    LOGICAL, INTENT(inout) :: t_buffer(:,:,:,:)
    INTEGER,   INTENT(in) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(in) :: p_count, comm

#ifndef NOMPI
    INTEGER :: p_comm, icount

    IF (PRESENT(comm)) THEN
      p_comm = comm
    ELSE
      p_comm = process_mpi_all_comm
    ENDIF
    IF (PRESENT(p_count)) THEN
      icount = p_count
    ELSE
      icount = SIZE(t_buffer)
    END IF


    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_bool, p_source, p_tag, &
         p_comm, p_request(p_irequest), p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_bool_4d

  SUBROUTINE p_pack_int (t_var, t_buffer, p_pos, comm)

    INTEGER,   INTENT(IN)    :: t_var
    CHARACTER, INTENT(INOUT) :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_PACK(t_var, 1, p_int, t_buffer, SIZE(t_buffer), p_pos, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_pack_int", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_pack_int

  SUBROUTINE p_pack_bool (t_var, t_buffer, p_pos, comm)

    LOGICAL,   INTENT(IN)    :: t_var
    CHARACTER, INTENT(INOUT) :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_PACK(t_var, 1, p_bool, t_buffer, SIZE(t_buffer), p_pos, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_pack_bool", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_pack_bool

  SUBROUTINE p_pack_real (t_var, t_buffer, p_pos, comm)

    REAL(wp),  INTENT(IN)    :: t_var
    CHARACTER, INTENT(INOUT) :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_PACK(t_var, 1, p_real_dp, t_buffer, SIZE(t_buffer), p_pos, p_comm, p_error)
#ifdef DEBUG
   IF (p_error /= MPI_SUCCESS) CALL finish ("p_pack_real", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_pack_real

  SUBROUTINE p_pack_int_1d (t_var, p_count, t_buffer, p_pos, comm)

    INTEGER,   INTENT(IN)    :: t_var(:)
    INTEGER,   INTENT(IN)    :: p_count
    CHARACTER, INTENT(INOUT) :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_PACK(t_var, p_count, p_int, t_buffer, SIZE(t_buffer), p_pos, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_pack_int_1d", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_pack_int_1d

  SUBROUTINE p_pack_real_1d (t_var, p_count, t_buffer, p_pos, comm)

    REAL(wp),  INTENT(IN)    :: t_var(:)
    INTEGER,   INTENT(IN)    :: p_count
    CHARACTER, INTENT(INOUT) :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_PACK(t_var, p_count, p_real_dp, t_buffer, SIZE(t_buffer), p_pos, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_pack_real_1d", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_pack_real_1d


  SUBROUTINE p_pack_string (t_var, t_buffer, p_pos, comm)

    CHARACTER(LEN=*), INTENT(IN) :: t_var
    CHARACTER, INTENT(INOUT) :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm, ilength

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    ilength = LEN_TRIM(t_var)
    ! first pack string length, then the character sequence
    CALL MPI_PACK(ilength,       1,  p_int, t_buffer, SIZE(t_buffer), p_pos, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_pack_char_1d", 'MPI call failed')
#endif
    CALL MPI_PACK(t_var(1:ilength), ilength, p_char, t_buffer, SIZE(t_buffer), p_pos, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_pack_char_1d", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_pack_string

  SUBROUTINE p_pack_real_2d (t_var, p_count, t_buffer, p_pos, comm)

    REAL(wp),  INTENT(IN)    :: t_var(:,:)
    INTEGER,   INTENT(IN)    :: p_count
    CHARACTER, INTENT(INOUT) :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_PACK(t_var, p_count, p_real_dp, t_buffer, SIZE(t_buffer), p_pos, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_pack_real_2d", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_pack_real_2d

  FUNCTION p_pack_size_int(p_count, comm) RESULT(pack_size)
    INTEGER, INTENT(in) :: p_count, comm
    INTEGER :: pack_size
#ifndef NOMPI
    CALL mpi_pack_size(p_count, p_int, comm, pack_size, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish("p_pack_size_int", 'MPI call failed')
#endif
#else
    ! packing is only supported when mpi is available
    pack_size = -1
#endif
  END FUNCTION p_pack_size_int

  FUNCTION p_pack_size_bool(p_count, comm) result(pack_size)
    INTEGER, INTENT(in) :: p_count, comm
    INTEGER :: pack_size
#ifndef NOMPI
    CALL mpi_pack_size(p_count, p_bool, comm, pack_size, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish("p_pack_size_bool", 'MPI call failed')
#endif
#else
    ! packing is only supported when mpi is available
    pack_size = -1
#endif
  END FUNCTION p_pack_size_bool

  FUNCTION p_pack_size_real_dp(p_count, comm) RESULT(pack_size)
    INTEGER, INTENT(in) :: p_count, comm
    INTEGER :: pack_size
#ifndef NOMPI
    CALL mpi_pack_size(p_count, p_real_dp, comm, pack_size, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish("p_pack_size_int", 'MPI call failed')
#endif
#else
    ! packing is only supported when mpi is available
    pack_size = -1
#endif
  END FUNCTION p_pack_size_real_dp

  FUNCTION p_pack_size_string(maxlen, comm) RESULT(pack_size)
    INTEGER, INTENT(in) :: maxlen, comm
    INTEGER :: pack_size, pack_size_int
#ifndef NOMPI
    CALL mpi_pack_size(1, p_int, comm, pack_size_int, p_error)
    IF (p_error == MPI_SUCCESS) &
         CALL mpi_pack_size(maxlen, p_char, comm, pack_size, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish("p_pack_size_string", 'MPI call failed')
#endif
    pack_size = pack_size + pack_size_int
#else
    ! packing is only supported when mpi is available
    pack_size = -1
#endif
  END FUNCTION p_pack_size_string

  SUBROUTINE p_unpack_int (t_buffer, p_pos, t_var, comm)

    CHARACTER, INTENT(IN)    :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    INTEGER,   INTENT(OUT)   :: t_var
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_UNPACK(t_buffer, SIZE(t_buffer), p_pos, t_var, 1, p_int, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_unpack_int", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_unpack_int

  SUBROUTINE p_unpack_bool (t_buffer, p_pos, t_var, comm)

    CHARACTER, INTENT(IN)    :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    LOGICAL,   INTENT(OUT)   :: t_var
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_UNPACK(t_buffer, SIZE(t_buffer), p_pos, t_var, 1, p_bool, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_unpack_int", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_unpack_bool

  SUBROUTINE p_unpack_real (t_buffer, p_pos, t_var, comm)

    CHARACTER, INTENT(IN)    :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    REAL(wp),  INTENT(OUT)   :: t_var
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_UNPACK(t_buffer, SIZE(t_buffer), p_pos, t_var, 1, p_real_dp, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_unpack_real", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_unpack_real

  SUBROUTINE p_unpack_int_1d (t_buffer, p_pos, t_var, p_count, comm)

    CHARACTER, INTENT(IN)    :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    INTEGER,   INTENT(INOUT) :: t_var(:)
    INTEGER,   INTENT(IN)    :: p_count
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_UNPACK(t_buffer, SIZE(t_buffer), p_pos, t_var, p_count, p_int, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_unpack_int_1d", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_unpack_int_1d

  SUBROUTINE p_unpack_real_1d (t_buffer, p_pos, t_var, p_count, comm)

    CHARACTER, INTENT(IN)    :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    REAL(wp),  INTENT(INOUT) :: t_var(:)
    INTEGER,   INTENT(IN)    :: p_count
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_UNPACK(t_buffer, SIZE(t_buffer), p_pos, t_var, p_count, p_real_dp, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_unpack_real_1d", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_unpack_real_1d


  SUBROUTINE p_unpack_string (t_buffer, p_pos, t_var, comm)

    CHARACTER, INTENT(IN) :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    CHARACTER(LEN=*), INTENT(INOUT) :: t_var
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm, ilength

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    ! first unpack string length, then the character sequence
    CALL MPI_UNPACK(t_buffer, SIZE(t_buffer), p_pos, ilength,       1,  p_int, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_unpack_char_1d", 'MPI call failed')
#endif
    t_var = " "
    CALL MPI_UNPACK(t_buffer, SIZE(t_buffer), p_pos,   t_var, ilength, p_char, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_unpack_char_1d", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_unpack_string


  SUBROUTINE p_unpack_real_2d (t_buffer, p_pos, t_var, p_count, comm)

    CHARACTER, INTENT(IN)    :: t_buffer(:)
    INTEGER,   INTENT(INOUT) :: p_pos
    REAL(wp),  INTENT(INOUT) :: t_var(:,:)
    INTEGER,   INTENT(IN)    :: p_count
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_UNPACK(t_buffer, SIZE(t_buffer), p_pos, t_var, p_count, p_real_dp, p_comm, p_error)
#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) CALL finish ("p_unpack_real_2d", 'MPI call failed')
#endif
#endif
  END SUBROUTINE p_unpack_real_2d

  SUBROUTINE p_recv_packed (t_buffer, p_source, p_tag, p_count, comm)

    CHARACTER, INTENT(INOUT) :: t_buffer(:)
    INTEGER,   INTENT(IN)    :: p_source, p_tag
    INTEGER,   INTENT(IN)    :: p_count
    INTEGER, OPTIONAL, INTENT(IN) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_RECV (t_buffer, p_count, MPI_PACKED, p_source, p_tag, &
      p_comm, p_status, p_error)

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_RECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL p_abort
    END IF
#endif
#endif

  END SUBROUTINE p_recv_packed

  SUBROUTINE p_irecv_packed (t_buffer, p_source, p_tag, p_count, comm, &
       request)

    CHARACTER, INTENT(INOUT) :: t_buffer(:)
    INTEGER,   INTENT(IN)    :: p_source, p_tag
    INTEGER,   INTENT(IN)    :: p_count
    INTEGER, OPTIONAL, INTENT(IN) :: comm
    INTEGER, OPTIONAL, INTENT(out) :: request
#ifndef NOMPI
    INTEGER :: p_comm, out_request

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_IRECV (t_buffer, p_count, MPI_PACKED, p_source, p_tag, &
      p_comm, out_request, p_error)

    IF (PRESENT(request)) THEN
      request               = out_request
    ELSE
      CALL p_inc_request
      p_request(p_irequest) = out_request
    END IF

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', my_process_mpi_all_id, &
            ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_irecv_packed

  SUBROUTINE p_send_packed (t_buffer, p_destination, p_tag, p_count, comm)

    CHARACTER, INTENT(in) :: t_buffer(:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER,   INTENT(in) :: p_count
    INTEGER, OPTIONAL, INTENT(in) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    CALL MPI_SEND (t_buffer, p_count, MPI_PACKED, p_destination, p_tag, &
      p_comm, p_error)

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_packed

  SUBROUTINE p_send_packed_2d(t_buffer, p_destination, p_tag, p_count, comm)

    CHARACTER, INTENT(in) :: t_buffer(:,:)
    INTEGER,   INTENT(in) :: p_destination, p_tag
    INTEGER,   INTENT(in) :: p_count
    INTEGER, OPTIONAL, INTENT(in) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF
    CALL MPI_SEND (t_buffer, p_count, MPI_PACKED, p_destination, p_tag, &
      p_comm, p_error)

#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_SEND from ', my_process_mpi_all_id, &
            ' to ', p_destination, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_send_packed_2d

  SUBROUTINE p_bcast_packed (t_buffer, p_source, p_count, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    CHARACTER :: t_buffer(:)
    INTEGER,   INTENT(in)    :: p_source
    INTEGER,   INTENT(in)    :: p_count
    INTEGER, OPTIONAL, INTENT(in) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_BCAST (t_buffer, p_count, MPI_PACKED, p_source, &
            p_comm, p_error)

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast successful.'

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL p_abort
    END IF
#endif

#endif
  END SUBROUTINE p_bcast_packed

  !
  !================================================================================================

  ! sendrecv implementation

  SUBROUTINE p_sendrecv_real_1d (sendbuf, p_dest, recvbuf, p_source, &
                                  p_tag, comm)

    REAL(dp), INTENT(in)           :: sendbuf (:)
    INTEGER,  INTENT(in)           :: p_dest
    REAL(dp), INTENT(out)          :: recvbuf (:)
    INTEGER,  INTENT(in)           :: p_source
    INTEGER,  INTENT(in)           :: p_tag
    INTEGER,  INTENT(in) ,OPTIONAL :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF


    CALL MPI_SENDRECV (sendbuf, SIZE(sendbuf), p_real_dp, p_dest,   p_tag, &
                        recvbuf, SIZE(recvbuf), p_real_dp, p_source, p_tag, &
                        p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i4,a,i6,a)') ' MPI_SENDRECV by ', my_process_mpi_all_id, &
            ' to ', p_dest, ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_sendrecv_real_1d

  SUBROUTINE p_sendrecv_real_2d (sendbuf, p_dest, recvbuf, p_source, &
                                  p_tag, comm)

    REAL(dp), INTENT(in)           :: sendbuf (:,:)
    INTEGER,  INTENT(in)           :: p_dest
    REAL(dp), INTENT(out)          :: recvbuf (:,:)
    INTEGER,  INTENT(in)           :: p_source
    INTEGER,  INTENT(in)           :: p_tag
    INTEGER,  INTENT(in) ,OPTIONAL :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF


    CALL MPI_SENDRECV (sendbuf, SIZE(sendbuf), p_real_dp, p_dest,   p_tag, &
                        recvbuf, SIZE(recvbuf), p_real_dp, p_source, p_tag, &
                        p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i4,a,i6,a)') ' MPI_SENDRECV by ', my_process_mpi_all_id, &
            ' to ', p_dest, ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_sendrecv_real_2d

  SUBROUTINE p_sendrecv_real_3d (sendbuf, p_dest, recvbuf, p_source, &
                                  p_tag, comm)

    REAL(dp), INTENT(in)           :: sendbuf (:,:,:)
    INTEGER,  INTENT(in)           :: p_dest
    REAL(dp), INTENT(out)          :: recvbuf (:,:,:)
    INTEGER,  INTENT(in)           :: p_source
    INTEGER,  INTENT(in)           :: p_tag
    INTEGER,  INTENT(in) ,OPTIONAL :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF


    CALL MPI_SENDRECV (sendbuf, SIZE(sendbuf), p_real_dp, p_dest,   p_tag, &
                        recvbuf, SIZE(recvbuf), p_real_dp, p_source, p_tag, &
                        p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i4,a,i6,a)') ' MPI_SENDRECV by ', my_process_mpi_all_id, &
            ' to ', p_dest, ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_sendrecv_real_3d

  SUBROUTINE p_sendrecv_real_4d (sendbuf, p_dest, recvbuf, p_source, &
                                  p_tag, comm)

    REAL(dp), INTENT(in)           :: sendbuf (:,:,:,:)
    INTEGER,  INTENT(in)           :: p_dest
    REAL(dp), INTENT(out)          :: recvbuf (:,:,:,:)
    INTEGER,  INTENT(in)           :: p_source
    INTEGER,  INTENT(in)           :: p_tag
    INTEGER,  INTENT(in) ,OPTIONAL :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF


    CALL MPI_SENDRECV (sendbuf, SIZE(sendbuf), p_real_dp, p_dest,   p_tag, &
                        recvbuf, SIZE(recvbuf), p_real_dp, p_source, p_tag, &
                        p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i4,a,i6,a)') ' MPI_SENDRECV by ', my_process_mpi_all_id, &
            ' to ', p_dest, ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_sendrecv_real_4d

  SUBROUTINE p_sendrecv_char_array (sendbuf, p_dest, recvbuf, p_source, p_tag, comm)
    CHARACTER(KIND = C_CHAR), INTENT(in) :: sendbuf (:)
    CHARACTER(KIND = C_CHAR), INTENT(out) :: recvbuf (:)
    INTEGER, VALUE :: p_dest, p_source, p_tag
    INTEGER, INTENT(in), OPTIONAL :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF


    CALL MPI_Sendrecv(sendbuf(1), SIZE(sendbuf, 1), MPI_CHARACTER, p_dest,   p_tag, &
    &                 recvbuf(1), SIZE(recvbuf, 1), MPI_CHARACTER, p_source, p_tag, &
    &                 p_comm, p_status, p_error)


#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a,i4,a,i4,a,i6,a)') ' MPI_SENDRECV by ', my_process_mpi_all_id, &
            ' to ', p_dest, ' from ', p_source, ' for tag ', p_tag, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_sendrecv_char_array

  ! bcast implementation

  SUBROUTINE p_bcast_real (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL(dp) :: t_buffer
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, 1, p_real_dp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real

  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Bcast
  !---------------------------------------------------------------------------------------------------------------------------------
  SUBROUTINE p_bcast_real_single (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL(sp) :: t_buffer
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, 1, p_real_sp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real_single

  SUBROUTINE p_bcast_real_1d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL(dp) :: t_buffer(:)
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_real_dp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real_1d

  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Bcast
  !---------------------------------------------------------------------------------------------------------------------------------
  SUBROUTINE p_bcast_real_1d_single (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL(sp) :: t_buffer(:)
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_real_sp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real_1d_single

  SUBROUTINE p_bcast_real_2d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL(dp) :: t_buffer(:,:)
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_real_dp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real_2d


  SUBROUTINE p_bcast_real_2d_single (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL(sp) :: t_buffer(:,:) ! SINGLE PRECISION
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_real_sp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real_2d_single


  SUBROUTINE p_bcast_real_3d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL (dp) :: t_buffer(:,:,:)
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_real_dp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real_3d

  SUBROUTINE p_bcast_real_4d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL(dp) :: t_buffer(:,:,:,:)
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_real_dp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real_4d

  SUBROUTINE p_bcast_real_5d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL(dp) :: t_buffer(:,:,:,:,:)
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_real_dp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real_5d

  SUBROUTINE p_bcast_real_7d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    REAL(dp) :: t_buffer(:,:,:,:,:,:,:)
    INTEGER,   INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_real_dp, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_real_7d

  SUBROUTINE p_bcast_int_i4 (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    INTEGER(i4) :: t_buffer
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, 1, p_int_i4, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_int_i4

  SUBROUTINE p_bcast_int_i8 (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    INTEGER(i8) :: t_buffer
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, 1, p_int_i8, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_int_i8

  SUBROUTINE p_bcast_int_1d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    INTEGER :: t_buffer(:)
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_int, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_int_1d

  SUBROUTINE p_bcast_int_i8_1d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    INTEGER(i8) :: t_buffer(:)
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_int_i8, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_int_i8_1d

  SUBROUTINE p_bcast_int_2d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    INTEGER :: t_buffer(:,:)
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_int, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_int_2d

  SUBROUTINE p_bcast_int_3d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    INTEGER :: t_buffer(:,:,:)
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_int, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_int_3d

  SUBROUTINE p_bcast_int_4d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    INTEGER :: t_buffer(:,:,:,:)
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_int, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_int_4d

  SUBROUTINE p_bcast_int_7d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    INTEGER                :: t_buffer(:,:,:,:,:,:,:)
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_int, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_int_7d

  SUBROUTINE p_bcast_bool (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    LOGICAL :: t_buffer
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, 1, p_bool, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_bool

  SUBROUTINE p_bcast_bool_1d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    LOGICAL :: t_buffer(:)
    INTEGER, INTENT(in) :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_bool, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_bool_1d

  SUBROUTINE p_bcast_bool_2d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    LOGICAL :: t_buffer(:,:)
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_bool, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_bool_2d

  SUBROUTINE p_bcast_bool_3d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    LOGICAL :: t_buffer(:,:,:)
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_bool, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_bool_3d

  SUBROUTINE p_bcast_bool_4d (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    LOGICAL :: t_buffer(:,:,:,:)
    INTEGER, INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, SIZE(t_buffer), p_bool, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_bool_4d

  SUBROUTINE p_bcast_char (t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    CHARACTER(len=*) :: t_buffer
    INTEGER,           INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, LEN(t_buffer), p_char, p_source, &
            p_comm, p_error)


    ENDIF

#ifdef DEBUG
    WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

     IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
#endif
#endif

  END SUBROUTINE p_bcast_char

  SUBROUTINE p_bcast_char_1d(t_buffer, p_source, comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    CHARACTER(*) :: t_buffer(:)
    INTEGER,       INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm
    INTEGER                      :: lexlength, flength

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE
       lexlength=LEN(t_buffer(1))
       flength=SIZE(t_buffer)
       lexlength=lexlength*flength


       CALL MPI_BCAST (t_buffer, lexlength, p_char, p_source, p_comm, p_error)


#ifdef DEBUG
       WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
               ' failed.'
          WRITE (nerr,'(a,i4)') ' Error = ', p_error
          CALL abort_mpi
       END IF
#endif
    ENDIF
#endif

  END SUBROUTINE p_bcast_char_1d

  SUBROUTINE p_bcast_char_2d(t_buffer, p_source, comm)
    CHARACTER(*) :: t_buffer(:,:)
    INTEGER,       INTENT(in)    :: p_source
    INTEGER, OPTIONAL, INTENT(in) :: comm
    INTEGER :: lexlength, flength

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE
       lexlength=LEN(t_buffer(1,1))
       flength=SIZE(t_buffer)
       lexlength=lexlength*flength
       CALL mpi_bcast(t_buffer, lexlength, p_char, p_source, p_comm, p_error)
#ifdef DEBUG
       WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
               ' failed.'
          WRITE (nerr,'(a,i4)') ' Error = ', p_error
          CALL abort_mpi
       END IF
#endif
    ENDIF
#endif
  END SUBROUTINE p_bcast_char_2d



  SUBROUTINE p_bcast_cchar (t_buffer, buflen, p_source, p_comm)
    ! intentionally without argument intent, because p_source determines if
    ! t_buffer is read or written
    INTEGER,           INTENT(IN)    :: buflen
    INTEGER(C_SIGNED_CHAR)                :: t_buffer(buflen)

    INTEGER,           INTENT(IN)    :: p_source
    INTEGER,           INTENT(IN)    :: p_comm

#ifndef NOMPI

#ifdef DEBUG
    nbcast = nbcast+1
#endif

    IF (process_mpi_all_size == 1) THEN
       RETURN
    ELSE


       CALL MPI_BCAST (t_buffer, buflen, p_char, p_source, p_comm, p_error)


#ifdef DEBUG
       WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_BCAST from ', p_source, &
            ' with broadcast number ', nbcast, ' successful.'

       IF (p_error /= MPI_SUCCESS) THEN
          WRITE (nerr,'(a,i4,a)') ' MPI_BCAST from ', p_source, &
               ' failed.'
          WRITE (nerr,'(a,i4)') ' Error = ', p_error
          CALL abort_mpi
       END IF
#endif
    ENDIF
#endif

  END SUBROUTINE p_bcast_cchar

  ! A bcast variant that handles ALLOCATABLE strings. Cannot be overloaded with p_bcast() because that would make the CALL ambigous.
  ! After this CALL, the string will always be ALLOCATED, even IF its length IS zero
  !
  !XXX: Deactivated due to a gfortran bug that looses the contents of allocatable strings on return.
!  SUBROUTINE p_bcast_achar(string, source, comm)
!    CHARACTER(:), ALLOCATABLE, INTENT(INOUT) :: string
!    INTEGER, VALUE :: source, comm
!
!    INTEGER :: length, error
!    CHARACTER(*), PARAMETER :: routine = modname//":p_bcast_achar"
!
!#ifndef NOMPI
!    ! inform the receivers about the length of the string
!    length = 0
!    IF(ALLOCATED(string)) length = LEN(string)
!    CALL p_bcast(length, source, comm)
!
!    ! ensure that the string IS ALLOCATED to the correct length
!    IF(ALLOCATED(string)) THEN
!        IF(length /= LEN(string) .OR. length == 0) DEALLOCATE(string)
!    END IF
!    IF(.NOT.ALLOCATED(string)) THEN
!        ALLOCATE(CHARACTER(LEN = length) :: string, STAT = error)
!        IF(error /= SUCCESS) CALL finish(routine, "memory allocation failed")
!    END IF
!
!    ! actually TRANSFER the string
!    CALL p_bcast(string, source, comm)
!#endif
!    END SUBROUTINE p_bcast_achar


!DR Test
  SUBROUTINE p_bcast_datetime (mtime_datetime, p_source, comm)

    TYPE(datetime), TARGET                  :: mtime_datetime
    INTEGER                 , INTENT(in)    :: p_source
    INTEGER       , OPTIONAL, INTENT(in)    :: comm
    !
    ! local
    character(len=max_datetime_str_len) :: mtime_datetime_str
    TYPE(datetime), POINTER             :: mtime_datetime_ptr
    TYPE(datetime), POINTER             :: datetime_loc
    INTEGER                             :: errno
 
#ifndef NOMPI
    mtime_datetime_ptr => mtime_datetime
    CALL datetimeToString(mtime_datetime_ptr, mtime_datetime_str, errno)

    CALL p_bcast_char (mtime_datetime_str, p_source, comm)

    datetime_loc => newDatetime(mtime_datetime_str, errno)

    mtime_datetime = datetime_loc

    CALL deallocateDatetime(datetime_loc)
#endif

  END SUBROUTINE p_bcast_datetime



  ! Collective CALL to determine whether this process IS a sender/receiver IN a broadcast operation.
  ! This routine IS robust IN the presence of inter-communicators (which IS its reason d'etre).
  SUBROUTINE p_get_bcast_role(root, comm, isSender, isReceiver)
    INTEGER, INTENT(IN) :: root, comm
    LOGICAL, INTENT(OUT) :: isSender, isReceiver
#ifndef NOMPI
    INTEGER :: ierr
    LOGICAL :: isInterComm
    CHARACTER(*), PARAMETER :: routine = modname//"::p_get_bcast_role"

    CALL MPI_Comm_test_inter(comm, isInterComm, ierr)
    IF(ierr /= MPI_SUCCESS) CALL finish(routine, "MPI_Comm_test_inter() returned an error")
    IF(isInterComm) THEN
        isSender = root == MPI_ROOT
        isReceiver = root /= MPI_ROOT .AND. root /= MPI_PROC_NULL
    ELSE
        isSender = root == p_comm_rank(comm)
        isReceiver = .NOT.isSender
    END IF
#else
    isSender = .TRUE.
    isReceiver = .FALSE.
#endif
  END SUBROUTINE p_get_bcast_role

  ! probe implementation

  SUBROUTINE p_probe (p_tagcount, p_tagtable, p_source, &
       p_tag, p_count, comm)

    INTEGER,   INTENT(in)  :: p_tagcount, p_tagtable(:)
    INTEGER,   INTENT(out) :: p_source, p_tag, p_count
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm, i
    LOGICAL :: flag

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    p_tag = -1
    DO WHILE (p_tag == -1)
       DO i = 1, p_tagcount
          CALL MPI_IPROBE (MPI_ANY_SOURCE, p_tagtable(i), p_comm, &
               flag, p_status, p_error)
#ifdef DEBUG
          IF (p_error /= MPI_SUCCESS) THEN
             WRITE (nerr,'(a,i4,a,i4,a)') ' MPI_IPROBE on ', my_process_mpi_all_id, &
                  ' for tag ', p_tagtable(i), ' failed.'
             WRITE (nerr,'(a,i4)') ' Error = ', p_error
             CALL abort_mpi
          END IF
#endif
          IF (flag) THEN
             p_source = p_status(MPI_SOURCE)
             p_tag = p_status(MPI_TAG)
             CALL MPI_GET_COUNT(p_status, p_real_dp, p_count, p_error)
#ifdef DEBUG
             IF (p_error /= MPI_SUCCESS) THEN
                WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_GET_COUNT on ', &
                  & my_process_mpi_all_id, &
                     ' for tag ', p_tag, ' from ' , p_source, ' failed.'
                WRITE (nerr,'(a,i4)') ' Error = ', p_error
                CALL abort_mpi
             END IF
#endif
          ELSE
             p_tag = -1
          END IF
       END DO
    END DO

#endif

  END SUBROUTINE p_probe
  !------------------------------------------------------

  !------------------------------------------------------
  SUBROUTINE p_wait
#ifndef NOMPI
    IF (p_irequest > 0) CALL mpi_waitall(p_irequest, p_request, mpi_statuses_ignore, p_error)
    p_irequest = 0
#endif
  END SUBROUTINE p_wait

  SUBROUTINE p_wait_1(request)
    INTEGER, INTENT(INOUT) :: request
#ifndef NOMPI
    CALL mpi_wait(request, mpi_status_ignore, p_error)
    CALL p_clear_request(request)
#endif
  END SUBROUTINE p_wait_1

  SUBROUTINE p_wait_n(requests)
    INTEGER, INTENT(INOUT) :: requests(:)
#ifndef NOMPI
    CALL mpi_waitall(SIZE(requests), requests, mpi_statuses_ignore, p_error)
    CALL p_clear_request(requests)
#endif
  END SUBROUTINE p_wait_n

  SUBROUTINE p_wait_n_stati(requests, stati)
    INTEGER, INTENT(INOUT) :: requests(:)
    INTEGER, INTENT(out) :: stati(:,:)
#ifndef NOMPI
    CALL mpi_waitall(SIZE(requests), requests, stati, p_error)
    CALL p_clear_request(requests)
#endif
  END SUBROUTINE p_wait_n_stati

  SUBROUTINE p_wait_any(return_pe)

    INTEGER, INTENT(out) :: return_pe
#ifndef NOMPI
    INTEGER :: i

    CALL MPI_WAITANY(p_irequest, p_request, i, p_status, p_error)
    IF (i == MPI_UNDEFINED) THEN
      p_irequest = 0
      return_pe = -1
    ELSE
      return_pe = p_status(MPI_SOURCE)
    ENDIF
#else
    return_pe = 0
#endif
  END SUBROUTINE p_wait_any
  !------------------------------------------------------


  !------------------------------------------------------
  FUNCTION p_test() RESULT(ret)
    LOGICAL :: ret
#ifndef NOMPI
    CALL MPI_TESTALL(p_irequest, p_request, ret, mpi_statuses_ignore, p_error)
#else
    ret = .TRUE.
#endif
  END FUNCTION p_test


  !------------------------------------------------------
  SUBROUTINE p_barrier (comm)
  INTEGER ,INTENT(IN) ,OPTIONAL :: comm
#ifndef NOMPI
    INTEGER :: com
    com = MPI_COMM_WORLD; IF(PRESENT(comm)) com = comm
    CALL MPI_BARRIER (com, p_error)

!#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' MPI_BARRIER on ', my_process_mpi_all_id, ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
!#endif
#endif

  END SUBROUTINE p_barrier
  !------------------------------------------------------

  !------------------------------------------------------
  SUBROUTINE global_mpi_barrier
#ifndef NOMPI
    CALL MPI_BARRIER (global_mpi_communicator, p_error)

!#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' global_mpi_barrier on ', get_my_global_mpi_id(), ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
!#endif
#endif

  END SUBROUTINE global_mpi_barrier
  !------------------------------------------------------

  !------------------------------------------------------
  SUBROUTINE work_mpi_barrier
#ifndef NOMPI
    CALL MPI_BARRIER (p_comm_work, p_error)

!#ifdef DEBUG
    IF (p_error /= MPI_SUCCESS) THEN
       WRITE (nerr,'(a,i4,a)') ' global_mpi_barrier on ', get_my_global_mpi_id(), ' failed.'
       WRITE (nerr,'(a,i4)') ' Error = ', p_error
       CALL abort_mpi
    END IF
!#endif
#endif

  END SUBROUTINE work_mpi_barrier
  !------------------------------------------------------

  !------------------------------------------------------
  !> computes a global sum of real single precision numbers
  !
  !  perform an ALLREDUCE operation
  FUNCTION p_sum_sp_0d (zfield, comm) RESULT (p_sum)

    REAL(sp)                        :: p_sum
    REAL(sp),  INTENT(in)           :: zfield
    INTEGER,   INTENT(in)           :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    p_comm = comm

    IF (my_process_is_mpi_parallel()) THEN
        CALL MPI_ALLREDUCE (zfield, p_sum, 1, p_real_sp, &
          mpi_sum, p_comm, p_error)
    ELSE
       p_sum = zfield
    END IF
#else
    p_sum = zfield
#endif
  END FUNCTION p_sum_sp_0d
  !------------------------------------------------------

  !------------------------------------------------------
  !> computes a global sum of real numbers
  !
  ! @param[in]    root     (Optional:) root PE, otherwise we perform an
  !                                    ALLREDUCE operation
  FUNCTION p_sum_dp_0d (zfield, comm, root) RESULT (p_sum)

    REAL(dp)                        :: p_sum
    REAL(dp),  INTENT(in)           :: zfield
    INTEGER,   INTENT(in)           :: comm
    INTEGER,   INTENT(in), OPTIONAL :: root
#ifndef NOMPI
    INTEGER :: p_comm, my_rank

    p_comm = comm

    IF (my_process_is_mpi_parallel()) THEN
      IF (PRESENT(root)) THEN
        CALL MPI_REDUCE (zfield, p_sum, 1, p_real_dp, &
          mpi_sum, root, p_comm, p_error)
        ! get local PE identification
        CALL MPI_COMM_RANK (p_comm, my_rank, p_error)
        ! do not use the result on all the other ranks:
        IF (root /= my_rank)  p_sum = zfield
      ELSE
        CALL MPI_ALLREDUCE (zfield, p_sum, 1, p_real_dp, &
          mpi_sum, p_comm, p_error)
      END IF
    ELSE
       p_sum = zfield
    END IF
#else
    p_sum = zfield
#endif
  END FUNCTION p_sum_dp_0d
  !------------------------------------------------------

  !------------------------------------------------------
  FUNCTION p_sum_sp_1d (zfield, comm, root) RESULT (p_sum)

    REAL(sp),          INTENT(in) :: zfield(:)
    INTEGER, OPTIONAL, INTENT(in) :: comm, root
    REAL(sp)                      :: p_sum (SIZE(zfield))

#ifndef NOMPI
    INTEGER :: p_comm, my_rank

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
      IF (PRESENT(root)) THEN
        CALL mpi_reduce(zfield, p_sum, SIZE(zfield), p_real_sp, &
             mpi_sum, root, p_comm, p_error)
        ! get local PE identification
        CALL mpi_comm_rank(p_comm, my_rank, p_error)
        ! do not use the result on all the other ranks:
        IF (root /= my_rank) p_sum = zfield
      ELSE
        CALL mpi_allreduce (zfield, p_sum, SIZE(zfield), p_real_sp, &
             mpi_sum, p_comm, p_error)
      END IF
    ELSE
       p_sum = zfield
    END IF
#else
    p_sum = zfield
#endif

  END FUNCTION p_sum_sp_1d

  !------------------------------------------------------
  FUNCTION p_sum_dp_1d (zfield, comm, root, use_g2g) RESULT (p_sum)

    REAL(dp),          INTENT(in) :: zfield(:)
    INTEGER, OPTIONAL, INTENT(in) :: comm, root
    REAL(dp)                      :: p_sum (SIZE(zfield))
    LOGICAL, OPTIONAL, INTENT(in) :: use_g2g
    LOGICAL :: loc_use_g2g

#ifndef NOMPI
    INTEGER :: p_comm, my_rank

    IF (PRESENT(use_g2g)) THEN
      loc_use_g2g = use_g2g
    ELSE
      loc_use_g2g = .FALSE.
    END IF

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
      IF (PRESENT(root)) THEN
        CALL mpi_reduce(zfield, p_sum, SIZE(zfield), p_real_dp, &
             mpi_sum, root, p_comm, p_error)
        ! get local PE identification
        CALL mpi_comm_rank(p_comm, my_rank, p_error)
        ! do not use the result on all the other ranks:
        IF (root /= my_rank) p_sum = zfield
      ELSE

        !$ACC HOST_DATA USE_DEVICE(zfield) IF(loc_use_g2g)

        CALL mpi_allreduce (zfield, p_sum, SIZE(zfield), p_real_dp, &
             mpi_sum, p_comm, p_error)

        !$ACC END HOST_DATA

      END IF
    ELSE
       p_sum = zfield
    END IF
#else
    p_sum = zfield
#endif

  END FUNCTION p_sum_dp_1d

  !------------------------------------------------------
  FUNCTION p_sum_dp_2d (zfield, comm, root) RESULT (p_sum)

    REAL(dp),          INTENT(in) :: zfield(:,:)
    INTEGER, OPTIONAL, INTENT(in) :: comm, root
    REAL(dp)                      :: p_sum (SIZE(zfield,1),SIZE(zfield,2))

#ifndef NOMPI
    INTEGER :: p_comm, my_rank

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
      IF (PRESENT(root)) THEN
        CALL mpi_reduce(zfield, p_sum, SIZE(zfield), p_real_dp, &
             mpi_sum, root, p_comm, p_error)
        ! get local PE identification
        CALL mpi_comm_rank(p_comm, my_rank, p_error)
        ! do not use the result on all the other ranks:
        IF (root /= my_rank) p_sum = zfield
      ELSE
        CALL mpi_allreduce (zfield, p_sum, SIZE(zfield), p_real_dp, &
             mpi_sum, p_comm, p_error)
      END IF
    ELSE
       p_sum = zfield
    END IF
#else
    p_sum = zfield
#endif

  END FUNCTION p_sum_dp_2d

  !------------------------------------------------------
  FUNCTION p_sum_dp_3d (zfield, comm, root) RESULT (p_sum)

    REAL(dp),          INTENT(in) :: zfield(:,:,:)
    INTEGER, OPTIONAL, INTENT(in) :: comm, root
    REAL(dp)                      :: p_sum (SIZE(zfield,1),SIZE(zfield,2),SIZE(zfield,3))

#ifndef NOMPI
    INTEGER :: p_comm, my_rank

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
      IF (PRESENT(root)) THEN
        CALL mpi_reduce(zfield, p_sum, SIZE(zfield), p_real_dp, &
             mpi_sum, root, p_comm, p_error)
        ! get local PE identification
        CALL mpi_comm_rank(p_comm, my_rank, p_error)
        ! do not use the result on all the other ranks:
        IF (root /= my_rank) p_sum = zfield
      ELSE
        CALL mpi_allreduce (zfield, p_sum, SIZE(zfield), p_real_dp, &
             mpi_sum, p_comm, p_error)
      END IF
    ELSE
       p_sum = zfield
    END IF
#else
    p_sum = zfield
#endif

  END FUNCTION p_sum_dp_3d

  FUNCTION p_sum_i8_1d (kfield, comm) RESULT (p_sum)

    INTEGER(i8),       INTENT(in) :: kfield(:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    INTEGER(i8)                   :: p_sum (SIZE(kfield))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (kfield, p_sum, SIZE(kfield), p_int_i8, &
            mpi_sum, p_comm, p_error)
    ELSE
       p_sum = kfield
    END IF
#else
    p_sum = kfield
#endif

  END FUNCTION p_sum_i8_1d

  FUNCTION p_sum_i_1d (kfield, comm, root) RESULT (p_sum)

    INTEGER,       INTENT(in) :: kfield(:)
    INTEGER, OPTIONAL, INTENT(in) :: comm, root
    INTEGER                   :: p_sum (SIZE(kfield))

#ifndef NOMPI
    INTEGER :: p_comm, my_rank

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
      IF (PRESENT(root)) THEN
        CALL MPI_REDUCE(kfield, p_sum, SIZE(kfield), p_int, &
             mpi_sum, root, p_comm, p_error)
        ! get local PE identification
        CALL mpi_comm_rank(p_comm, my_rank, p_error)
        ! do not use the result on all the other ranks:
        IF (root /= my_rank) p_sum = kfield
      ELSE
        CALL MPI_ALLREDUCE (kfield, p_sum, SIZE(kfield), p_int, &
            mpi_sum, p_comm, p_error)
      END IF
    ELSE
       p_sum = kfield
    END IF
#else
    p_sum = kfield
#endif

  END FUNCTION p_sum_i_1d


  FUNCTION p_sum_i_0d (kfield, comm) RESULT (p_sum)

    INTEGER,       INTENT(in) :: kfield
    INTEGER, OPTIONAL, INTENT(in) :: comm
    INTEGER                   :: p_sum

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (kfield, p_sum, 1, p_int, &
            mpi_sum, p_comm, p_error)
    ELSE
       p_sum = kfield
    END IF
#else
    p_sum = kfield
#endif

  END FUNCTION p_sum_i_0d

  FUNCTION p_global_sum_1d (zfield, comm) RESULT (p_sum)

    REAL(dp),          INTENT(in) :: zfield(:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(dp)                      :: p_sum
    REAL(dp)                      :: pe_sums(SIZE(zfield))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_REDUCE (zfield, pe_sums, SIZE(zfield), p_real_dp, &
            mpi_sum, process_mpi_root_id, p_comm, p_error)
       p_sum = SUM(pe_sums)
    ELSE
       p_sum = SUM(zfield)
    END IF
#else
    p_sum = SUM(zfield)
#endif

  END FUNCTION p_global_sum_1d

  FUNCTION p_field_sum_1d (zfield, comm) RESULT (p_sum)

    REAL(dp),          INTENT(in) :: zfield(:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(dp)                      :: p_sum (SIZE(zfield))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_REDUCE (zfield, p_sum, SIZE(zfield), p_real_dp, &
            mpi_sum, process_mpi_root_id, p_comm, p_error)
       IF (.NOT. my_process_is_stdio()) p_sum = 0.0_dp
    ELSE
       p_sum = zfield
    END IF
#else
    p_sum = zfield
#endif

  END FUNCTION p_field_sum_1d

  FUNCTION p_field_sum_2d (zfield, comm) RESULT (p_sum)

    REAL(dp),          INTENT(in) :: zfield(:,:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(dp)                      :: p_sum (SIZE(zfield,1),SIZE(zfield,2))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_REDUCE (zfield, p_sum, SIZE(zfield), p_real_dp, &
            mpi_sum, process_mpi_root_id, p_comm, p_error)
       IF (.NOT. my_process_is_stdio()) p_sum = 0.0_dp
    ELSE
       p_sum = zfield
    END IF
#else
    p_sum = zfield
#endif

  END FUNCTION p_field_sum_2d

  !> common code of min/max reductions
  SUBROUTINE p_minmax_common(in_field, out_field, n, op, loc_op, &
       proc_id, keyval, comm, root)
    INTEGER, INTENT(in) :: n, op, loc_op
    REAL(dp), INTENT(in) :: in_field(n)
    REAL(dp), INTENT(out) :: out_field(n)

    INTEGER, OPTIONAL, INTENT(inout) :: proc_id(n)
    INTEGER, OPTIONAL, INTENT(inout) :: keyval(n)
    INTEGER, OPTIONAL, INTENT(in)    :: root
    INTEGER, OPTIONAL, INTENT(in)    :: comm

#ifndef NOMPI
    INTEGER  :: p_comm, rank, comm_size
    LOGICAL :: compute_ikey
    INTEGER, ALLOCATABLE  :: meta_info(:), ikey(:)
#ifndef SLOW_MPI_MAXMINLOC
    DOUBLE PRECISION, ALLOCATABLE :: val_loc(:,:,:)
#endif

    IF (PRESENT(comm)) THEN
       p_comm = comm
       comm_size = p_comm_size(comm)
    ELSE
       p_comm = process_mpi_all_comm
       comm_size = process_mpi_all_size
    ENDIF

    IF (comm_size > 1) THEN

      IF (PRESENT(proc_id) .OR. PRESENT(keyval)) THEN
        ! encode meta information
        ALLOCATE(meta_info(n), ikey(n))
        meta_info = 0
        IF (PRESENT(keyval))  meta_info = meta_info + keyval*comm_size
        IF (PRESENT(proc_id)) meta_info = meta_info + proc_id
        ! use mpi_minloc to transfer additional data
#ifndef SLOW_MPI_MAXMINLOC
        ALLOCATE(val_loc(2, n, 2))
#endif
        IF (PRESENT(root)) THEN
          CALL MPI_COMM_RANK(p_comm, rank, p_error)
#ifdef SLOW_MPI_MAXMINLOC
          ! on BG/Q, {max|min}loc is slow
          CALL mpi_allreduce(in_field, out_field, n, mpi_double_precision, &
               op, p_comm, p_error)
          ikey = MERGE(meta_info, HUGE(1), in_field == out_field)
          CALL mpi_reduce(ikey, meta_info, n, mpi_integer, &
               mpi_min, root, p_comm, p_error)
#else
          val_loc(1, :, 1) = DBLE(in_field)
          val_loc(2, :, 1) = DBLE(meta_info)
          CALL mpi_reduce(val_loc(:, :, 1), val_loc(:, :, 2), &
               n, mpi_2double_precision, loc_op, root, p_comm, p_error)
          IF (rank == root) THEN
             out_field = val_loc(1, :, 2)
          ELSE
             out_field = 0.
          END IF
#endif
          compute_ikey = rank == root
        ELSE
#ifdef SLOW_MPI_MAXMINLOC
          CALL mpi_allreduce(in_field, out_field, n, mpi_double_precision, &
               op, p_comm, p_error)
          ikey = MERGE(meta_info, HUGE(1), in_field == out_field)
          CALL mpi_allreduce(ikey, meta_info, n, mpi_integer, &
               mpi_min, p_comm, p_error)
#else
          val_loc(1, :, 1) = DBLE(in_field)
          val_loc(2, :, 1) = DBLE(meta_info)
          CALL mpi_allreduce(val_loc(:, :, 1), val_loc(:, :, 2), &
               n, mpi_2double_precision, loc_op, p_comm, p_error)
          out_field = val_loc(1, :, 2)
#endif
          compute_ikey = .TRUE.
        END IF
        ! decode meta info:
        IF (compute_ikey) THEN
#ifdef SLOW_MPI_MAXMINLOC
          ikey = meta_info / comm_size
          IF (PRESENT(proc_id)) proc_id = mod(meta_info,comm_size)
#else
          ikey = NINT(val_loc(2, :, 2)) / comm_size
          IF (PRESENT(proc_id)) proc_id = mod(nint(val_loc(2, :, 2)),comm_size)
#endif
          IF (PRESENT(keyval)) keyval = ikey
        END IF
      ELSE
        ! compute simple (standard) minimum
        IF (PRESENT(root)) THEN
          CALL mpi_reduce(in_field, out_field, n, p_real_dp, &
               op, root, p_comm, p_error)
        ELSE
          CALL mpi_allreduce(in_field, out_field, n, p_real_dp, &
               op, p_comm, p_error)
        END IF
     END IF
    ELSE
      out_field = in_field
    END IF
#else
    out_field = in_field
#endif
  END SUBROUTINE p_minmax_common

  SUBROUTINE p_minmax_common_sp(in_field, out_field, n, op, loc_op, &
       proc_id, keyval, comm, root)
    INTEGER, INTENT(in) :: n, op, loc_op
    REAL(sp), INTENT(in) :: in_field(n)
    REAL(sp), INTENT(out) :: out_field(n)

    INTEGER, OPTIONAL, INTENT(inout) :: proc_id(n)
    INTEGER, OPTIONAL, INTENT(inout) :: keyval(n)
    INTEGER, OPTIONAL, INTENT(in)    :: root
    INTEGER, OPTIONAL, INTENT(in)    :: comm

#ifndef NOMPI
    INTEGER  :: p_comm, rank, comm_size
    LOGICAL :: compute_ikey
    INTEGER, ALLOCATABLE  :: meta_info(:), ikey(:)
#ifndef SLOW_MPI_MAXMINLOC
    DOUBLE PRECISION, ALLOCATABLE :: val_loc(:,:,:)
#endif

    IF (PRESENT(comm)) THEN
       p_comm = comm
       comm_size = p_comm_size(comm)
    ELSE
       p_comm = process_mpi_all_comm
       comm_size = process_mpi_all_size
    ENDIF

    IF (comm_size > 1) THEN

      IF (PRESENT(proc_id) .OR. PRESENT(keyval)) THEN
        ! encode meta information
        ALLOCATE(meta_info(n), ikey(n))
        meta_info = 0
        IF (PRESENT(keyval))  meta_info = meta_info + keyval*comm_size
        IF (PRESENT(proc_id)) meta_info = meta_info + proc_id
        ! use mpi_minloc to transfer additional data
#ifndef SLOW_MPI_MAXMINLOC
        ALLOCATE(val_loc(2, n, 2))
#endif
        IF (PRESENT(root)) THEN
          CALL MPI_COMM_RANK(p_comm, rank, p_error)
#ifdef SLOW_MPI_MAXMINLOC
          ! on BG/Q, {max|min}loc is slow
          CALL mpi_allreduce(in_field, out_field, n, mpi_double_precision, &
               op, p_comm, p_error)
          ikey = MERGE(meta_info, HUGE(1), in_field == out_field)
          CALL mpi_reduce(ikey, meta_info, n, mpi_integer, &
               p_min_op(), root, p_comm, p_error)
#else
          val_loc(1, :, 1) = DBLE(in_field)
          val_loc(2, :, 1) = DBLE(meta_info)
          CALL mpi_reduce(val_loc(:, :, 1), val_loc(:, :, 2), &
               n, mpi_2double_precision, loc_op, root, p_comm, p_error)
          IF (rank == root) THEN
             out_field = val_loc(1, :, 2)
          ELSE
             out_field = 0.
          END IF
#endif
          compute_ikey = rank == root
        ELSE
#ifdef SLOW_MPI_MAXMINLOC
          CALL mpi_allreduce(in_field, out_field, n, mpi_double_precision, &
               op, p_comm, p_error)
          ikey = MERGE(meta_info, HUGE(1), in_field == out_field)
          CALL mpi_allreduce(ikey, meta_info, n, mpi_integer, &
               p_min_op(), p_comm, p_error)
#else
          val_loc(1, :, 1) = DBLE(in_field)
          val_loc(2, :, 1) = DBLE(meta_info)
          CALL mpi_allreduce(val_loc(:, :, 1), val_loc(:, :, 2), &
               n, mpi_2double_precision, loc_op, p_comm, p_error)
          out_field = val_loc(1, :, 2)
#endif
          compute_ikey = .TRUE.
        END IF
        ! decode meta info:
        IF (compute_ikey) THEN
#ifdef SLOW_MPI_MAXMINLOC
          ikey = meta_info / comm_size
          IF (PRESENT(proc_id)) proc_id = mod(meta_info,comm_size)
#else
          ikey = NINT(val_loc(2, :, 2)) / comm_size
          IF (PRESENT(proc_id)) proc_id = mod(nint(val_loc(2, :, 2)),comm_size)
#endif
          IF (PRESENT(keyval)) keyval = ikey
        END IF
      ELSE
        ! compute simple (standard) minimum
        IF (PRESENT(root)) THEN
          CALL mpi_reduce(in_field, out_field, n, p_real_dp, &
               op, root, p_comm, p_error)
        ELSE
          CALL mpi_allreduce(in_field, out_field, n, p_real_dp, &
               op, p_comm, p_error)
        END IF
     END IF
    ELSE
      out_field = in_field
    END IF
#else
    out_field = in_field
#endif
  END SUBROUTINE p_minmax_common_sp


  !> computes a global maximum of real numbers
  !
  ! @param[out]   proc_id  (Optional:) PE number of maximum value
  ! @param[inout] keyval   (Optional:) additional meta information
  ! @param[in]    root     (Optional:) root PE, otherwise we perform an
  !                                    ALL-TO-ALL operation
  !
  ! The parameter @p keyval can be used to communicate
  ! additional data on the maximum value, e.g., the level
  ! index where the maximum occurred.
  !
  FUNCTION p_max_0d (zfield, proc_id, keyval, comm, root) RESULT (p_max)

    REAL(dp)                         :: p_max
    REAL(dp),          INTENT(in)    :: zfield
    INTEGER, OPTIONAL, INTENT(inout) :: proc_id
    INTEGER, OPTIONAL, INTENT(inout) :: keyval
    INTEGER, OPTIONAL, INTENT(in)    :: root
    INTEGER, OPTIONAL, INTENT(in)    :: comm

    REAL(dp) :: temp_in(1), temp_out(1)
    INTEGER :: temp_keyval(1), temp_proc_id(1)
    temp_in(1) = zfield
    IF (PRESENT(proc_id) .AND. PRESENT(keyval)) THEN
      temp_keyval(1) = keyval; temp_proc_id(1) = proc_id
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_max, mpi_maxloc, &
           proc_id=temp_proc_id, keyval=temp_keyval, comm=comm, root=root)
      keyval = temp_keyval(1); proc_id = temp_proc_id(1)
    ELSE IF (PRESENT(proc_id)) THEN
      temp_proc_id(1) = proc_id
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_max, mpi_maxloc, &
           proc_id=temp_proc_id, comm=comm, root=root)
      proc_id = temp_proc_id(1)
    ELSE IF (PRESENT(keyval)) THEN
      temp_keyval(1) = keyval
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_max, mpi_maxloc, &
         keyval=temp_keyval, comm=comm, root=root)
      keyval = temp_keyval(1)
    ELSE ! .not. present(keyval) .and. .not. present(proc_id)
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_max, mpi_maxloc, &
           comm=comm, root=root)
    END IF

    p_max = temp_out(1)

  END FUNCTION p_max_0d

  FUNCTION p_max_0d_sp (zfield, proc_id, keyval, comm, root) RESULT (p_max)

    REAL(sp)                         :: p_max
    REAL(sp),          INTENT(in)    :: zfield
    INTEGER, OPTIONAL, INTENT(inout) :: proc_id
    INTEGER, OPTIONAL, INTENT(inout) :: keyval
    INTEGER, OPTIONAL, INTENT(in)    :: root
    INTEGER, OPTIONAL, INTENT(in)    :: comm

    REAL(dp) :: temp_in(1), temp_out(1)
    INTEGER :: temp_keyval(1), temp_proc_id(1)
    temp_in(1) = zfield
    IF (PRESENT(proc_id) .AND. PRESENT(keyval)) THEN
      temp_keyval(1) = keyval; temp_proc_id(1) = proc_id
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_max, mpi_maxloc, &
           proc_id=temp_proc_id, keyval=temp_keyval, comm=comm, root=root)
      keyval = temp_keyval(1); proc_id = temp_proc_id(1)
    ELSE IF (PRESENT(proc_id)) THEN
      temp_proc_id(1) = proc_id
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_max, mpi_maxloc, &
           proc_id=temp_proc_id, comm=comm, root=root)
      proc_id = temp_proc_id(1)
    ELSE IF (PRESENT(keyval)) THEN
      temp_keyval(1) = keyval
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_max, mpi_maxloc, &
         keyval=temp_keyval, comm=comm, root=root)
      keyval = temp_keyval(1)
    ELSE ! .not. present(keyval) .and. .not. present(proc_id)
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_max, mpi_maxloc, &
           comm=comm, root=root)
    END IF

    p_max = temp_out(1)

  END FUNCTION p_max_0d_sp

  FUNCTION p_max_int_0d (zfield, comm) RESULT (p_max)

    INTEGER                          :: p_max
    INTEGER,           INTENT(in)    :: zfield
    INTEGER, OPTIONAL, INTENT(in)    :: comm
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (zfield, p_max, 1, p_int, &
            mpi_max, p_comm, p_error)
    ELSE
       p_max = zfield
    END IF
#else
    p_max = zfield
#endif

  END FUNCTION p_max_int_0d

  !> computes a global maximum of a real number array
  !
  ! @param[out]   proc_id  (Optional:) PE number of maximum value
  ! @param[inout] keyval   (Optional:) additional meta information
  ! @param[in]    root     (Optional:) root PE, otherwise we perform an
  !                                    ALL-TO-ALL operation
  !
  ! The parameter @p keyval can be used to communicate
  ! additional data on the maximum value, e.g., the level
  ! index where the maximum occurred.
  !
  FUNCTION p_max_1d (zfield, proc_id, keyval, comm, root) RESULT (p_max)

    REAL(dp),          INTENT(in)    :: zfield(:)
    INTEGER, OPTIONAL, INTENT(inout) :: proc_id(SIZE(zfield))
    INTEGER, OPTIONAL, INTENT(inout) :: keyval(SIZE(zfield))
    INTEGER, OPTIONAL, INTENT(in)    :: root
    INTEGER, OPTIONAL, INTENT(in)    :: comm
    REAL(dp)                         :: p_max (SIZE(zfield))

    CALL p_minmax_common(zfield, p_max, SIZE(zfield), mpi_max, mpi_maxloc, &
           proc_id=proc_id, keyval=keyval, comm=comm, root=root)

  END FUNCTION p_max_1d

  FUNCTION p_max_1d_sp (zfield, proc_id, keyval, comm, root) RESULT (p_max)

    REAL(sp),          INTENT(in)    :: zfield(:)
    INTEGER, OPTIONAL, INTENT(inout) :: proc_id(SIZE(zfield))
    INTEGER, OPTIONAL, INTENT(inout) :: keyval(SIZE(zfield))
    INTEGER, OPTIONAL, INTENT(in)    :: root
    INTEGER, OPTIONAL, INTENT(in)    :: comm
    REAL(sp)                         :: p_max (SIZE(zfield))

    CALL p_minmax_common(zfield, p_max, SIZE(zfield), mpi_max, mpi_maxloc, &
           proc_id=proc_id, keyval=keyval, comm=comm, root=root)

  END FUNCTION p_max_1d_sp

  ! Computes maximum of a 1D field of integers.
  !
  ! @param[in] root Optional root PE, otherwise we perform an
  !                 ALL-TO-ALL operation
  FUNCTION p_max_int_1d (zfield, comm, root) RESULT (p_max)

    INTEGER,           INTENT(in) :: zfield(:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    INTEGER, OPTIONAL, INTENT(in) :: root
    INTEGER                       :: p_max (SIZE(zfield))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
      IF (PRESENT(root)) THEN
        CALL MPI_REDUCE (zfield, p_max, SIZE(zfield), p_int, &
          mpi_max, root, p_comm, p_error)
      ELSE
        CALL MPI_ALLREDUCE (zfield, p_max, SIZE(zfield), p_int, &
          mpi_max, p_comm, p_error)
      END IF ! if present(root)
    ELSE
       p_max = zfield
    END IF
#else
    p_max = zfield
#endif

  END FUNCTION p_max_int_1d

  FUNCTION p_max_2d (zfield, comm) RESULT (p_max)

    REAL(dp),          INTENT(in) :: zfield(:,:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(dp)                      :: p_max (SIZE(zfield,1),SIZE(zfield,2))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (zfield, p_max, SIZE(zfield), p_real_dp, &
            mpi_max, p_comm, p_error)
    ELSE
       p_max = zfield
    END IF
#else
    p_max = zfield
#endif

  END FUNCTION p_max_2d

  FUNCTION p_max_2d_sp (zfield, comm) RESULT (p_max)

    REAL(sp),          INTENT(in) :: zfield(:,:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(sp)                      :: p_max (SIZE(zfield,1),SIZE(zfield,2))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (zfield, p_max, SIZE(zfield), p_real_dp, &
            mpi_max, p_comm, p_error)
    ELSE
       p_max = zfield
    END IF
#else
    p_max = zfield
#endif

  END FUNCTION p_max_2d_sp

  FUNCTION p_max_3d (zfield, comm) RESULT (p_max)

    REAL(dp),          INTENT(in) :: zfield(:,:,:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(dp)                      :: p_max (SIZE(zfield,1),SIZE(zfield,2)&
                                           ,SIZE(zfield,3))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (zfield, p_max, SIZE(zfield), p_real_dp, &
            mpi_max, p_comm, p_error)
    ELSE
       p_max = zfield
    END IF
#else
    p_max = zfield
#endif

  END FUNCTION p_max_3d

  FUNCTION p_max_3d_sp (zfield, comm) RESULT (p_max)

    REAL(sp),          INTENT(in) :: zfield(:,:,:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(sp)                      :: p_max (SIZE(zfield,1),SIZE(zfield,2)&
                                           ,SIZE(zfield,3))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (zfield, p_max, SIZE(zfield), p_real_dp, &
            mpi_max, p_comm, p_error)
    ELSE
       p_max = zfield
    END IF
#else
    p_max = zfield
#endif

  END FUNCTION p_max_3d_sp


  !> computes a global minimum of real numbers
  !
  ! @param[out]   proc_id  (Optional:) PE number of maximum value
  ! @param[inout] keyval   (Optional:) additional meta information
  ! @param[in]    root     (Optional:) root PE, otherwise we perform an
  !                                    ALL-TO-ALL operation
  !
  ! The parameter @p keyval can be used to communicate
  ! additional data on the maximum value, e.g., the level
  ! index where the maximum occurred.
  !
  FUNCTION p_min_0d (zfield, proc_id, keyval, comm, root) RESULT (p_min)

    REAL(dp)                         :: p_min
    REAL(dp),          INTENT(in)    :: zfield
    INTEGER, OPTIONAL, INTENT(inout) :: proc_id
    INTEGER, OPTIONAL, INTENT(inout) :: keyval
    INTEGER, OPTIONAL, INTENT(in)    :: root
    INTEGER, OPTIONAL, INTENT(in)    :: comm

    REAL(dp) :: temp_in(1), temp_out(1)
    INTEGER :: temp_keyval(1), temp_proc_id(1)
    temp_in(1) = zfield
    IF (PRESENT(proc_id) .AND. PRESENT(keyval)) THEN
      temp_keyval(1) = keyval; temp_proc_id(1) = proc_id
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_min, mpi_minloc, &
           proc_id=temp_proc_id, keyval=temp_keyval, comm=comm, root=root)
      keyval = temp_keyval(1); proc_id = temp_proc_id(1)
    ELSE IF (PRESENT(proc_id)) THEN
      temp_proc_id(1) = proc_id
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_min, mpi_minloc, &
           proc_id=temp_proc_id, comm=comm, root=root)
      proc_id = temp_proc_id(1)
    ELSE IF (PRESENT(keyval)) THEN
      temp_keyval(1) = keyval
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_min, mpi_minloc, &
         keyval=temp_keyval, comm=comm, root=root)
      keyval = temp_keyval(1)
    ELSE ! .not. present(keyval) .and. .not. present(proc_id)
      CALL p_minmax_common(temp_in, temp_out, 1, mpi_min, mpi_minloc, &
           comm=comm, root=root)
    END IF

    p_min = temp_out(1)

  END FUNCTION p_min_0d

  FUNCTION p_min_int_0d (zfield, comm) RESULT (p_min)

    INTEGER,           INTENT(in) :: zfield
    INTEGER, OPTIONAL, INTENT(in) :: comm
    INTEGER                       :: p_min
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (zfield, p_min, 1, p_int, &
            mpi_min, p_comm, p_error)
    ELSE
       p_min = zfield
    END IF
#else
    p_min = zfield
#endif

  END FUNCTION p_min_int_0d

  FUNCTION p_min_1d (zfield, proc_id, keyval, comm, root) RESULT (p_min)

    REAL(dp),          INTENT(in) :: zfield(:)
    INTEGER, OPTIONAL, INTENT(inout) :: proc_id(SIZE(zfield))
    INTEGER, OPTIONAL, INTENT(inout) :: keyval(SIZE(zfield))
    INTEGER, OPTIONAL, INTENT(in) :: root
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(dp)                      :: p_min (SIZE(zfield))

    CALL p_minmax_common(zfield, p_min, SIZE(zfield), mpi_min, mpi_minloc, &
           proc_id=proc_id, keyval=keyval, comm=comm, root=root)

  END FUNCTION p_min_1d

  FUNCTION p_min_int_1d (zfield, comm) RESULT (p_min)

    INTEGER,           INTENT(in) :: zfield(:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    INTEGER                       :: p_min (SIZE(zfield))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (zfield, p_min, SIZE(zfield), p_int, &
            mpi_min, p_comm, p_error)
    ELSE
       p_min = zfield
    END IF
#else
    p_min = zfield
#endif

  END FUNCTION p_min_int_1d

  FUNCTION p_min_2d (zfield, comm) RESULT (p_min)

    REAL(dp),          INTENT(in) :: zfield(:,:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(dp)                      :: p_min (SIZE(zfield,1),SIZE(zfield,2))

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (zfield, p_min, SIZE(zfield), p_real_dp, &
            mpi_min, p_comm, p_error)
    ELSE
       p_min = zfield
    END IF
#else
    p_min = zfield
#endif

  END FUNCTION p_min_2d

  FUNCTION p_min_3d (zfield, comm) RESULT (p_min)

    REAL(dp),          INTENT(in) :: zfield(:,:,:)
    INTEGER, OPTIONAL, INTENT(in) :: comm
    REAL(dp)                      :: p_min (SIZE(zfield,1),SIZE(zfield,2)&
                                           ,SIZE(zfield,3))
#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    IF (my_process_is_mpi_all_parallel()) THEN
       CALL MPI_ALLREDUCE (zfield, p_min, SIZE(zfield), p_real_dp, &
            mpi_min, p_comm, p_error)
    ELSE
       p_min = zfield
    END IF
#else
    p_min = zfield
#endif

  END FUNCTION p_min_3d

  FUNCTION p_lor_0d(zfield, comm) RESULT(res)
    LOGICAL :: res
    LOGICAL, INTENT(in) :: zfield
    INTEGER, OPTIONAL, INTENT(in) :: comm
#ifndef NOMPI
    INTEGER :: pcomm
    IF (PRESENT(comm)) THEN
       pcomm = comm
    ELSE
       pcomm = process_mpi_all_comm
    ENDIF
    IF (my_process_is_mpi_all_parallel()) THEN
      CALL mpi_allreduce(zfield, res, 1, p_bool, mpi_lor, pcomm, p_error)
    ELSE
      res = zfield
    END IF
#else
    res = zfield
#endif
  END FUNCTION p_lor_0d

  ! Computes maximum of a 1D field of integers.
  !
  ! @param[in] root Optional root PE, otherwise we perform an
  !                 ALL-TO-ALL operation
  SUBROUTINE p_allreduce_max_int_1d (zfield, comm, root)
    INTEGER, INTENT(inout) :: zfield(:)
    INTEGER, INTENT(in) :: comm
    INTEGER, INTENT(in), OPTIONAL :: root

#if !defined(NOMPI)
    INTEGER :: p_comm, p_error

    p_comm = comm
    IF (PRESENT(root)) THEN
      CALL MPI_REDUCE (MPI_IN_PLACE, zfield, SIZE(zfield), p_int, &
        mpi_max, root, p_comm, p_error)
    ELSE
      CALL MPI_ALLREDUCE (MPI_IN_PLACE, zfield, SIZE(zfield), p_int, &
        mpi_max, p_comm, p_error)
    END IF ! if present(root)
#endif
  END SUBROUTINE p_allreduce_max_int_1d

  FUNCTION p_reduce_i8_0d(send, op, root, comm) RESULT(recv)
    INTEGER(i8) :: recv
    INTEGER(i8), INTENT(IN) :: send
    INTEGER, INTENT(in) :: op, root, comm

#ifndef NOMPI
    INTEGER :: ierror
    CHARACTER(*), PARAMETER :: routine = modname//":p_reduce_i8_0d"

    CALL MPI_Reduce(send, recv, 1, p_int_i8, op, root, comm, ierror)
    IF (ierror /= MPI_SUCCESS) CALL finish(routine, "error in MPI call.")
#else
    recv = send
#endif
  END FUNCTION p_reduce_i8_0d

  FUNCTION p_reduce_i4_0d(send, op, root, comm) &
       RESULT(recv)
    INTEGER(i4) :: recv
    INTEGER(i4), INTENT(IN) :: send
    INTEGER, INTENT(in) :: op, root, comm

#ifndef NOMPI
    INTEGER :: ierror
    CHARACTER(*), PARAMETER :: routine = modname//":p_reduce_i4_0d"

    CALL MPI_Reduce(send, recv, 1, p_int_i4, op, root, comm, ierror)
    IF (ierror /= MPI_SUCCESS) CALL finish(routine, "error in MPI call.")
#else
    recv = send
#endif
  END FUNCTION p_reduce_i4_0d

  INTEGER(i4) FUNCTION p_allreduce_int4_0d(input, reductionOp, comm) RESULT(resultVar)
    INTEGER(i4), INTENT(IN) :: input
    INTEGER, INTENT(in) :: reductionOp, comm

#ifndef NOMPI
    INTEGER :: error
    CHARACTER(*), PARAMETER :: routine = modname//":p_allreduce_int4_0d"

    CALL MPI_Allreduce(input, resultVar, 1, p_int_i4, reductionOp, comm, error)
    IF(error /= MPI_SUCCESS) CALL finish(routine, "error in MPI call.")
#else
    resultVar = input
#endif
  END FUNCTION p_allreduce_int4_0d

  FUNCTION p_allreduce_bool_0d(input, reductionOp, comm) &
       RESULT(res)
    LOGICAL :: res
    LOGICAL, INTENT(IN) :: input
    INTEGER, INTENT(in) :: reductionOp, comm

#ifndef NOMPI
    INTEGER :: ierror
    CHARACTER(*), PARAMETER :: routine = modname//":p_allreduce_bool_0d"

    CALL mpi_allreduce(input, res, 1, mpi_logical, reductionOp, comm, ierror)
    IF (ierror /= MPI_SUCCESS) CALL finish(routine, "error in mpi_allreduce.")
#else
    res = input
#endif
  END FUNCTION p_allreduce_bool_0d

  INTEGER(i8) FUNCTION p_allreduce_int8_0d(input, reductionOp, comm) RESULT(resultVar)
    INTEGER(i8), INTENT(IN) :: input
    INTEGER, INTENT(in) :: reductionOp, comm

#ifndef NOMPI
    INTEGER :: error
    CHARACTER(*), PARAMETER :: routine = modname//":p_allreduce_int8_0d"

    CALL MPI_Allreduce(input, resultVar, 1, p_int_i8, reductionOp, comm, error)
    IF(error /= MPI_SUCCESS) CALL finish(routine, "error in MPI call.)")
#else
    resultVar = input
#endif
  END FUNCTION p_allreduce_int8_0d


  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Scatter
  !---------------------------------------------------------------------------------------------------------------------------------
  SUBROUTINE p_scatter_real_1d1d(sendbuf, recvbuf, p_src, comm)
    REAL(dp),          INTENT(inout) :: sendbuf(:), recvbuf(:)
    INTEGER,           INTENT(in) :: p_src
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    CHARACTER(*), PARAMETER :: routine = modname//"::p_scatter_real_1d1d"
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_Scatter(sendbuf, SIZE(recvbuf), p_real_dp, &
    &                recvbuf, SIZE(recvbuf), p_real_dp, &
    &                p_src, p_comm, p_error)
    IF(p_error /= MPI_SUCCESS) CALL finish(routine, 'Error in MPI_Scatter operation!')
#else
     recvbuf = sendbuf
#endif
   END SUBROUTINE p_scatter_real_1d1d


   SUBROUTINE p_scatter_real_2d1d(sendbuf, recvbuf, p_src, comm)
    REAL(dp),          INTENT(inout) :: sendbuf(:,:), recvbuf(:)
    INTEGER,           INTENT(in) :: p_src
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    CHARACTER(*), PARAMETER :: routine = modname//"::p_scatter_real_1d1d"
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_Scatter(sendbuf, SIZE(recvbuf), p_real_dp, &
    &                recvbuf, SIZE(recvbuf), p_real_dp, &
    &                p_src, p_comm, p_error)
    IF(p_error /= MPI_SUCCESS) CALL finish(routine, 'Error in MPI_Scatter operation!')
#else
    recvbuf = sendbuf(:,1)
#endif
  END SUBROUTINE p_scatter_real_2d1d

  SUBROUTINE p_scatter_sp_1d1d(sendbuf, recvbuf, p_src, comm)
    REAL(sp),          INTENT(inout) :: sendbuf(:), recvbuf(:)
    INTEGER,           INTENT(in) :: p_src
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    CHARACTER(*), PARAMETER :: routine = modname//"::p_scatter_single_1d1d"
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_Scatter(sendbuf, SIZE(recvbuf), p_real_sp, &
    &                recvbuf, SIZE(recvbuf), p_real_sp, &
    &                p_src, p_comm, p_error)
    IF(p_error /= MPI_SUCCESS) CALL finish(routine, 'Error in MPI_Scatter operation!')
#else
     recvbuf = sendbuf
#endif
   END SUBROUTINE p_scatter_sp_1d1d

  SUBROUTINE p_scatter_sp_2d1d(sendbuf, recvbuf, p_src, comm)
    REAL(sp),          INTENT(inout) :: sendbuf(:,:), recvbuf(:)
    INTEGER,           INTENT(in) :: p_src
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    CHARACTER(*), PARAMETER :: routine = modname//"::p_scatter_real_1d1d"
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_Scatter(sendbuf, SIZE(recvbuf), p_real_sp, &
    &                recvbuf, SIZE(recvbuf), p_real_sp, &
    &                p_src, p_comm, p_error)
    IF(p_error /= MPI_SUCCESS) CALL finish(routine, 'Error in MPI_Scatter operation!')
#else
    recvbuf = sendbuf(:,1)
#endif
  END SUBROUTINE p_scatter_sp_2d1d

  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Scatter
  !---------------------------------------------------------------------------------------------------------------------------------
  SUBROUTINE p_scatter_int_1d1d(sendbuf, recvbuf, p_src, comm)
    INTEGER,           INTENT(inout) :: sendbuf(:), recvbuf(:)
    INTEGER,           INTENT(in) :: p_src
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    CHARACTER(*), PARAMETER :: routine = modname//"::p_scatter_int_1d1d"
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_Scatter(sendbuf, SIZE(recvbuf), p_int, &
    &                recvbuf, SIZE(recvbuf), p_int, &
    &                p_src, p_comm, p_error)
    IF(p_error /= MPI_SUCCESS) CALL finish(routine, 'Error in MPI_Scatter operation!')
#else
     recvbuf = sendbuf
#endif
  END SUBROUTINE p_scatter_int_1d1d

  SUBROUTINE p_scatter_int_2d1d(sendbuf, recvbuf, p_src, comm)
    INTEGER,           INTENT(inout) :: sendbuf(:,:), recvbuf(:)
    INTEGER,           INTENT(in) :: p_src
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    CHARACTER(*), PARAMETER :: routine = modname//"::p_scatter_int_2d1d"
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL MPI_Scatter(sendbuf, SIZE(recvbuf), p_int, &
    &                recvbuf, SIZE(recvbuf), p_int, &
    &                p_src, p_comm, p_error)
    IF(p_error /= MPI_SUCCESS) CALL finish(routine, 'Error in MPI_Scatter operation!')
#else
     recvbuf = sendbuf(:,1)
#endif
  END SUBROUTINE p_scatter_int_2d1d

  SUBROUTINE p_gather_real_0d1d (sendbuf, recvbuf, p_dest, comm)
    REAL(dp),          INTENT(in   ) :: sendbuf
    REAL(dp),          INTENT(inout) :: recvbuf(:)
    INTEGER,           INTENT(in   ) :: p_dest
    INTEGER, OPTIONAL, INTENT(in   ) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

     CALL MPI_GATHER(sendbuf, 1, p_real_dp, &
                     recvbuf, 1, p_real_dp, &
                     p_dest, p_comm, p_error)
#else
     recvbuf(:) = sendbuf
#endif
  END SUBROUTINE p_gather_real_0d1d

  SUBROUTINE p_gather_real_1d2d (sendbuf, recvbuf, p_dest, comm)
    REAL(dp),          INTENT(in   ) :: sendbuf(:)
    REAL(dp),          INTENT(inout) :: recvbuf(:,:)
    INTEGER,           INTENT(in   ) :: p_dest
    INTEGER, OPTIONAL, INTENT(in   ) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

     CALL MPI_GATHER(sendbuf, SIZE(sendbuf), p_real_dp, &
                     recvbuf, SIZE(sendbuf), p_real_dp, &
                     p_dest, p_comm, p_error)
#else
     recvbuf(:,1) = sendbuf(:)
#endif
  END SUBROUTINE p_gather_real_1d2d

  SUBROUTINE p_gather_real_2d3d(sendbuf, recvbuf, p_dest, comm)
    REAL(dp),          INTENT(in   ) :: sendbuf(:,:)
    REAL(dp),          INTENT(inout) :: recvbuf(:,:,:)
    INTEGER,           INTENT(in   ) :: p_dest
    INTEGER, OPTIONAL, INTENT(in   ) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL mpi_gather(sendbuf, SIZE(sendbuf), p_real_dp, &
                    recvbuf, SIZE(sendbuf), p_real_dp, &
                    p_dest, p_comm, p_error)
#else
    recvbuf(:,:,1) = sendbuf(:,:)
#endif
  END SUBROUTINE p_gather_real_2d3d

   SUBROUTINE p_gather_real_5d6d (sendbuf, recvbuf, p_dest, comm)
     REAL(dp),          INTENT(in   ) :: sendbuf(:,:,:,:,:)
     REAL(dp),          INTENT(inout) :: recvbuf(:,:,:,:,:,:)
     INTEGER,           INTENT(in   ) :: p_dest
     INTEGER, OPTIONAL, INTENT(in   ) :: comm

#ifndef NOMPI
     INTEGER :: p_comm

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF

     CALL MPI_GATHER(sendbuf, SIZE(sendbuf), p_real_dp, &
                     recvbuf, SIZE(sendbuf), p_real_dp, &
                     p_dest, p_comm, p_error)

     IF (p_error /= MPI_SUCCESS) THEN
       CALL finish('p_gather_real_5d6d', message_text)
     END IF

#else
     recvbuf(:,:,:,:,:,LBOUND(recvbuf,6)) = sendbuf(:,:,:,:,:)
#endif
   END SUBROUTINE p_gather_real_5d6d


  SUBROUTINE p_gather_real_1d1d (sendbuf, recvbuf, p_dest, comm)
    REAL(dp),          INTENT(in   ) :: sendbuf(:)
    REAL(dp),          INTENT(inout) :: recvbuf(:)
    INTEGER,           INTENT(in   ) :: p_dest
    INTEGER, OPTIONAL, INTENT(in   ) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

     CALL MPI_GATHER(sendbuf, SIZE(sendbuf), p_real_dp, &
                     recvbuf, SIZE(sendbuf), p_real_dp, &
                     p_dest, p_comm, p_error)
#else
     recvbuf(:) = sendbuf(:)
#endif
   END SUBROUTINE p_gather_real_1d1d


  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Gather()
  !---------------------------------------------------------------------------------------------------------------------------------
   SUBROUTINE p_gather_int_0d1d (sendbuf, recvbuf, p_dest, comm)
     INTEGER,           INTENT(in   ) :: sendbuf
     INTEGER,           INTENT(inout) :: recvbuf(:)
     INTEGER,           INTENT(in   ) :: p_dest
     INTEGER, OPTIONAL, INTENT(in   ) :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_gather_int_0d1d"
     INTEGER :: p_comm

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF

     CALL MPI_GATHER(sendbuf, 1, MPI_INTEGER, &
       &             recvbuf, 1, MPI_INTEGER, &
       &             p_dest, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_GATHER operation!')
#else
     recvbuf = sendbuf
#endif
   END SUBROUTINE p_gather_int_0d1d


  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Gather()
  !---------------------------------------------------------------------------------------------------------------------------------
  SUBROUTINE p_gather_int_1d1d (sendbuf, recvbuf, p_dest, comm)
     INTEGER,           INTENT(in   ) :: sendbuf(:)
     INTEGER,           INTENT(inout) :: recvbuf(:)
     INTEGER,           INTENT(in   ) :: p_dest
     INTEGER, OPTIONAL, INTENT(in   ) :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_gather_int_1d1d"
     INTEGER :: p_comm

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF

     CALL MPI_GATHER(sendbuf, SIZE(sendbuf), MPI_INTEGER, &
       &             recvbuf, SIZE(sendbuf), MPI_INTEGER, &
       &             p_dest, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_GATHER operation!')
#else
     recvbuf = sendbuf
#endif
  END SUBROUTINE p_gather_int_1d1d

  SUBROUTINE p_gather_int_1d2d(sendbuf, recvbuf, p_dest, comm)
    INTEGER,           INTENT(inout) :: recvbuf(:,:)
    INTEGER,           INTENT(in) :: p_dest, sendbuf(:)
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL mpi_gather(sendbuf, SIZE(sendbuf), mpi_integer, &
      &             recvbuf, SIZE(sendbuf), mpi_integer, &
      &             p_dest, p_comm, p_error)
#else
     recvbuf(:,1) = sendbuf(:)
#endif
   END SUBROUTINE p_gather_int_1d2d

  SUBROUTINE p_gather_int_2d3d(sendbuf, recvbuf, p_dest, comm)
    INTEGER,           INTENT(in) :: sendbuf(:,:)
    INTEGER,        INTENT(inout) :: recvbuf(:,:,:)
    INTEGER,           INTENT(in) :: p_dest
    INTEGER, OPTIONAL, INTENT(in) :: comm

#ifndef NOMPI
    INTEGER :: p_comm

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    CALL mpi_gather(sendbuf, SIZE(sendbuf), mpi_integer, &
      &             recvbuf, SIZE(sendbuf), mpi_integer, &
      &             p_dest, p_comm, p_error)
#else
     recvbuf(:,:,1) = sendbuf(:,:)
#endif
   END SUBROUTINE p_gather_int_2d3d

  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Gather()
  !---------------------------------------------------------------------------------------------------------------------------------
   SUBROUTINE p_gather_char_0d1d (sbuf, recvbuf, p_dest, comm)
     CHARACTER(len=*),  INTENT(in)    ::  sbuf
     CHARACTER(len=*),  INTENT(inout) ::  recvbuf(:)
     INTEGER,           INTENT(in) :: p_dest
     INTEGER, OPTIONAL, INTENT(in) :: comm
     CHARACTER(*), PARAMETER :: routine = modname//"::p_gather_char_0d1d"

#ifndef NOMPI
     INTEGER :: p_comm

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF

     ! recvbuf argument is only significant on root
     IF (p_comm_rank(p_comm) == p_dest) THEN
       IF (LEN(sbuf) /= LEN(recvbuf(1))) THEN
         CALL finish (routine, 'Internal error: String lengths do not match!')
       END IF
     END IF

     CALL MPI_GATHER(sbuf, LEN(sbuf), p_char,    &
       &             recvbuf, LEN(sbuf), p_char, &
       &             p_dest, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_GATHER operation!')
#else
     IF (LEN(sbuf) /= LEN(recvbuf(1))) THEN
       CALL finish (routine, 'Internal error: String lengths do not match!')
     END IF
     recvbuf = sbuf
#endif
   END SUBROUTINE p_gather_char_0d1d


  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Gather()
  !---------------------------------------------------------------------------------------------------------------------------------
   SUBROUTINE p_gather_bool_0d1d (sendbuf, recvbuf, p_dest, comm)
     LOGICAL,           INTENT(in   ) :: sendbuf
     LOGICAL,           INTENT(inout) :: recvbuf(:)
     INTEGER,           INTENT(in   ) :: p_dest
     INTEGER, OPTIONAL, INTENT(in   ) :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = "mo_mpi:p_gather_bool_0d1d"
     INTEGER :: p_comm, comm_size, this_rank

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF

     CALL MPI_COMM_SIZE (comm, comm_size, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_COMM_SIZE operation!')
     CALL MPI_COMM_RANK (comm, this_rank, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_COMM_RANK operation!')

     IF ((this_rank == p_dest) .AND. (comm_size /= SIZE(recvbuf))) THEN
       WRITE (0,*) "comm_size     = ", comm_size
       WRITE (0,*) "SIZE(recvbuf) = ", SIZE(recvbuf)
       CALL finish(routine, "Receive buffer too small!")
     END IF

     CALL MPI_GATHER(sendbuf, 1, p_bool, &
       &             recvbuf, 1, p_bool, &
       &             p_dest, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_GATHER operation!')
#else
     recvbuf = sendbuf
#endif
   END SUBROUTINE p_gather_bool_0d1d


   SUBROUTINE p_gatherv_int (sendbuf, sendcount, recvbuf, recvcounts, &
     &                       displs, p_dest, comm)
     INTEGER,           INTENT(in)    :: sendbuf(:), sendcount
     INTEGER,           INTENT(inout) :: recvbuf(:), recvcounts(:), displs(:)
     INTEGER,           INTENT(in)    :: p_dest
     INTEGER, OPTIONAL, INTENT(in)    :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_gatherv_int"
     INTEGER :: p_comm

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF

     CALL MPI_GATHERV(sendbuf, sendcount,  MPI_INTEGER, &
       &              recvbuf, recvcounts, displs, MPI_INTEGER, &
       &              p_dest, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_GATHERV operation!')
#else
     recvbuf((displs(1)+1):(displs(1)+sendcount)) = sendbuf(1:sendcount)
#endif
   END SUBROUTINE p_gatherv_int


   SUBROUTINE p_gatherv_real2D2D (sendbuf, sendcount, recvbuf, recvcounts, &
     &                            displs, p_dest, comm)
     REAL(DP), INTENT(IN) :: sendbuf(:,:)
     INTEGER, INTENT(IN)  :: sendcount
     REAL(DP), INTENT(OUT) :: recvbuf(:,:)
     INTEGER, INTENT(IN)  :: recvcounts(:)
     INTEGER, INTENT(IN)  :: displs(:)
     INTEGER, INTENT(IN)  :: p_dest
     INTEGER, INTENT(IN)  :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_gatherv_real2D2D"

     INTEGER :: dim1_size

     dim1_size = SIZE(sendbuf, 1)

     CALL MPI_GATHERV(sendbuf, sendcount*dim1_size,  p_real_dp, &
       &              recvbuf, recvcounts(:)*dim1_size, displs*dim1_size, &
       &              p_real_dp, p_dest, comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in MPI_GATHERV operation!')
#else
     recvbuf(:, (displs(1)+1):(displs(1)+sendcount)) = sendbuf(:, 1:sendcount)
#endif
   END SUBROUTINE p_gatherv_real2D2D


   SUBROUTINE p_gatherv_sreal2D2D (sendbuf, sendcount, recvbuf, recvcounts, &
     &                            displs, p_dest, comm)
     REAL(SP), INTENT(IN) :: sendbuf(:,:)
     INTEGER, INTENT(IN)  :: sendcount
     REAL(SP), INTENT(OUT) :: recvbuf(:,:)
     INTEGER, INTENT(IN)  :: recvcounts(:)
     INTEGER, INTENT(IN)  :: displs(:)
     INTEGER, INTENT(IN)  :: p_dest
     INTEGER, INTENT(IN)  :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_gatherv_sreal2D2D"

     INTEGER :: dim1_size

     dim1_size = SIZE(sendbuf, 1)

     CALL MPI_GATHERV(sendbuf, sendcount*dim1_size,  p_real_sp, &
       &              recvbuf, recvcounts(:)*dim1_size, displs*dim1_size, &
       &              p_real_sp, p_dest, comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in MPI_GATHERV operation!')
#else
     recvbuf(:, (displs(1)+1):(displs(1)+sendcount)) = sendbuf(:, 1:sendcount)
#endif
   END SUBROUTINE p_gatherv_sreal2D2D


   SUBROUTINE p_gatherv_int2D2D (sendbuf, sendcount, recvbuf, recvcounts, &
     &                            displs, p_dest, comm)
     INTEGER, INTENT(IN)  :: sendbuf(:,:)
     INTEGER, INTENT(IN)  :: sendcount
     INTEGER, INTENT(OUT)  :: recvbuf(:,:)
     INTEGER, INTENT(IN)  :: recvcounts(:)
     INTEGER, INTENT(IN)  :: displs(:)
     INTEGER, INTENT(IN)  :: p_dest
     INTEGER, INTENT(IN)  :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_gatherv_int2D2D"

     INTEGER :: dim1_size

     dim1_size = SIZE(sendbuf, 1)

     CALL MPI_GATHERV(sendbuf, sendcount*dim1_size,  p_int, &
       &              recvbuf, recvcounts(:)*dim1_size, displs*dim1_size, &
       &              p_int, p_dest, comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in MPI_GATHERV operation!')
#else
     recvbuf(:, (displs(1)+1):(displs(1)+sendcount)) = sendbuf(:, 1:sendcount)
#endif
   END SUBROUTINE p_gatherv_int2D2D


   SUBROUTINE p_gatherv_real2D1D (sendbuf, sendcount, recvbuf, recvcounts, displs, p_dest, comm)
     REAL(dp),          INTENT(IN)    :: sendbuf(:,:)
     INTEGER,           INTENT(IN)    :: sendcount
     REAL(dp),          INTENT(INOUT) :: recvbuf(:)
     INTEGER,           intent(IN)    :: recvcounts(:), displs(:)
     INTEGER,           INTENT(in)    :: p_dest
     INTEGER,           INTENT(in)    :: comm

#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_gatherv_real2D1D"
     INTEGER :: p_error

     CALL MPI_GATHERV(sendbuf, sendcount, p_real_dp,   &    ! sendbuf, sendcount, sendtype
       &              recvbuf, recvcounts, displs,     &    ! recvbuf, recvcounts, displs
       &              p_real_dp, p_dest, comm, p_error)     ! recvtype, root, comm, error
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_GATHERV operation!')
#else
     recvbuf(:) = RESHAPE(sendbuf, (/ SIZE(recvbuf) /) )
#endif
   END SUBROUTINE p_gatherv_real2D1D


  SUBROUTINE p_gatherv_int2D1D (sendbuf, sendcount, recvbuf, recvcounts, displs, p_dest, comm)
    INTEGER,           INTENT(IN)    :: sendbuf(:,:)
    INTEGER,           INTENT(IN)    :: sendcount
    INTEGER,           INTENT(INOUT) :: recvbuf(:)
    INTEGER,           INTENT(IN)    :: recvcounts(:), displs(:)
    INTEGER,           INTENT(in)    :: p_dest
    INTEGER,           INTENT(in)    :: comm

#if !defined(NOMPI)
    CHARACTER(*), PARAMETER :: routine = modname//"::p_gatherv_int2D1D"
    INTEGER :: p_error

    CALL MPI_GATHERV(sendbuf, sendcount, p_int,       &    ! sendbuf, sendcount, sendtype
      &              recvbuf, recvcounts, displs,     &    ! recvbuf, recvcounts, displs
      &              p_int, p_dest, comm, p_error)         ! recvtype, root, comm, error
    IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_GATHERV operation!')
#else
    recvbuf(:) = RESHAPE(sendbuf, (/ SIZE(recvbuf) /) )
#endif
  END SUBROUTINE p_gatherv_int2D1D


   SUBROUTINE p_gatherv_real3D1D (sendbuf, sendcount, recvbuf, recvcounts, displs, p_dest, comm)
     REAL(dp),          INTENT(IN)    :: sendbuf(:,:,:)
     INTEGER,           INTENT(IN)    :: sendcount
     REAL(dp),          INTENT(INOUT) :: recvbuf(:)
     INTEGER,           intent(IN)    :: recvcounts(:), displs(:)
     INTEGER,           INTENT(in)    :: p_dest
     INTEGER,           INTENT(in)    :: comm

#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_gatherv_real2D1D"
     INTEGER :: p_error

     CALL MPI_GATHERV(sendbuf, sendcount, p_real_dp,   &    ! sendbuf, sendcount, sendtype
       &              recvbuf, recvcounts, displs,     &    ! recvbuf, recvcounts, displs
       &              p_real_dp, p_dest, comm, p_error)     ! recvtype, root, comm, error
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_GATHERV operation!')
#else
     recvbuf(:) = RESHAPE(sendbuf, (/ SIZE(recvbuf) /) )
#endif
   END SUBROUTINE p_gatherv_real3D1D


   SUBROUTINE p_scatterv_real1D2D (sendbuf, sendcounts, displs, recvbuf, recvcount, p_dest, comm)
     REAL(dp),          INTENT(IN)    :: sendbuf(:)
     INTEGER,           INTENT(IN)    :: sendcounts(:), displs(:)
     REAL(dp),          INTENT(INOUT) :: recvbuf(:,:)
     INTEGER,           INTENT(IN)    :: recvcount
     INTEGER,           INTENT(in)    :: p_dest
     INTEGER,           INTENT(in)    :: comm

#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_scatterv_real1D2D"
     INTEGER :: p_error

     CALL MPI_SCATTERV(sendbuf, sendcounts, displs,   &    ! sendbuf, sendcount, displs
       &               p_real_dp, recvbuf, recvcount, &    ! sendtype, recvbuf, recvcounts,
       &               p_real_dp, p_dest, comm, p_error)   ! recvtype, root, comm, error
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_SCATTERV operation!')
#else
     recvbuf(:,:) = RESHAPE(sendbuf, (/ SIZE(recvbuf,1), SIZE(recvbuf,2) /))
#endif
   END SUBROUTINE p_scatterv_real1D2D


  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Scatterv()
  !---------------------------------------------------------------------------------------------------------------------------------
   SUBROUTINE p_scatterv_real1D1D (sendbuf, sendcounts, displs, recvbuf, recvcount, p_src, comm)
        implicit none
        REAL(wp), INTENT(IN) :: sendbuf(:)
        INTEGER, INTENT(IN)  :: sendcounts(:)
        REAL(wp), INTENT(INOUT) :: recvbuf(:)
        INTEGER, INTENT(IN)  :: recvcount
        INTEGER, INTENT(IN)  :: displs(:)
        INTEGER, INTENT(IN)  :: p_src
        INTEGER, INTENT(IN)  :: comm

#ifndef NOMPI
        CHARACTER(*), PARAMETER :: routine = modname//"::p_scatterv_real1D1D"
        INTEGER :: ierr

        CALL MPI_Scatterv(sendbuf, sendcounts, displs, p_real_dp, &
        &                 recvbuf, recvcount, p_real_dp, &
        &                 p_src, comm, ierr)
        IF (ierr /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_Scatterv operation!')
#else
        recvbuf(1:recvcount) = sendbuf((displs(1)+1):(displs(1)+recvcount))
#endif
   END SUBROUTINE p_scatterv_real1D1D

   SUBROUTINE p_scatterv_int_1d1d(sendbuf, sendcounts, displs, recvbuf, &
     recvcount, p_src, comm)
     IMPLICIT NONE
     INTEGER, INTENT(IN) :: sendbuf(:)
     INTEGER, INTENT(IN) :: sendcounts(:)
     INTEGER, INTENT(INOUT) :: recvbuf(:)
     INTEGER, INTENT(IN)  :: recvcount
     INTEGER, INTENT(IN)  :: displs(:)
     INTEGER, INTENT(IN)  :: p_src
     INTEGER, INTENT(IN)  :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_scatterv_real1D1D"
     INTEGER :: ierr

     CALL MPI_Scatterv(sendbuf, sendcounts, displs, p_int, &
       &               recvbuf, recvcount, p_int, p_src, comm, ierr)
     IF (ierr /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_Scatterv operation!')
#else
     recvbuf(1:recvcount) = sendbuf((displs(1)+1):(displs(1)+recvcount))
#endif
   END SUBROUTINE p_scatterv_int_1d1d


  !---------------------------------------------------------------------------------------------------------------------------------
  !> wrapper for MPI_Scatterv()
  !---------------------------------------------------------------------------------------------------------------------------------
   SUBROUTINE p_scatterv_single1D1D (sendbuf, sendcounts, displs, recvbuf, recvcount, p_src, comm)
        implicit none
        REAL(sp), INTENT(IN) :: sendbuf(:)
        INTEGER, INTENT(IN)  :: sendcounts(:)
        REAL(sp), INTENT(INOUT) :: recvbuf(:)
        INTEGER, INTENT(IN)  :: recvcount
        INTEGER, INTENT(IN)  :: displs(:)
        INTEGER, INTENT(IN)  :: p_src
        INTEGER, INTENT(IN)  :: comm

#ifndef NOMPI
        CHARACTER(*), PARAMETER :: routine = modname//"::p_scatterv_single1D1D"
        INTEGER :: ierr

        CALL MPI_Scatterv(sendbuf, sendcounts, displs, p_real_sp, &
        &                 recvbuf, recvcount, p_real_sp, &
        &                 p_src, comm, ierr)
        IF (ierr /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_Scatterv operation!')
#else
        recvbuf(1:recvcount) = sendbuf((displs(1)+1):(displs(1)+recvcount))
#endif
   END SUBROUTINE p_scatterv_single1D1D

   SUBROUTINE p_allgather_int_0d1d(sendbuf, recvbuf, sendcount, recvcount, comm)
     INTEGER,           INTENT(inout) :: recvbuf(:)
     INTEGER,           INTENT(in) :: sendbuf
     INTEGER, OPTIONAL, INTENT(in) :: sendcount, recvcount, comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_allgather_int_0d1d"
     INTEGER :: p_comm, nsend, nrecv

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF
     IF (PRESENT(sendcount)) THEN
       nsend = sendcount
     ELSE
       nsend = 1
     END IF
     IF (PRESENT(recvcount)) THEN
       nrecv = recvcount
     ELSE
       nrecv = 1
     END IF

     CALL mpi_allgather(sendbuf, nsend, mpi_integer, &
          &             recvbuf, nrecv, mpi_integer, &
          &             p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in mpi_allgather operation!')
#else
     recvbuf = sendbuf
#endif
   END SUBROUTINE p_allgather_int_0d1d

   SUBROUTINE p_allgather_int_1d2d(sendbuf, recvbuf, sendcount, recvcount, comm)
     INTEGER,           INTENT(inout) :: recvbuf(:,:)
     INTEGER,           INTENT(in) :: sendbuf(:)
     INTEGER, OPTIONAL, INTENT(in) :: sendcount, recvcount, comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_allgather_int_1d2d"
     INTEGER :: p_comm, nrecv, nsend

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF
     IF (PRESENT(sendcount)) THEN
       nsend = sendcount
     ELSE
       nsend = SIZE(sendbuf)
     END IF
     IF (PRESENT(recvcount)) THEN
       nrecv = recvcount
     ELSE
       nrecv = SIZE(recvbuf, 1)
     END IF
     CALL mpi_allgather(sendbuf, nsend, mpi_integer, &
          &             recvbuf, nrecv, mpi_integer, &
          &             p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in mpi_allgather operation!')
#else
     recvbuf(:, 1) = sendbuf
#endif
   END SUBROUTINE p_allgather_int_1d2d

   SUBROUTINE p_allgatherv_real_1d(sendbuf, recvbuf, recvcounts, comm)
     REAL(dp),          INTENT(in)    :: sendbuf(:)
     REAL(dp),          INTENT(inout) :: recvbuf(:)
     INTEGER,           INTENT(in)    :: recvcounts(:)
     INTEGER, OPTIONAL, INTENT(in)    :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_allgatherv_real_1d"
     INTEGER :: p_comm, sendcount, comm_size, i
     INTEGER, ALLOCATABLE :: displs(:)

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF

     IF (p_comm_is_intercomm(p_comm)) THEN
      comm_size = p_comm_remote_size(p_comm)
     ELSE
      comm_size = p_comm_size(p_comm)
     END IF

     IF ((comm_size > SIZE(recvcounts, 1)) .OR. &
      &  (SUM(recvcounts) > SIZE(recvbuf, 1))) &
       CALL finish(routine, "invalid recvcounts")

     ALLOCATE(displs(comm_size))
     displs(1) = 0
     DO i = 2, comm_size
       displs(i) = displs(i-1) + recvcounts(i-1)
     END DO

     sendcount = SIZE(sendbuf)
     CALL mpi_allgatherv(sendbuf, sendcount, p_real_dp, &
          &              recvbuf, recvcounts, displs, p_real_dp, &
          &              p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in mpi_allgatherv operation!')

     DEALLOCATE(displs)
#else
     recvbuf = sendbuf
#endif
   END SUBROUTINE p_allgatherv_real_1d

   SUBROUTINE p_allgatherv_int_1d(sendbuf, recvbuf, recvcounts, displs, &
     &                            comm)
     INTEGER,           INTENT(in)    :: sendbuf(:)
     INTEGER,           INTENT(inout) :: recvbuf(:)
     INTEGER,           INTENT(in)    :: recvcounts(:), displs(:)
     INTEGER, OPTIONAL, INTENT(in)    :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: routine = modname//"::p_allgatherv_int_1d"
     INTEGER :: p_comm, sendcount, comm_size

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF
     IF (p_comm_is_intercomm(p_comm)) THEN
      comm_size = p_comm_remote_size(p_comm)
     ELSE
      comm_size = p_comm_size(p_comm)
     END IF

     IF (comm_size > SIZE(displs)) CALL finish(routine, "invalid recvdispls")

     sendcount = SIZE(sendbuf)
     CALL mpi_allgatherv(sendbuf, sendcount, mpi_integer, &
          &              recvbuf, recvcounts, displs, mpi_integer, &
          &              p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in mpi_allgatherv operation!')

#else
     recvbuf = sendbuf
#endif
   END SUBROUTINE p_allgatherv_int_1d

   SUBROUTINE p_allgatherv_int_1d_contiguous(sendbuf, recvbuf, recvcounts, &
     &                                       comm)
     INTEGER,           INTENT(in)    :: sendbuf(:)
     INTEGER,           INTENT(inout) :: recvbuf(:)
     INTEGER,           INTENT(in)    :: recvcounts(:)
     INTEGER, OPTIONAL, INTENT(in)    :: comm

#ifndef NOMPI
     CHARACTER(*), PARAMETER :: &
          routine = modname//"::p_allgatherv_int_1d_contiguous"
     INTEGER :: p_comm, comm_size, i, n
     INTEGER, ALLOCATABLE :: displs(:)

     IF (PRESENT(comm)) THEN
       p_comm = comm
     ELSE
       p_comm = process_mpi_all_comm
     ENDIF
     IF (p_comm_is_intercomm(p_comm)) THEN
      comm_size = p_comm_remote_size(p_comm)
     ELSE
      comm_size = p_comm_size(p_comm)
     END IF

     ALLOCATE(displs(comm_size))
     n = 0
     DO i = 1, comm_size
       displs(i) = n
       n = n + recvcounts(i)
     END DO

     CALL p_allgatherv(sendbuf, recvbuf, recvcounts, displs, p_comm)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in mpi_allgatherv operation!')

#else
     recvbuf = sendbuf
#endif
   END SUBROUTINE p_allgatherv_int_1d_contiguous


   ! Commits a user-defined MPI type
   !
   FUNCTION p_commit_type_struct(oldtypes, blockcounts) RESULT(newtype)
     INTEGER :: newtype
     INTEGER, INTENT(IN) :: oldtypes(2), blockcounts(2)
#if !defined(NOMPI)
     INTEGER :: ierr
     INTEGER(MPI_ADDRESS_KIND) :: typeLB, extent, offsets(2)
     ! define structured type and commit it
     CALL MPI_TYPE_GET_EXTENT(oldtypes(1), typeLB, extent, ierr)
     offsets(:) = (/ 0_MPI_ADDRESS_KIND, extent /)
     CALL MPI_TYPE_CREATE_STRUCT(2, blockcounts, offsets, oldtypes, newtype, ierr)
     CALL MPI_TYPE_COMMIT(newtype, ierr)
#else
     newtype = 0
#endif
   END FUNCTION p_commit_type_struct


   SUBROUTINE p_alltoall_int (sendbuf, recvbuf, comm)
     INTEGER,           INTENT(inout) :: sendbuf(:), recvbuf(:)
     INTEGER,           INTENT(in) :: comm
#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoall_int"
     INTEGER :: p_comm, p_error

     p_comm = comm
     CALL MPI_ALLTOALL(sendbuf, 1, p_int, recvbuf, 1, p_int, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) CALL finish (routine, 'Error in MPI_ALLTOALL operation!')
#else
     recvbuf(:) = sendbuf(:)
#endif
   END SUBROUTINE p_alltoall_int


   SUBROUTINE p_alltoallv_real_2d (sendbuf, sendcounts, sdispls, &
     &                             recvbuf, recvcounts, rdispls, comm)
     REAL(dp), TARGET,  INTENT(in) :: sendbuf(:,:)
     INTEGER,           INTENT(in) :: sendcounts(:), sdispls(:)
     REAL(dp),          INTENT(inout) :: recvbuf(:,:)
     INTEGER,           INTENT(in) :: recvcounts(:), rdispls(:)
     INTEGER,           INTENT(in) :: comm
#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoallv_real_2d"
     INTEGER :: p_comm, p_error, dim1_size
     REAL(dp), POINTER :: p_sendbuf(:,:)
     REAL(dp), TARGET :: dummy(1,1)

     p_comm = comm
     dim1_size = SIZE(sendbuf, 1)
     IF (SIZE(sendbuf) > 0) THEN
       p_sendbuf => sendbuf
     ELSE
       p_sendbuf => dummy
     END IF
     CALL MPI_ALLTOALLV(p_sendbuf, sendcounts(:)*dim1_size, &
       &                sdispls(:)*dim1_size, p_real_dp, recvbuf, &
       &                recvcounts(:)*dim1_size, rdispls(:)*dim1_size, &
       &                p_real_dp, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in MPI_ALLTOALLV operation!')
#else
     ! displs are zero based -> have to add 1
     recvbuf(:,rdispls(1)+1:rdispls(1)+recvcounts(1)) = &
       sendbuf(:,sdispls(1)+1:sdispls(1)+sendcounts(1))
#endif
   END SUBROUTINE p_alltoallv_real_2d


   SUBROUTINE p_alltoallv_sreal_2d (sendbuf, sendcounts, sdispls, &
     &                              recvbuf, recvcounts, rdispls, comm)
     REAL(sp), TARGET,  INTENT(in) :: sendbuf(:,:)
     INTEGER,           INTENT(in) :: sendcounts(:), sdispls(:)
     REAL(sp),          INTENT(inout) :: recvbuf(:,:)
     INTEGER,           INTENT(in) :: recvcounts(:), rdispls(:)
     INTEGER,           INTENT(in) :: comm
#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoallv_sreal_2d"
     INTEGER :: p_comm, p_error, dim1_size
     REAL(sp), POINTER :: p_sendbuf(:,:)
     REAL(sp), TARGET :: dummy(1,1)

     p_comm = comm
     dim1_size = SIZE(sendbuf, 1)
     IF (SIZE(sendbuf) > 0) THEN
       p_sendbuf => sendbuf
     ELSE
       p_sendbuf => dummy
     END IF
     CALL MPI_ALLTOALLV(p_sendbuf, sendcounts(:)*dim1_size, &
       &                sdispls(:)*dim1_size, p_real_sp, recvbuf, &
       &                recvcounts(:)*dim1_size, rdispls(:)*dim1_size, &
       &                p_real_sp, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in MPI_ALLTOALLV operation!')
#else
     ! displs are zero based -> have to add 1
     recvbuf(:,rdispls(1)+1:rdispls(1)+recvcounts(1)) = &
       sendbuf(:,sdispls(1)+1:sdispls(1)+sendcounts(1))
#endif
   END SUBROUTINE p_alltoallv_sreal_2d


   SUBROUTINE p_alltoallv_int_2d (sendbuf, sendcounts, sdispls, &
     &                            recvbuf, recvcounts, rdispls, comm)
     INTEGER, TARGET,   INTENT(in) :: sendbuf(:,:)
     INTEGER,           INTENT(in) :: sendcounts(:), sdispls(:)
     INTEGER,           INTENT(inout) :: recvbuf(:,:)
     INTEGER,           INTENT(in) :: recvcounts(:), rdispls(:)
     INTEGER,           INTENT(in) :: comm
#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoallv_int_2d"
     INTEGER :: p_comm, p_error, dim1_size
     INTEGER, POINTER :: p_sendbuf(:,:)
     INTEGER, TARGET :: dummy(1,1)

     p_comm = comm
     dim1_size = SIZE(sendbuf, 1)
     IF (SIZE(sendbuf) > 0) THEN
       p_sendbuf => sendbuf
     ELSE
       p_sendbuf => dummy
     END IF
     CALL MPI_ALLTOALLV(p_sendbuf, sendcounts(:)*dim1_size, &
       &                sdispls(:)*dim1_size, p_int, recvbuf, &
       &                recvcounts(:)*dim1_size, rdispls(:)*dim1_size, &
       &                p_int, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in MPI_ALLTOALLV operation!')
#else
     ! displs are zero based -> have to add 1
     recvbuf(:,rdispls(1)+1:rdispls(1)+recvcounts(1)) = &
       sendbuf(:,sdispls(1)+1:sdispls(1)+sendcounts(1))
#endif
   END SUBROUTINE p_alltoallv_int_2d


   SUBROUTINE p_alltoallv_int (sendbuf, sendcounts, sdispls, &
     &                         recvbuf, recvcounts, rdispls, comm)
     INTEGER,           INTENT(in) :: sendbuf(:), sendcounts(:), sdispls(:)
     INTEGER,           INTENT(inout) :: recvbuf(:)
     INTEGER,           INTENT(in) :: recvcounts(:), rdispls(:)
     INTEGER,           INTENT(in) :: comm
#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoallv_int"
     INTEGER :: p_comm, p_error

     p_comm = comm
     CALL MPI_ALLTOALLV(sendbuf, sendcounts, sdispls, p_int, &
       &                recvbuf, recvcounts, rdispls, p_int, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in MPI_ALLTOALLV operation!')
#else
     ! displs are zero based -> have to add 1
     recvbuf(rdispls(1)+1:rdispls(1)+recvcounts(1)) = &
       sendbuf(sdispls(1)+1:sdispls(1)+sendcounts(1))
#endif
   END SUBROUTINE p_alltoallv_int

   SUBROUTINE p_alltoallv_int_i8_1d(sendbuf, sendcounts, sdispls, &
     &                         recvbuf, recvcounts, rdispls, comm)
     INTEGER(i8),       INTENT(in) :: sendbuf(:)
     INTEGER(i8),       INTENT(inout) :: recvbuf(:)
     INTEGER,           INTENT(in) :: sendcounts(:), sdispls(:), &
       &                              recvcounts(:), rdispls(:)
     INTEGER,           INTENT(in) :: comm
#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoallv_int_i8_1d"
     INTEGER :: p_comm, p_error

     p_comm = comm
     CALL mpi_alltoallv(sendbuf, sendcounts, sdispls, p_int_i8, &
       &                recvbuf, recvcounts, rdispls, p_int_i8, p_comm, p_error)
     IF (p_error /=  MPI_SUCCESS) &
       CALL finish (routine, 'Error in MPI_ALLTOALLV operation!')
#else
     ! displs are zero based -> have to add 1
     recvbuf(rdispls(1)+1:rdispls(1)+recvcounts(1)) = &
       sendbuf(sdispls(1)+1:sdispls(1)+sendcounts(1))
#endif
   END SUBROUTINE p_alltoallv_int_i8_1d

#if !defined(NOMPI)
   SUBROUTINE p_alltoallv_p2p_real_2d_core(dim1_size, sendbuf, sendcounts, sdispls, &
        &                             recvbuf, recvcounts, rdispls, comm)
     INTEGER, INTENT(in) :: dim1_size
     REAL(wp),          INTENT(in) :: sendbuf(dim1_size,*)
     INTEGER,           INTENT(in) :: sendcounts(:), sdispls(:)
     REAL(wp),          INTENT(inout) :: recvbuf(dim1_size,*)
     INTEGER,           INTENT(in) :: recvcounts(:), rdispls(:)
     INTEGER,           INTENT(in) :: comm
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoallv_p2p_real_2d"
     INTEGER :: i, comm_size, tag, ofs, datatype

     CALL p_wait

     comm_size = p_comm_size(comm)
     tag = 1
     SELECT CASE (wp)
       CASE(sp)
         datatype = p_real_sp
       CASE(dp)
         datatype = p_real_dp
       CASE DEFAULT
         datatype = -1
         CALL finish (routine, 'invalid read type')
     END SELECT
     DO i = 1, comm_size
       IF (recvcounts(i) > 0) THEN
         ofs = 1 + rdispls(i)
         CALL p_inc_request
         CALL mpi_irecv(recvbuf(:, ofs:ofs+recvcounts(i)-1), &
              &         recvcounts(i) * dim1_size, datatype, i - 1, tag, &
              &         comm, p_request(p_irequest), p_error)
#ifdef DEBUG
         IF (p_error /= MPI_SUCCESS) THEN
           WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', &
                my_process_mpi_all_id, &
                ' from ', i - 1, ' for tag ', tag, ' failed.'
           WRITE (nerr,'(a,i4)') ' Error = ', p_error
           CALL abort_mpi
         END IF
#endif
       END IF
       IF (sendcounts(i) > 0) THEN
         ofs = 1 + sdispls(i)
         CALL p_inc_request
         CALL mpi_isend(sendbuf(:, ofs:ofs+sendcounts(i)-1), &
              &         sendcounts(i) * dim1_size, datatype, i - 1, tag, &
              &         comm, p_request(p_irequest), p_error)
#ifdef DEBUG
         IF (p_error /= MPI_SUCCESS) THEN
           WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND on ', &
                my_process_mpi_all_id, &
                ' from ', i - 1, ' for tag ', tag, ' failed.'
           WRITE (nerr,'(a,i4)') ' Error = ', p_error
           CALL abort_mpi
         END IF
#endif
       END IF
     END DO
     CALL p_wait
   END SUBROUTINE p_alltoallv_p2p_real_2d_core
#endif

   SUBROUTINE p_alltoallv_p2p_real_2d (sendbuf, sendcounts, sdispls, &
     &                             recvbuf, recvcounts, rdispls, comm)
     REAL(wp),          INTENT(in) :: sendbuf(:,:)
     INTEGER,           INTENT(in) :: sendcounts(:), sdispls(:)
     REAL(wp),          INTENT(inout) :: recvbuf(:,:)
     INTEGER,           INTENT(in) :: recvcounts(:), rdispls(:)
     INTEGER,           INTENT(in) :: comm
#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoallv_p2p_real_2d"
     INTEGER :: dim1_size

     dim1_size = SIZE(sendbuf, 1)
     CALL p_alltoallv_p2p_real_2d_core(dim1_size, &
          &                            sendbuf, sendcounts, sdispls, &
          &                            recvbuf, recvcounts, rdispls, comm)
#else
     ! displs are zero based -> have to add 1
     recvbuf(:,rdispls(1)+1:rdispls(1)+recvcounts(1)) = &
       sendbuf(:,sdispls(1)+1:sdispls(1)+sendcounts(1))
#endif
   END SUBROUTINE p_alltoallv_p2p_real_2d

#if !defined(NOMPI)
   SUBROUTINE p_alltoallv_p2p_int_2d_core(dim1_size, sendbuf, sendcounts, sdispls, &
        &                                 recvbuf, recvcounts, rdispls, comm)
     INTEGER, INTENT(in) :: dim1_size
     INTEGER,           INTENT(in) :: sendbuf(dim1_size,*)
     INTEGER,           INTENT(in) :: sendcounts(:), sdispls(:)
     INTEGER,           INTENT(inout) :: recvbuf(dim1_size,*)
     INTEGER,           INTENT(in) :: recvcounts(:), rdispls(:)
     INTEGER,           INTENT(in) :: comm
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoallv_p2p_int_2d"
     INTEGER :: i, comm_size, tag, ofs

     CALL p_wait

     comm_size = p_comm_size(comm)
     tag = 1
     DO i = 1, comm_size
       IF (recvcounts(i) > 0) THEN
         ofs = 1 + rdispls(i)
         CALL p_inc_request
         CALL mpi_irecv(recvbuf(:, ofs:ofs+recvcounts(i)-1), &
              &         recvcounts(i) * dim1_size, p_int, i - 1, tag, &
              &         comm, p_request(p_irequest), p_error)
#ifdef DEBUG
         IF (p_error /= MPI_SUCCESS) THEN
           WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_IRECV on ', &
                my_process_mpi_all_id, &
                ' from ', i - 1, ' for tag ', tag, ' failed.'
           WRITE (nerr,'(a,i4)') ' Error = ', p_error
           CALL abort_mpi
         END IF
#endif
       END IF
       IF (sendcounts(i) > 0) THEN
         ofs = 1 + sdispls(i)
         CALL p_inc_request
         CALL mpi_isend(sendbuf(:, ofs:ofs+sendcounts(i)-1), &
              &         sendcounts(i) * dim1_size, p_int, i - 1, tag, &
              &         comm, p_request(p_irequest), p_error)
#ifdef DEBUG
         IF (p_error /= MPI_SUCCESS) THEN
           WRITE (nerr,'(a,i4,a,i4,a,i6,a)') ' MPI_ISEND on ', &
                my_process_mpi_all_id, &
                ' from ', i - 1, ' for tag ', tag, ' failed.'
           WRITE (nerr,'(a,i4)') ' Error = ', p_error
           CALL abort_mpi
         END IF
#endif
       END IF
     END DO
     CALL p_wait
   END SUBROUTINE p_alltoallv_p2p_int_2d_core
#endif

   SUBROUTINE p_alltoallv_p2p_int_2d (sendbuf, sendcounts, sdispls, &
     &                                recvbuf, recvcounts, rdispls, comm)
     INTEGER,           INTENT(in) :: sendbuf(:,:)
     INTEGER,           INTENT(in) :: sendcounts(:), sdispls(:)
     INTEGER,           INTENT(inout) :: recvbuf(:,:)
     INTEGER,           INTENT(in) :: recvcounts(:), rdispls(:)
     INTEGER,           INTENT(in) :: comm
#if !defined(NOMPI)
     CHARACTER(*), PARAMETER :: routine = modname//"::p_alltoallv_p2p_int_2d"
     INTEGER :: dim1_size

     dim1_size = SIZE(sendbuf, 1)
     CALL p_alltoallv_p2p_int_2d_core(dim1_size, &
          &                           sendbuf, sendcounts, sdispls, &
          &                           recvbuf, recvcounts, rdispls, comm)
#else
     ! displs are zero based -> have to add 1
     recvbuf(:,rdispls(1)+1:rdispls(1)+recvcounts(1)) = &
       sendbuf(:,sdispls(1)+1:sdispls(1)+sendcounts(1))
#endif
   END SUBROUTINE p_alltoallv_p2p_int_2d

  SUBROUTINE p_clear_request(request)
    INTEGER, INTENT(INOUT) :: request
    request = mpi_request_null
  END SUBROUTINE p_clear_request

  SUBROUTINE p_clear_requests(requests)
    INTEGER, INTENT(INOUT) :: requests(:)
    requests = mpi_request_null
  END SUBROUTINE p_clear_requests

  FUNCTION p_mpi_wtime()
    REAL(dp) :: p_mpi_wtime
    p_mpi_wtime = MERGE_HAVE_MPI(MPI_Wtime(), 0d0)
  END FUNCTION p_mpi_wtime


  !--------------------------------------------------------------------


  !> @return Global MPI ranks within communicator "comm"
  !
  SUBROUTINE get_mpi_comm_world_ranks(comm, global_ranks, nranks)
    INTEGER, INTENT(IN)  :: comm               !< MPI communicator
    INTEGER, ALLOCATABLE, INTENT(OUT) :: global_ranks(:)    !< Output: list of global MPI ranks in communicator "comm"
    INTEGER, INTENT(OUT) :: nranks             !< Output: number of entries in rank list
    ! local variables
    CHARACTER(*), PARAMETER :: routine = modname//"::get_mpi_comm_world_ranks"
    INTEGER              :: p_error, grp_comm, grp_comm_world, i
    INTEGER, ALLOCATABLE :: comm_ranks(:)

#if !defined(NOMPI)
    nranks = 0
    IF (comm /= MPI_COMM_NULL) THEN
      nranks = p_comm_size(comm)    ! inquire communicator size

      ALLOCATE(comm_ranks(nranks), global_ranks(nranks))
      comm_ranks(1:nranks) = (/ (i, i=0,(nranks-1)) /)

      CALL MPI_COMM_GROUP(comm, grp_comm, p_error)
      IF (p_error /= MPI_SUCCESS)  CALL finish (routine, 'Error in MPI_COMM_GROUP operation!')
      CALL MPI_COMM_GROUP(MPI_COMM_WORLD, grp_comm_world,  p_error)
      IF (p_error /= MPI_SUCCESS)  CALL finish (routine, 'Error in MPI_COMM_GROUP operation!')

      global_ranks(:) = 0
      CALL MPI_GROUP_TRANSLATE_RANKS(grp_comm, nranks, comm_ranks, &
        &                            grp_comm_world, global_ranks, p_error)
      IF (p_error /= MPI_SUCCESS)  CALL finish (routine, 'Error in MPI_GROUP_TRANSLATE_RANKS operation!')

      CALL MPI_GROUP_FREE(grp_comm,       p_error)
      IF (p_error /= MPI_SUCCESS)  CALL finish (routine, 'Error in MPI_GROUP_FREE operation!')
      CALL MPI_GROUP_FREE(grp_comm_world, p_error)
      IF (p_error /= MPI_SUCCESS)  CALL finish (routine, 'Error in MPI_GROUP_FREE operation!')

      DEALLOCATE(comm_ranks)
    END IF
#else
    nranks = 1
    ALLOCATE(global_ranks(1))
    global_ranks(1) = 0
#endif
  END SUBROUTINE get_mpi_comm_world_ranks

  !--------------------------------------------------------------------

  LOGICAL FUNCTION p_comm_is_intercomm(intercomm)

    INTEGER, INTENT(IN) :: intercomm
    LOGICAL :: flag
    INTEGER :: p_error

#ifndef NOMPI
    CALL MPI_COMM_TEST_INTER(intercomm, flag, p_error)
    IF (p_error /= MPI_SUCCESS) &
      CALL finish ("p_comm_is_intercomm", &
        &          'Error in MPI_Comm_test_inter operation!')

    p_comm_is_intercomm = flag
#else
    p_comm_is_intercomm = .FALSE.
#endif
  END FUNCTION p_comm_is_intercomm

  !--------------------------------------------------------------------

  INTEGER FUNCTION p_comm_remote_size(intercomm)

    INTEGER, INTENT(IN) :: intercomm
    INTEGER :: remote_size, p_error

#ifndef NOMPI
    CALL MPI_COMM_REMOTE_SIZE(intercomm, remote_size, p_error)
    IF (p_error /= MPI_SUCCESS) &
      CALL finish ("p_comm_remote_size", &
        &          'Error in MPI_Comm_remote_size operation!')

    p_comm_remote_size = remote_size
#else
    p_comm_remote_size = 0
#endif

  END FUNCTION p_comm_remote_size


  LOGICAL FUNCTION p_isEqual_int(val, comm) RESULT(resultVar)
    INTEGER, INTENT(IN) :: val
    INTEGER, OPTIONAL, INTENT(IN) :: comm

#ifndef NOMPI
    CHARACTER(*), PARAMETER :: routine = modname//":p_isEqual_int"
    INTEGER :: sendBuffer(2), minmax(2)

    !Compute the MIN AND MAX of the values using the fact that MAX_i(x_i) = -min_i(-x_i)
    !i. e. we compute both with a single MPI MIN reduction.
    sendBuffer(1) = val
    sendBuffer(2) = -val
    minmax = p_min(sendBuffer, comm = comm)
    resultVar = minmax(1) == -minmax(2)
#else
    resultVar = .TRUE.
#endif
  END FUNCTION p_isEqual_int

  LOGICAL FUNCTION p_isEqual_charArray(charArray, comm) RESULT(resultVar)
    CHARACTER(KIND = C_CHAR), INTENT(IN) :: charArray(:)
    INTEGER, OPTIONAL, INTENT(IN) :: comm

#ifndef NOMPI
    INTEGER :: myProc, nextProc, prevProc, p_comm, commSize, i, error
    CHARACTER(KIND = C_CHAR) :: prevArray(SIZE(charArray, 1))
    LOGICAL :: stringsEqual

    IF (PRESENT(comm)) THEN
       p_comm = comm
    ELSE
       p_comm = process_mpi_all_comm
    ENDIF

    !Check whether all processes have the same string SIZE.
    resultVar = p_isEqual(SIZE(charArray, 1), comm = p_comm)
    IF(.NOT. resultVar) RETURN

    !Get the neighbor ranks for a cyclic DATA exchange.
    myProc = p_comm_rank(p_comm)
    commSize = p_comm_size(p_comm)
    nextProc = myProc + 1
    IF(nextProc == commSize) nextProc = 0
    prevProc = myProc - 1
    IF(prevProc == -1) prevProc = commSize - 1

    !Do a cyclic exchange of the strings, compare the local string to
    !the one passed IN from the prevProc, AND reduce whether all
    !strings compared equal.
    CALL p_sendrecv_char_array(charArray, nextProc, prevArray, prevProc, 0, comm = p_comm)
    stringsEqual = .TRUE.
    DO i = 1, SIZE(charArray, 1)
        IF(charArray(i) /= prevArray(i)) stringsEqual = .FALSE.
    END DO

    CALL MPI_Allreduce(stringsEqual, resultVar, 1, MPI_LOGICAL, MPI_LAND, p_comm, error)
#else
    resultVar = .TRUE.
#endif
  END FUNCTION p_isEqual_charArray

END MODULE mo_mpi