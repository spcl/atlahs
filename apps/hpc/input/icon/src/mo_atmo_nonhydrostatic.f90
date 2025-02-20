! @brief branch for the non-hydrostatic ICON workflow
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

MODULE mo_atmo_nonhydrostatic
use mpi
USE mo_kind,                 ONLY: wp, i4, i8
USE mo_exception,            ONLY: message, message_text, finish, print_value
USE mtime,                   ONLY: OPERATOR(>)
USE mo_fortran_tools,        ONLY: init
USE mo_impl_constants,       ONLY: SUCCESS, max_dom, inwp, iaes, LSS_JSBACH
USE mo_timer,                ONLY: timers_level, timer_start, timer_stop, timer_init_latbc, &
  &                                timer_model_init, timer_init_icon, timer_read_restart, timer_init_dace
USE mo_master_config,        ONLY: isRestart, getModelBaseDir
USE mo_time_config,          ONLY: t_time_config, time_config
USE mo_load_restart,         ONLY: read_restart_files
USE mo_key_value_store,      ONLY: t_key_value_store
USE mo_restart_nml_and_att,  ONLY: getAttributesForRestarting
USE mo_restart,              ONLY: t_RestartDescriptor, createRestartDescriptor, deleteRestartDescriptor
USE mo_io_config,            ONLY: configure_io, init_var_in_output, var_in_output
USE mo_parallel_config,      ONLY: nproma, num_prefetch_proc
USE mo_nh_pzlev_config,      ONLY: configure_nh_pzlev, deallocate_nh_pzlev
USE mo_advection_config,     ONLY: configure_advection
USE mo_art_config,           ONLY: configure_art
USE mo_assimilation_config,  ONLY: configure_lhn, assimilation_config
USE mo_run_config,           ONLY: dtime,                & !    namelist parameter
  &                                lart,                 &
  &                                ltestcase,            &
  &                                iforcing,             & !    namelist parameter
  &                                output_mode,          &
  &                                lvert_nest, ntracer,  &
  &                                ldass_lhn, msg_level, &
  &                                iqc, iqt,             &
  &                                ico2, io3,            &
  &                                number_of_grid_used
USE mo_lnd_nwp_config,       ONLY: lseaice
USE mo_initicon_config,      ONLY: pinit_seed, pinit_amplitude, init_mode, iterate_iau
USE mo_nh_testcases,         ONLY: init_nh_testcase, init_nh_testcase_scm
USE mo_nh_testcases_nml,     ONLY: nh_test_name
#ifndef __NO_ICON_LES__
USE mo_ls_forcing_nml,       ONLY: is_ls_forcing, is_nudging
USE mo_ls_forcing,           ONLY: init_ls_forcing
USE mo_turbulent_diagnostic, ONLY: init_les_turbulent_output, close_les_turbulent_output
USE mo_nh_vert_interp_les,   ONLY: init_vertical_grid_for_les
#endif
USE mo_dynamics_config,      ONLY: nnow, nnew, nnow_rcf
! Horizontal grid
USE mo_model_domain,         ONLY: p_patch, p_patch_local_parent
USE mo_grid_config,          ONLY: n_dom, n_dom_start, start_time, end_time, &
     &                             is_plane_torus, l_limited_area, l_scm_mode
USE mo_intp_data_strc,       ONLY: p_int_state
USE mo_intp_lonlat_types,    ONLY: lonlat_grids
USE mo_grf_intp_data_strc,   ONLY: p_grf_state, p_grf_state_local_parent
! Vertical grid
USE mo_vertical_grid,        ONLY: set_nh_metrics
! Grid nesting
USE mo_nh_nest_utilities,    ONLY: complete_nesting_setup
! NH-namelist state
USE mo_nonhydrostatic_config,ONLY: configure_nonhydrostatic, kstart_moist, kend_qvsubstep, &
  &                                kstart_tracer, ndyn_substeps

! NH-Model states
USE mo_nonhydro_state,       ONLY: p_nh_state, p_nh_state_lists,               &
  &                                construct_nh_state, destruct_nh_state,      &
  &                                duplicate_prog_state
USE mo_prepadv_state,        ONLY: construct_prepadv_state, destruct_prepadv_state
USE mo_opt_diagnostics,      ONLY: construct_opt_diag, destruct_opt_diag,      &
  &                                compute_lonlat_area_weights
! Time integration
USE mo_nh_stepping,          ONLY: perform_nh_stepping
! Initialization with real data
USE mo_initicon,            ONLY: init_icon
USE mo_ext_data_state,      ONLY: ext_data
! Community Interface (ComIn)
USE mo_comin_config,        ONLY: configure_comin
! meteogram output
USE mo_meteogram_output,    ONLY: meteogram_init, meteogram_finalize
USE mo_meteogram_config,    ONLY: meteogram_output_config
USE mo_name_list_output_config,   ONLY: is_variable_in_output
USE mo_name_list_output_init, ONLY:  init_name_list_output,        &
  &                                  parse_variable_groups,        &
  &                                  collect_requested_ipz_levels, &
  &                                  output_file, create_vertical_axes
USE mo_level_selection,     ONLY: create_mipz_level_selections
USE mo_name_list_output,    ONLY: close_name_list_output
USE mo_pp_scheduler,        ONLY: pp_scheduler_init, pp_scheduler_finalize

! NWP physics
USE mo_nwp_phy_state,        ONLY: prm_diag, prm_nwp_tend, prm_nwp_stochconv,  &
  &                                construct_nwp_phy_state
USE mo_nwp_lnd_state,        ONLY: p_lnd_state, construct_nwp_lnd_state
USE mo_atm_phy_nwp_config,   ONLY: configure_atm_phy_nwp, atm_phy_nwp_config
USE mo_synsat_config,        ONLY: configure_synsat
USE mo_iau,                  ONLY: save_initial_state, reset_to_initial_state
#ifndef __NO_NWP__
USE mo_ensemble_pert_config, ONLY: configure_ensemble_pert
USE mo_ext_data_init,        ONLY: init_index_lists
USE mo_sppt_state,           ONLY: construct_sppt_state, destruct_sppt_state
USE mo_sppt_config,          ONLY: sppt_config, configure_sppt
USE mo_nwp_phy_cleanup,      ONLY: cleanup_nwp_phy
USE mo_nwp_ww,               ONLY: configure_ww
#endif
#ifdef __ICON_ART
! ICON-ART
USE mo_art_init_interface,   ONLY: art_init_atmo_tracers_nwp
#endif

! AES physics
USE mo_aes_phy_config,      ONLY: aes_phy_tc, dt_zero, aes_phy_config
USE mo_aes_rad_config,      ONLY: aes_rad_config
USE mo_aes_vdf_config,      ONLY: aes_vdf_config
#ifndef __NO_AES__
USE mo_aes_phy_memory,      ONLY: construct_aes_phy_memory
USE mo_cloud_mig_memory,    ONLY: construct_cloud_mig_memory
USE mo_cloud_two_memory,    ONLY: construct_cloud_two_memory
USE mo_radiation_forcing_memory, ONLY: construct_rad_forcing_list => construct_radiation_forcing_list
USE mo_physical_constants,  ONLY: amd, amco2
USE mo_aes_phy_dims,        ONLY: init_aes_phy_dims
USE mo_aes_phy_init,        ONLY: init_aes_phy_params, init_aes_phy_external, &
   &                              init_aes_phy_field, init_o3_lcariolle
USE mo_aes_phy_cleanup,     ONLY: cleanup_aes_phy
USE mo_interface_aes_tmx,   ONLY: init_tmx
#endif
#ifndef __NO_JSBACH__
  USE mo_jsb_model_init,    ONLY: jsbach_init_after_restart
#endif
! Needed for upper atmosphere configuration
USE mo_sleve_config,        ONLY: flat_height
USE mo_io_units,            ONLY: filename_max
USE mo_vertical_coord_table,ONLY: vct_a
USE mo_upatmo_impl_const,   ONLY: iUpatmoPrcStat
USE mo_upatmo_config,       ONLY: upatmo_config, upatmo_phy_config, &
    &                             configure_upatmo, destruct_upatmo
#ifndef __NO_ICON_UPATMO__
USE mo_upatmo_state,        ONLY: construct_upatmo_state, &
    &                             destruct_upatmo_state
USE mo_upatmo_phy_setup,    ONLY: finalize_upatmo_phy_nwp
#endif

USE mo_util_mtime,          ONLY: getElapsedSimTimeInSeconds
USE mo_output_event_types,  ONLY: t_sim_step_info
USE mo_action,              ONLY: ACTION_RESET, reset_act
USE mo_turb_vdiff_params,   ONLY: VDIFF_TURB_3DSMAGORINSKY
USE mo_limarea_config,      ONLY: latbc_config
USE mo_async_latbc_types,   ONLY: t_latbc_data
USE mo_async_latbc,         ONLY: init_prefetch, close_prefetch
USE mo_radar_data_state,    ONLY: init_radar_data, destruct_radar_data, &
  &                               construct_lhn_state, destruct_lhn_state
USE mo_rttov_interface,     ONLY: rttov_finalize, rttov_initialize
USE mo_synsat_config,       ONLY: lsynsat
USE mo_mpi,                 ONLY: my_process_is_stdio, p_comm_work_only, my_process_is_work_only
USE mo_var_list_register_utils, ONLY: vlr_print_groups
USE mo_sync,                ONLY: sync_patch_array, sync_c
USE mo_nudging_config,      ONLY: l_global_nudging
USE mo_random_util,         ONLY: add_random_noise_3d, add_random_noise_2d

USE mo_icon2dace,           ONLY: init_dace, finish_dace

!-------------------------------------------------------------------------
#ifdef HAVE_CDI_PIO
  USE mo_impl_constants,      ONLY: pio_type_cdipio
  USE mo_parallel_config,     ONLY: pio_type
  USE mo_cdi,                 ONLY: namespaceGetActive, namespaceSetActive
  USE mo_cdi_pio_interface,         ONLY: nml_io_cdi_pio_namespace
#endif

#ifndef __NO_ICON_COMIN__
  USE comin_host_interface, ONLY: comin_callback_context_call,        &
    &                             EP_SECONDARY_CONSTRUCTOR,           &
    &                             EP_ATM_INIT_FINALIZE,               &
    &                             EP_DESTRUCTOR,                      &
    &                             comin_var_list_finalize,            &
    &                             comin_descrdata_finalize,           &
    &                             comin_setup_finalize,               &
    &                             COMIN_DOMAIN_OUTSIDE_LOOP
  USE mo_comin_adapter,     ONLY: icon_append_comin_variables,        &
    &                             icon_append_comin_tracer_variables, &
    &                             icon_append_comin_tracer_phys_tend, &
    &                             icon_expose_variables
#endif


IMPLICIT NONE
PRIVATE

PUBLIC :: atmo_nonhydrostatic
PUBLIC :: construct_atmo_nonhydrostatic, destruct_atmo_nonhydrostatic


CONTAINS

  !---------------------------------------------------------------------
  SUBROUTINE atmo_nonhydrostatic(latbc)
    TYPE(t_latbc_data)           :: latbc   !< data structure for async latbc prefetching

    INTEGER                      :: iter
    TYPE(t_time_config), TARGET  :: time_config_iau
    TYPE(t_time_config), POINTER :: ptr_time_config  => NULL()

    CLASS(t_RestartDescriptor), POINTER  :: restartDescriptor

    CHARACTER(*), PARAMETER :: routine = "atmo_nonhydrostatic"
    INTEGER :: ierr
    !------------------------------------------------------------------
    ! Latency injection experiment
    !------------------------------------------------------------------
    real(kind=8) :: start_time
    INTEGER :: p_error

    !------------------------------------------------------------------
    ! Latency injection experiment
    !------------------------------------------------------------------
    CALL MPI_BARRIER(MPI_COMM_WORLD, p_error)
    CALL MPI_PCONTROL(1)
    start_time = MPI_WTIME()
    ! print *, "[DEBUG] Starting time: ", start_time
    !------------------------------------------------------------------
    ! Now start the time stepping:
    !------------------------------------------------------------------

    restartDescriptor => createRestartDescriptor("atm")

    ! for iterative IAU, perform_nh_stepping is called twice with distinct
    ! model stop dates.
    IF (iterate_iau .AND. .NOT. isRestart()) THEN

      ! create a local copy of the time_config object
      CALL time_config%copy(tc_new=time_config_iau)

      DO iter=1,2
        IF (iter==1) THEN
          WRITE(message_text,'(a)') 'IAU iteration is activated: Start of first cycle with halved IAU window'
          !
          ! save initial state if IAU iteration mode is chosen
          CALL save_initial_state(p_patch(1:))
          !
          ! modify model stop date for first iteration
          time_config_iau%tc_stopdate = time_config%tc_exp_startdate
          ptr_time_config => time_config_iau
        ELSE IF (iter==2) THEN
          WRITE(message_text,'(a)') 'Reset model to initial state, repeat IAU with full incrementation window'
          !
          ! restore initial state, and some additional control fields
          CALL reset_to_initial_state(p_patch(1:), p_nh_state)
          ptr_time_config => time_config
        ENDIF
        !
        CALL message(routine, message_text)
        CALL perform_nh_stepping( time_config       = ptr_time_config, &
          &                       iau_iter          = iter,            &
          &                       latbc             = latbc,           &
          &                       restartDescriptor = restartDescriptor )
      ENDDO

      ! cleanup
      ptr_time_config => NULL()
      CALL time_config_iau%destruct()
    ELSE

      CALL perform_nh_stepping( time_config       = time_config, &
        &                       iau_iter          = 0,           &
        &                       latbc             = latbc,       &
        &                       restartDescriptor = restartDescriptor )
    ENDIF
    print *, "[DEBUG] Total runtime: ", MPI_WTIME() - start_time
    CALL MPI_PCONTROL(0)
    CALL deleteRestartDescriptor(restartDescriptor)

#ifndef __NO_ICON_COMIN__
    CALL comin_callback_context_call(EP_DESTRUCTOR, COMIN_DOMAIN_OUTSIDE_LOOP)

    CALL comin_var_list_finalize(ierr)
    IF (ierr /= 0) STOP
    CALL comin_descrdata_finalize(ierr)
    IF (ierr /= 0) STOP
    CALL comin_setup_finalize(ierr)
    IF (ierr /= 0) STOP
#endif

    !---------------------------------------------------------------------
    ! Integration finished. Clean up.
    !---------------------------------------------------------------------
    CALL destruct_atmo_nonhydrostatic(latbc, lacc=.TRUE.)

  END SUBROUTINE atmo_nonhydrostatic
  !---------------------------------------------------------------------

  !---------------------------------------------------------------------
  SUBROUTINE construct_atmo_nonhydrostatic(latbc)
    TYPE(t_latbc_data), INTENT(INOUT) :: latbc !< data structure for async latbc prefetching

    CHARACTER(*), PARAMETER :: routine = "construct_atmo_nonhydrostatic"

    INTEGER :: jg, ist, jt

    TYPE(t_sim_step_info) :: sim_step_info  
    REAL(wp) :: sim_time
    TYPE(t_key_value_store), POINTER :: restartAttributes
    LOGICAL :: lrestart
    CHARACTER(LEN=filename_max) :: model_base_dir
    INTEGER :: seed_size, i
    INTEGER, ALLOCATABLE :: seed(:)


    IF (timers_level > 1) CALL timer_start(timer_model_init)


    DO jg =1,n_dom
      CALL configure_nonhydrostatic( jg, p_patch(jg)%nlev,     &
        &                            p_patch(jg)%nshift_total  )
    ENDDO

    IF (iforcing == iaes) THEN
#ifdef __NO_AES__   
      CALL finish (routine, 'Error: remove --disable-aes and reconfigure')
#else
      CALL init_aes_phy_dims
      CALL init_aes_phy_params( p_patch(1:) )
#endif
    END IF

    IF(iforcing == inwp) THEN

#ifndef __NO_NWP__
      CALL configure_ensemble_pert(ext_data, time_config%tc_exp_startdate)

      ! - generate index lists for tiles (land, ocean, lake)
      ! index lists for ice-covered and non-ice covered ocean points
      ! are initialized in init_nwp_phy
      CALL init_index_lists (p_patch(1:), ext_data)
#endif

      CALL configure_atm_phy_nwp(n_dom, p_patch(1:), dtime)

      CALL configure_synsat()

      DO jg = 1, n_dom
#ifndef __NO_NWP__
        CALL configure_ww( time_config%tc_startdate, jg, p_patch(jg)%nlev, p_patch(jg)%nshift_total, 'ICON')
#endif
        !
        ! initialize number of chemical tracers for convection
        CALL configure_art(jg)
      ENDDO

#ifndef __NO_NWP__
      ! configure SPPT
      IF ( ANY(sppt_config(1:n_dom)%lsppt) ) THEN
        CALL configure_sppt(n_dom, p_patch(1:), time_config%tc_current_date)
      ENDIF
#endif

    ENDIF

    ! Check if optional diagnostics are requested for output
    CALL init_var_in_output(n_dom, iforcing == inwp)

    ! initialize ldom_active flag if this is not a restart run

    ! calculate elapsed simulation time in seconds
    sim_time = getElapsedSimTimeInSeconds(time_config%tc_current_date) 

    DO jg=1, n_dom
      p_patch(jg)%ldom_active &
           =        (jg <= 1 .OR. start_time(jg) <= sim_time) &
           &  .AND. end_time(jg) > sim_time
    ENDDO

    !---------------------------------------------------------------------
    ! 4.c Non-Hydrostatic / NWP
    !---------------------------------------------------------------------

    ! Note(GZ): Land state now needs to be allocated even if physics is turned
    ! off because ground temperature is included in feedback since r8133
    ! However, setting inwp_surface = 0 effects that only a few 2D fields are allocated
    ALLOCATE(p_lnd_state(n_dom), stat=ist)
    IF (ist /= success) CALL finish(routine, &
      &                             'allocation for (NWP) land state failed')

    IF(iforcing /= inwp) atm_phy_nwp_config(:)%inwp_surface = 0

    ! Now allocate memory for the nonhydrostatic state
    CALL construct_nh_state(p_patch(1:), n_timelevels=2, var_in_output=var_in_output(:))

    ! Add optional diagnostic variable lists (might remain empty)
    CALL construct_opt_diag(p_patch(1:), .TRUE.)

    ! construct prep_adv state, which is required for tracer transport
    CALL construct_prepadv_state (p_patch(1:))

    IF (iforcing == inwp) THEN
#ifndef __NO_NWP__
      CALL construct_nwp_phy_state( p_patch(1:), var_in_output)
      CALL construct_nwp_lnd_state( p_patch(1:), p_lnd_state, var_in_output(:)%smi, n_timelevels=2 )

      ! Construct SPPT state
      IF ( ANY(sppt_config(1:n_dom)%lsppt) ) THEN
        CALL construct_sppt_state(p_patch(1:))
      ENDIF
#endif
    END IF

    IF (iforcing == iaes) THEN
#ifdef __NO_AES__   
      CALL finish (routine, 'Error: remove --disable-aes and reconfigure')
#else
      CALL construct_aes_phy_memory    ( p_patch(1:), ntracer )
      CALL construct_cloud_mig_memory  ( p_patch(1:) )
      CALL construct_cloud_two_memory  ( p_patch(1:) )
      CALL construct_rad_forcing_list  ( p_patch(1:) )
#endif
    END IF

! Upper atmosphere

    model_base_dir = getModelBaseDir()
    lrestart       = isRestart()

    CALL configure_upatmo( n_dom_start, n_dom, p_patch(n_dom_start:), isRestart(), atm_phy_nwp_config(:)%lupatmo_phy,      &
      &                    init_mode, iforcing, time_config%tc_exp_startdate, time_config%tc_exp_stopdate, start_time(:),  & 
      &                    end_time(:), dtime, atm_phy_nwp_config(:)%dt_rad, ndyn_substeps, flat_height,                   &
      &                    aes_rad_config(:)%l_orbvsop87, aes_rad_config(:)%cecc, aes_rad_config(:)%cobld,                 & 
      &                    aes_rad_config(:)%clonp, aes_rad_config(:)%lyr_perp, aes_rad_config(:)%yr_perp, model_base_dir, &
      &                    msg_level, vct_a )

#ifndef __NO_ICON_UPATMO__
! Create state only if enabled
    CALL construct_upatmo_state( n_dom, nproma, p_patch(1:), upatmo_config(1:), upatmo_phy_config(1:), vct_a )
#endif

#ifndef __NO_ICON_COMIN__
    ! ----------------------------------------------------------
    ! UNDER DEVELOPMENT (ICON ComIn)
    !
    ! loop over the total list of additional requested variables and
    ! perform `add_var` / `add_ref` operations needed.
    ! remark: variables are added to a separate variable list.
    CALL icon_append_comin_tracer_variables(p_patch(1:))
    CALL icon_append_comin_tracer_phys_tend(p_patch(1:))
    CALL icon_append_comin_variables(p_patch(1:))

    ! expose ICON's variables to the ComIn infrastructure.

    CALL icon_expose_variables()

    ! call to secondary constructor
    !   third party modules retrieve pointers to data arrays, telling
    !   ICON ComIn about the context where these will be accessed.
    CALL comin_callback_context_call(EP_SECONDARY_CONSTRUCTOR, COMIN_DOMAIN_OUTSIDE_LOOP)
    ! ----------------------------------------------------------
#endif

#ifdef MESSY
    CALL messy_init_memory(n_dom)
#endif

    ! Due to the requirement for ICON-ART to overwrite advection-Namelist settings
    ! via add_ref/add_tracer_ref, configure_advection is called AFTER the nh_state
    ! is created. Otherwise, potential modifications of the advection-Namelist
    ! can not be taken into account properly. Unfortunately this conflicts with
    ! our policy to call the config-routines as early as possible.
    DO jg =1,n_dom
      CALL configure_advection( jg             = jg,                                  & !in
        &                       nlev           = p_patch(jg)%nlev,                    & !in
        &                       nlev_1         = p_patch(1)%nlev,                     & !in
        &                       iforcing       = iforcing,                            & !in
        &                       iqc            = iqc,                                 & !in
        &                       iqt            = iqt,                                 & !in
        &                       kstart_moist   = kstart_moist(jg),                    & !in
        &                       kend_qvsubstep = kend_qvsubstep(jg),                  & !in
        &                       lvert_nest     = lvert_nest,                          & !in
        &                       ntracer        = ntracer,                             & !in
        &                       prog_list      = p_nh_state_lists(jg)%prog_list(:),   & !in
        &                       tracer_list    = p_nh_state_lists(jg)%tracer_list(:), & !inout
        &                       kstart_tracer  = kstart_tracer(jg,:) )                  !in
    ENDDO

    IF (ldass_lhn) THEN
      CALL message(routine,'configure_lhn')
      DO jg =1,n_dom
        CALL configure_lhn(jg)
      ENDDO
      !$ACC ENTER DATA COPYIN(assimilation_config)

      CALL init_radar_data(p_patch(1:))

      CALL construct_lhn_state(p_patch(1:))
    ENDIF

    !------------------------------------------------------------------
    ! Prepare for time integration
    !------------------------------------------------------------------

    CALL set_nh_metrics(p_patch(1:)     ,&
         &              p_nh_state      ,&
         &              p_nh_state_lists ,&
         &              p_int_state(1:) ,&
         &              ext_data        )

    IF (n_dom > 1) THEN
      CALL complete_nesting_setup(p_patch(1:), p_patch_local_parent(n_dom_start+1:), &
        &                         p_grf_state_local_parent(n_dom_start+1:), p_nh_state(1:))
    END IF

    ! Initialize DACE routines
    IF (assimilation_config(1)% dace_coupling) then
      IF (timers_level > 4) CALL timer_start(timer_init_dace)
      CALL init_dace (comm=p_comm_work_only, p_io=0, ldetached=.NOT.my_process_is_work_only())
      IF (timers_level > 4) CALL timer_stop(timer_init_dace)
    END IF
#ifndef __NO_ICON_LES__
    ! init LES
    DO jg = 1 , n_dom
      IF(atm_phy_nwp_config(jg)%is_les_phy .OR. aes_vdf_config(jg)%turb==VDIFF_TURB_3DSMAGORINSKY) THEN
        CALL init_vertical_grid_for_les(jg, p_patch(jg)       ,&
           &                            p_int_state(jg)       ,&
           &                            p_nh_state(jg)%metrics)
      END IF
    END DO
#endif
    !------------------------------------------------------------------
    ! Prepare initial conditions for time integration.
    !------------------------------------------------------------------
    !
    IF (isRestart()) THEN
      !
      ! This is a resumed integration. Read model state from restart file(s).
      !
      IF (timers_level > 4) CALL timer_start(timer_read_restart)
      !
      DO jg = 1,n_dom
        IF (p_patch(jg)%ldom_active) THEN
          CALL read_restart_files( p_patch(jg), n_dom)
        END IF
      END DO
      !
      CALL message(routine,'normal exit from read_restart_files')
      !
      IF (timers_level > 4) CALL timer_stop(timer_read_restart)
      !
#ifndef __NO_JSBACH__
      DO jg = 1,n_dom
        IF (.NOT. p_patch(jg)%ldom_active) CYCLE
        IF (aes_phy_config(jg)%ljsb &
            & .OR. atm_phy_nwp_config(jg)%inwp_surface == LSS_JSBACH) THEN
          CALL jsbach_init_after_restart(jg)
        END IF
      END DO
#endif
      !
    ELSE
      !
      ! This is a new integration.
      !
      IF (ltestcase) THEN
        !
        ! Initialize testcase analytically
        !
        IF (l_scm_mode) THEN
          CALL init_nh_testcase_scm(p_patch(1:)     ,&
            &                       p_nh_state      ,&
            &                       p_int_state(1:) ,&
            &                       p_lnd_state(1:) ,&
            &                       ext_data        )
        ELSE
          CALL init_nh_testcase    (p_patch(1:)     ,&
            &                       p_nh_state      ,&
            &                       p_int_state(1:) ,&
            &                       p_lnd_state(1:) ,&
            &                       ext_data        ,&
            &                       ntl=2           )
        ENDIF
        !
#ifndef __NO_ICON_LES__
        IF(is_ls_forcing .OR. is_nudging) &
          CALL init_ls_forcing(p_nh_state(1)%metrics)
#endif
        !
      ELSE
        !
        ! Initialize with real atmospheric data
        !
        IF (timers_level > 4) CALL timer_start(timer_init_icon)
        !
        IF (iforcing == inwp) THEN
          !
          ! Initialize atmosphere, surface and land
          !
          CALL init_icon (p_patch(1:)     ,&
            &             p_int_state(1:) ,&
            &             p_grf_state(1:) ,&
            &             p_nh_state(1:)  ,&
            &             ext_data(1:)    ,&
            &             prm_diag(1:)    ,&
            &             prm_nwp_stochconv(1:),&
            &             p_lnd_state(1:) )
          !
        ELSE ! iforcing == iaes, inoforcing, ...
#ifdef __NO_AES__   
          CALL finish (routine, 'Error: remove --disable-aes and reconfigure')
#else
          !
          ! Initialize the atmosphere only
          !
          CALL init_icon (p_patch(1:)     ,&
            &             p_int_state(1:) ,&
            &             p_grf_state(1:) ,&
            &             p_nh_state(1:)  ,&
            &             ext_data(1:)    )
          !
#endif
        END IF ! iforcing
        !
        IF (timers_level > 4) CALL timer_stop(timer_init_icon)
        !
      END IF ! ltestcase

      IF(pinit_seed /= 0_i8) THEN
        CALL RANDOM_SEED(SIZE = seed_size)
        ALLOCATE(seed(seed_size))

        seed(1) = INT(pinit_seed, i4)
        IF (seed_size>1_i8) seed(2) = INT(pinit_seed/2_i8**32_i8, i4)

        DO i=3,seed_size
            seed(i) = ISHFTC(seed(i-2), 1)
        ENDDO

        CALL RANDOM_SEED(PUT = seed)

        DO jg=1,n_dom
          CALL add_random_noise_3d(p_patch(jg)%cells%all, pinit_amplitude, &
                                   p_nh_state(jg)%prog(nnow(jg))%w)
          CALL add_random_noise_3d(p_patch(jg)%edges%all, pinit_amplitude, &
                                   p_nh_state(jg)%prog(nnow(jg))%vn)
          CALL add_random_noise_3d(p_patch(jg)%cells%all, pinit_amplitude, &
                                   p_nh_state(jg)%prog(nnow(jg))%theta_v)
          CALL add_random_noise_3d(p_patch(jg)%cells%all, pinit_amplitude, &
                                   p_nh_state(jg)%prog(nnow(jg))%exner)
          CALL add_random_noise_3d(p_patch(jg)%cells%all, pinit_amplitude, &
                                   p_nh_state(jg)%prog(nnow(jg))%rho)

          IF (lseaice) THEN
            CALL add_random_noise_2d(p_patch(jg)%cells%all, pinit_amplitude, &
                                     p_lnd_state(jg)%prog_wtr(nnow_rcf(jg))%t_ice)
          ENDIF

          IF (.NOT. ltestcase .OR. nh_test_name == 'dcmip_pa_12') THEN
            CALL add_random_noise_3d(p_patch(jg)%cells%all, pinit_amplitude, &
                                     p_nh_state(jg)%prog(nnow_rcf(jg))%tracer(:,:,:,1))
          ENDIF

          IF (iforcing == inwp ) THEN
            DO jt = 1, SIZE(p_lnd_state(jg)%prog_lnd(nnow_rcf(jg))%t_so_t,4)
              CALL add_random_noise_3d(p_patch(jg)%cells%all, pinit_amplitude, &
                                  p_lnd_state(jg)%prog_lnd(nnow_rcf(jg))%t_so_t(:,:,:,jt) )
            ENDDO
          ENDIF

          CALL duplicate_prog_state(p_nh_state(jg)%prog(nnow(jg)), p_nh_state(jg)%prog(nnew(jg)))
        ENDDO
      ENDIF

      !
      ! Initialize tracers which are not available in the analysis file,
      ! but may be used with AES physics, for real cases or test cases.
      !
      IF (iforcing == iaes ) THEN
#ifdef __NO_AES__   
        CALL finish (routine, 'Error: remove --disable-aes and reconfigure')
#else
        DO jg = 1,n_dom
          IF (.NOT. p_patch(jg)%ldom_active) CYCLE
          !
          ! CO2 tracer
          IF ( iqt <= ico2 .AND. ico2 <= ntracer) THEN
!$OMP PARALLEL
            CALL init(p_nh_state(jg)%prog(nnow_rcf(jg))%tracer(:,:,:,ico2),aes_rad_config(jg)% vmr_co2*amco2/amd)
!$OMP END PARALLEL
            CALL print_value('CO2 tracer initialized with constant vmr', &
              &              aes_rad_config(jg)% vmr_co2*amco2/amd,    &
              &              routine=routine)
          END IF
          !
          ! O3 tracer
          IF ( iqt <= io3 .AND. io3 <= ntracer) THEN
            IF (aes_phy_tc(jg)%dt_car > dt_zero) THEN
              CALL init_o3_lcariolle( time_config%tc_current_date                          ,&
                &                     p_patch(jg)                                          ,&
                &                     p_nh_state(jg)%diag% pres               (:,:,:)      ,&
                &                     p_nh_state(jg)%prog(nnow_rcf(jg))%tracer(:,:,:,io3)  )
              CALL sync_patch_array ( sync_c,p_patch(jg)                                   ,&
                &                     p_nh_state(jg)%prog(nnow_rcf(jg))%tracer(:,:,:,io3)  )
              CALL message(routine,'o3 tracer is initialized by the Cariolle lin. o3 scheme')
            ELSE
!$OMP PARALLEL
              CALL init(p_nh_state(jg)%prog(nnow_rcf(jg))%tracer(:,:,:,io3),0.0_wp)
!$OMP END PARALLEL
              CALL message(routine,'o3 tracer is initialized to zero, check setup')
            END IF
          END IF
          !
        END DO
        !
#endif
      END IF
      !
    END IF ! isRestart()


    ! Now set up AES physics fields
    !
    IF ( iforcing == iaes ) THEN
#ifdef __NO_AES__   
        CALL finish (routine, 'Error: remove --disable-aes and reconfigure')
#else
      !
      ! read external data for real case
      IF (.NOT. ltestcase) THEN 
        CALL init_aes_phy_external( p_patch(1:)                   ,&
           &                          time_config%tc_current_date )
      END IF
      !
      ! prepare fields of the physics state, real and test case
      DO jg = 1,n_dom
        CALL init_aes_phy_field( p_patch(jg)                       ,&
          &                      p_nh_state(jg)% diag% temp(:,:,:) )
      END DO
      !
#endif
    END IF

    !------------------------------------------------------------------
    ! Asynchronous pre-fetching
    !------------------------------------------------------------------

    ! If async prefetching is in effect, init_prefetch is a collective call
    ! with the prefetching processor and effectively starts async prefetching
    IF ((num_prefetch_proc == 1) .AND. (latbc_config%itype_latbc > 0)) THEN
      IF (timers_level > 4) CALL timer_start(timer_init_latbc)
      CALL init_prefetch(latbc)
      IF (timers_level > 4) CALL timer_stop(timer_init_latbc)
    ENDIF


    !------------------------------------------------------------------
    ! Status output for the Community Interface (ComIn)
    !------------------------------------------------------------------

    CALL configure_comin()


    !------------------------------------------------------------------
    ! Prepare output file
    !------------------------------------------------------------------

    CALL configure_io()   ! set n_chkpt and n_diag, which control
                          ! writing of restart files and tot_int diagnostics.

    ! Add a special metrics variable containing the area weights of
    ! the regular lon-lat grid.
    CALL compute_lonlat_area_weights(lonlat_grids, p_nh_state_lists)

    ! Map the variable groups given in the output namelist onto the
    ! corresponding variable subsets:
    ! ATTENTION: all add_vars must be finished before calling this routine.
    IF (output_mode%l_nml) THEN
      CALL parse_variable_groups()
    END IF

    ! if output on z and/or p-levels is required do some config
    !
    ! Note that on the compute PEs we must call this *before* the
    ! initialization of the vertical interpolation,
    ! "pp_scheduler_init"
    IF (output_mode%l_nml) THEN
      CALL collect_requested_ipz_levels()
      DO jg = 1, n_dom
        CALL configure_nh_pzlev(jg, nproma, p_patch(jg)%npromz_c,  &
          &                     p_patch(jg)%nblks_c)
      ENDDO
    END IF

    ! setup of post-processing job queue, e.g. setup of optional
    ! diagnostic quantities like pz-level interpolation
    CALL pp_scheduler_init( (iforcing == inwp) )

    ! setup of RTTOV interface (assumes expanded variable groups)
    IF (ANY(lsynsat(:)))  CALL rttov_initialize()

    ! If async IO is in effect, init_name_list_output is a collective call
    ! with the IO procs and effectively starts async IO
    IF (output_mode%l_nml) THEN
      ! compute sim_start, sim_end
      sim_step_info%sim_start = time_config%tc_exp_startdate
      sim_step_info%sim_end = time_config%tc_exp_stopdate
      sim_step_info%run_start = time_config%tc_startdate
      sim_step_info%restart_time = time_config%tc_stopdate

      sim_step_info%dtime      = dtime
      sim_step_info%jstep0 = 0

      CALL getAttributesForRestarting(restartAttributes)
      ! get start counter for time loop from restart file:
      IF (restartAttributes%is_init) &
        & CALL restartAttributes%get("jstep", sim_step_info%jstep0)
      CALL init_name_list_output(sim_step_info)

      !---------------------------------------------------------------------
      !     Setup of meteogram output
      !---------------------------------------------------------------------
      DO jg =1,n_dom
        IF (meteogram_output_config(jg)%lenabled) THEN
          ! For dry test cases: do not sample moist variables
          ! (but allow for TORUS moist runs; see also mo_mtgrm_output.F90)
          IF (ltestcase .and. .not. is_plane_torus) THEN
            CALL meteogram_init(meteogram_output_config(jg), jg, p_patch(jg), &
              &                ext_data(jg), p_nh_state(jg),                  &
              &                p_lnd_state=p_lnd_state(jg), iforcing=iforcing,&
              &                grid_uuid=p_patch(jg)%grid_uuid,               &
              &                number_of_grid_used=number_of_grid_used(jg) )
          ELSE
            CALL meteogram_init(meteogram_output_config(jg), jg, p_patch(jg), &
              &                ext_data(jg), p_nh_state(jg), prm_diag(jg),    &
              &                p_lnd_state(jg), prm_nwp_tend(jg), iforcing,                     &
              &                grid_uuid=p_patch(jg)%grid_uuid,               &
              &                number_of_grid_used=number_of_grid_used(jg) )
          END IF
        END IF
      END DO

      CALL create_mipz_level_selections(output_file)
      CALL create_vertical_axes(output_file)
    END IF

    ! tmx
#ifndef __NO_AES__
    IF (aes_vdf_config(1)%use_tmx) THEN
      IF (n_dom == 1) THEN
        CALL init_tmx(p_patch(1), dtime)
      ELSE
        CALL finish(routine, 'Only one domain supported currently for new tmx')
      END IF
    END IF
#endif

#ifdef MESSY
    CALL messy_init_coupling
    CALL messy_init_tracer
#endif

#ifndef __NO_ICON_COMIN__
    CALL comin_callback_context_call(EP_ATM_INIT_FINALIZE, COMIN_DOMAIN_OUTSIDE_LOOP)
#endif
    ! Determine if temporally averaged vertically integrated moisture quantities need to be computed

    IF (iforcing == inwp) THEN
        atm_phy_nwp_config(1:n_dom)%lcalc_extra_avg = &
        is_variable_in_output(var_name="astr_u_sso")      .OR. &
        is_variable_in_output(var_name="accstr_u_sso")    .OR. &
        is_variable_in_output(var_name="astr_v_sso")      .OR. &
        is_variable_in_output(var_name="accstr_v_sso")    .OR. &
        is_variable_in_output(var_name="adrag_u_grid")    .OR. &
        is_variable_in_output(var_name="adrag_v_grid")
     ENDIF


    !----------------------!
    !  Initialize actions  !
    !----------------------!
    !

    ! Initialize reset-Action, i.e. assign variables to action object
    CALL reset_act%initialize(ACTION_RESET)

#ifndef __NO_ICON_LES__
    !Anurag Dipankar, MPIM (2014-01-14)
    !Special 1D and 0D output for LES runs till we get add_var/nml_out working
    !Only for Torus runs with single domain
    DO jg = 1 , n_dom

      IF(atm_phy_nwp_config(jg)%is_les_phy .AND. is_plane_torus) &
           CALL init_les_turbulent_output(p_patch(jg), p_nh_state(jg)%metrics, &
           &    time_config%tc_startdate, var_in_output(jg)%rh, ldelete=(.NOT. isRestart()))

    END DO
#endif

#ifndef __NO_NWP__
#ifdef __ICON_ART
    IF (iforcing == inwp) THEN
      !--------------------------!
      !  Initialize ART for NWP  !
      !--------------------------!
      IF (lart) THEN
        DO jg=1, n_dom
          IF (.NOT. p_patch(jg)%ldom_active) CYCLE
          CALL art_init_atmo_tracers_nwp(                       &
               &  jg,                                           &
               &  time_config%tc_current_date,                  &
               &  p_nh_state(jg),                               &
               &  ext_data(jg),                                 &
               &  prm_diag(jg),                                 &
               &  p_nh_state(jg)%prog(nnow(jg)),                &
               &  p_nh_state(jg)%prog(nnow_rcf(jg))%tracer,     &
               &  p_nh_state_lists(jg)%prog_list(nnow_rcf(jg)), & 
               &  p_patch(jg)%nest_level)
        ENDDO
      END IF  ! lart
    END IF  ! iforcing == inwp
#endif
#endif /* __NO_NWP__ */

    !-------------------------------------------------------!
    !  (Optional) detailed print-out of some variable info  !
    !-------------------------------------------------------!

    ! variable group information
    IF (my_process_is_stdio() .AND. (msg_level >= 15)) &
      & CALL vlr_print_groups(idom=1)

    IF (timers_level > 1) CALL timer_stop(timer_model_init)

  END SUBROUTINE construct_atmo_nonhydrostatic

  !---------------------------------------------------------------------
  SUBROUTINE destruct_atmo_nonhydrostatic(latbc, lacc)
    TYPE(t_latbc_data), INTENT(INOUT) :: latbc !< data structure for async latbc prefetching
    LOGICAL, INTENT(IN), OPTIONAL :: lacc ! If true, use openacc
    CHARACTER(*), PARAMETER :: routine = "destruct_atmo_nonhydrostatic"


    INTEGER :: jg
    
#ifdef HAVE_CDI_PIO
    INTEGER :: prev_cdi_namespace
#endif

    !---------------------------------------------------------------------
    ! 6. Integration finished. Clean up.
    !---------------------------------------------------------------------

    CALL message(routine,'start to clean up')

#ifdef MESSY
    CALL messy_free_memory
#endif

    ! Destruction of post-processing job queue
    CALL pp_scheduler_finalize()

    ! Destruction of some RTTOV data structures  (if enabled)
    IF (ANY(lsynsat(:)))  CALL rttov_finalize()

    ! Delete optional diagnostics
    CALL destruct_opt_diag()

    ! Delete state variables
    CALL destruct_nh_state( p_nh_state, p_nh_state_lists )

    ! Delete state variables for transport
    CALL destruct_prepadv_state()

#ifndef __NO_ICON_LES__
    ! close LES diag files
    DO jg = 1 , n_dom
      IF(atm_phy_nwp_config(jg)%is_les_phy .AND. is_plane_torus) &
        CALL close_les_turbulent_output(jg)
    END DO
#endif
    ! cleanup NWP physics
    IF (iforcing == inwp) THEN
#ifndef __NO_NWP__
      CALL cleanup_nwp_phy()

      ! Destruct SPPT state
      IF ( ANY(sppt_config(1:n_dom)%lsppt) ) THEN
        CALL destruct_sppt_state()
      ENDIF
#endif
    ENDIF

    IF (iforcing == iaes) THEN
#ifdef __NO_AES__   
      CALL finish (routine, 'Error: remove --disable-aes and reconfigure')
#else
      CALL cleanup_aes_phy()
#endif
    ENDIF

#ifndef __NO_ICON_UPATMO__
    ! This is required for NWP forcing only. 
    ! For AES forcing, the following will likely be done in 
    ! 'src/atm_phy_aes/mo_aes_phy_cleanup: cleanup_aes_phy'
    DO jg = 1, n_dom
      IF (upatmo_config( jg )%nwp_phy%l_phy_stat( iUpatmoPrcStat%enabled )) THEN
        CALL finalize_upatmo_phy_nwp( p_patch( jg ) ) !in
      ENDIF
    ENDDO  !jg

    !---------------------------------------------------------------------
    !             Destruct the upper-atmosphere data types
    !---------------------------------------------------------------------

    CALL destruct_upatmo_state( n_dom, upatmo_config(1:) )
#endif

    !---------------------------------------------------------------------
    !          Destruct the upper-atmosphere configuration type
    !---------------------------------------------------------------------

    ! After the following call, 'upatmo_config' cannot be used anymore!
    CALL destruct_upatmo( n_dom_start, n_dom )

    ! call close name list prefetch
    IF ((l_limited_area .OR. l_global_nudging) .AND. latbc_config%itype_latbc > 0) THEN
      IF (num_prefetch_proc >= 1) THEN
        CALL close_prefetch()
        CALL latbc%finalize()
      END IF
    END IF

    ! Delete output variable lists
    IF (output_mode%l_nml) THEN
      CALL message(routine, 'delete output variable lists')
      CALL close_name_list_output
      CALL message(routine, 'finish statistics streams')
    END IF
#ifdef HAVE_CDI_PIO
    IF (pio_type == pio_type_cdipio) THEN
      prev_cdi_namespace = namespaceGetActive()
      CALL namespaceSetActive(nml_io_cdi_pio_namespace)
      CALL pioFinalize
      CALL namespaceSetActive(prev_cdi_namespace)
    END IF
#endif
    ! finalize meteogram output
    IF (output_mode%l_nml) THEN
      CALL message(routine, 'finalize meteogram output')
      DO jg = 1, n_dom
        IF (meteogram_output_config(jg)%lenabled) THEN
          CALL meteogram_finalize(jg, lacc=lacc)
        END IF
      END DO
      DO jg = 1, max_dom
        DEALLOCATE(meteogram_output_config(jg)%station_list)
      END DO
    END IF


    IF (ldass_lhn) THEN
      ! delete state variables and lists for radar
      CALL destruct_radar_data()

      ! delete state variables and lists for LHN
      CALL destruct_lhn_state()
    ENDIF

    IF (assimilation_config(1)% dace_coupling .AND. my_process_is_work_only()) then
       CALL finish_dace ()
    END IF

    IF (output_mode%l_nml) THEN
      DO jg = 1, n_dom
        CALL deallocate_nh_pzlev(jg)
      ENDDO
    END IF

    CALL message(routine,'clean-up finished')

  END SUBROUTINE destruct_atmo_nonhydrostatic
  !---------------------------------------------------------------------

END MODULE mo_atmo_nonhydrostatic