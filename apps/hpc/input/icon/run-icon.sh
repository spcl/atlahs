#! /bin/bash
#=============================================================================

# LUMI gpu batch job parameters
# ------------------------------
#SBATCH --job-name=icon
#SBATCH --account=g34
#SBATCH --nodes=8
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=256
#SBATCH --ntasks-per-core=2
#SBATCH --gpus-per-node=0
#SBATCH --time=01:00:00
#SBATCH --partition=amdrtx
#=============================================================================
set +x
ulimit -s unlimited
#=============================================================================
#
# ICON run script:
# !ATTENTION! Do not change the format of the following lines.
#             They are evaluated by checksuite scripts.
# created by /users/klockeda/tmp/icon-mpim/run/make_target_runscript
# target machine is lumi
# target use_compiler is cray
# with_mpi=yes
# with_openmp=no
# memory_model=large
# submit with sbatch
#
builder=generalgcc
#=============================================================================
#
# OpenMP environment variables
# ----------------------------
export OMP_NUM_THREADS=1
export ICON_THREADS=1
export OMP_SCHEDULE=static,1
export OMP_DYNAMIC="false"
export OMP_STACKSIZE=200M

rankmap_file=""
num_procs=8
output_file="/scratch/sshen/lgs-mpi-data/run.out"

# if [ $# -eq 1 ]; then
#     rankmap_file="$1"
#     echo "Rankmap file: ${rankmap_file}"
# elif [ $# -eq 2 ]; then
#     rankmap_file="$1"
#     num_procs="$2"
#     echo "Rankmap file: ${rankmap_file}"
#     echo "Number of processes: ${num_procs}"
# fi
# ====================== DEFAULT RUN COMMAND ======================
export START="mpirun --envall -env INJECTED_LATENCY 0 -env MPICH_ASYNC_PROGRESS 1 -env UCX_RNDV_THRESH 256000 -env UCX_RC_VERBS_SEG_SIZE 256000 -env OMP_NUM_THREADS 4 -np 8"
# Prints out the number of arguments
echo "[DEBUG] Number of arguments: $#"
if [ $# -eq 1 ]; then
  export START="$1"
fi

sim_time="2000-01-01T00:30:00Z"
if [ $# -eq 2 ]; then
  export START="$1"
  export sim_time="$2"
fi
echo "Run command: ${START}"
echo "Simulation end time: ${sim_time}"
# INJECTED_LATENCY=0
# if [ $# -eq 1 ]; then
#     num_procs="$1"
# elif [ $# -eq 2 ]; then
#     num_procs="$1"
#     INJECTED_LATENCY="$2"
# elif [ $# -eq 3 ]; then
#     num_procs="$1"
#     INJECTED_LATENCY="$2"
#     export START="$3"
# fi

# echo "[INFO] Number of processes: ${num_procs}"
# echo "[INFO] Injected latency: ${INJECTED_LATENCY}"


# export LD_PRELOAD=/users/sshen/workspace/lgs-mpi/liballprof/.libs/liballprof_f77.so
# export LD_PRELOAD=/users/sshen/workspace/LogGOPSim/src/liballprof2/liballprof_f77.so
# export LD_LIBRARY_PATH=/users/sshen/workspace/LogGOPSim/src/liballprof2/:${LD_LIBRARY_PATH}
# TRACE_LIB=/users/sshen/workspace/lgs-mpi/liballprof/.libs/liballprof_f77.so
TRACE_LIB=/scratch/sshen/lgs-mpi-data/liballprof2/liballprof_f77.so

# module load gcc/10.2.0
# module load python
#
# MPI variables
# -------------
no_of_nodes=1
mpi_procs_pernode=16
((mpi_total_procs=no_of_nodes * mpi_procs_pernode))
#
# blocking length
# ---------------
nproma=32
nproma_sub=
nblocks_c=0
#
#=============================================================================

# load local setting, if existing
# -------------------------------
if [ -a ../setting ]
then
  echo "Load Setting"
  . ../setting
fi

# environment variables for the experiment and the target system
# --------------------------------------------------------------
export EXPNAME="aquaplanet_04"

#=============================================================================
# directories with absolute paths
# -------------------------------
thisdir=$(pwd)
basedir=/capstor/scratch/cscs/sshen/workspace/atlahs/apps/hpc/icon
echo $basedir
# experiments_dir can be predefined in a machine specific run_target_* header
#experiments_dir="${experiments_dir:=${basedir}/experiments}"
experiments_dir=$basedir/results
# export ${basedir}
icon_data_rootFolder=$basedir/data
# Round robin option
if [ -z "$rankmap_file" ]; then
    # If the rankmap file is not provided by the user
    # num_nodes=$(( num_procs / mpi_procs_pernode ))
    # rankmap="(0,${num_nodes},${mpi_procs_pernode})"
    rankmap=""
else
    # Otherwise, read from the rankmap file
    if [ -f "$rankmap_file" ]; then
	rankmap=$(<"$rankmap_file")
	IFS=',' read -ra ADDR <<< "$rankmap"
	rank_count=${#ADDR[@]}
	# Makes sure that the number of ranks in the rankmap file is equal
	# to the total number of processes
	if [ "$rank_count" -ne "$num_procs" ]; then
	    echo "Assertion failed: Expected $num_procs ranks in the rankmap file, but found $rank_count." >&2
	    exit 1
	fi
	rankmap="-rankmap $rankmap"
    else
	echo "[ERROR] The given rankmap file does not exist: ${rankmap_file}" >&2
	exit -1
    fi
fi

# how to start the icon model
# ---------------------------
# export LD_PRELOAD=${TRACE_LIB}
# rank_file="/users/sshen/workspace/run_scripts/icon_rank"
# export START="${MPIRUN} -np $mpi_procs_pernode -rf ${rank_file}"

# export START="mpirun --envall -env INJECTED_LATENCY ${INJECTED_LATENCY} -env MPICH_ASYNC_PROGRESS 1 -env UCX_RNDV_THRESH 256000 -env UCX_RC_VERBS_SEG_SIZE 256000 -env OMP_NUM_THREADS 4 -np $num_procs"
# export HOSTS=/scratch/sshen/lgs-mpi/case-studies/icon/hosts16_openmpi
# export MPIRUN=`which mpirun`
# export NUM_RANKS=32
# export START="${MPIRUN} --mca btl ^openib -npernode 32 --mca pml ucx -x LD_LIBRARY_PATH -x INJECTED_LATENCY=0 -x UCX_RNDV_THRESH=256000 -x UCX_RC_VERBS_SEG_SIZE=256000 -H slimfly1:32 -np $NUM_RANKS"
export MODEL="${basedir}/bin/icon"

set | grep SLURM

# how to submit the next job
# --------------------------
submit="mpirun"
job_name="icon.sh"


# define script functions used in the experiment run script
# ---------------------------------------------------------
. ${basedir}/run/add_run_routines

#=============================================================================


export FI_MR_CACHE_MONITOR=memhooks
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=0
#=============================================================================
# Aqua Planet Experiment (APE)
author_list="Marco Giorgetta, Sebastian Mueller, Sebastian Rast, MPIM"
#=============================================================================
# This file describes an aqua planet experiment based on the non-hydrostatic atmosphere and the
# AES physics. The experiment is intialized analytically and uses constant-in-time
# boundary conditions for:
# - SST and sea ice
# - spectral solar irradiation
# - well mixed greenhouse gases CO2, CH4, N2O, CFCs
# - O3 concentration
# No aerosols are used
# _bb_table_Description_  aqua planet experiment
# _bb_table_Model_        non-hydrostatic atmosphere with ECHAM physics
#=============================================================================
# (1) Variables provided by the scripting mechanism
# EXPNAME                       = name of exp. in 'exp.<name>'
# basedir                       = base directory, where src/, run/ etc exist
# icon_data_poolFolder          = base directory, where grids/, input/ and setup/ exist
# nproma                        = blocking length for array dimensioning and inner loop lengths
#icon_data_poolFolder="${icon_data_rootFolder:-/pool/data/ICON}/grids/public/mpim"
#icon_data_poolFolder_gases="/pool/data/ICON/grids/private/mpim/icon_preprocessing/source"
#icon_data_poolFolderPrivate="${icon_data_rootFolder:-/pool/data/ICON}/grids/private/mpim"
#--------------------------------------------------------------------------------------------------
# domain globe
grid_id=0013
grid_refinement=R02B04
grid_label=G
grid_name=icon_grid_${grid_id}_${grid_refinement}_${grid_label}
atmo_dyn_grids="'${grid_name}.nc',"
# horizontal grid(s)
grids_folder=${icon_data_rootFolder}
# start and end date+time

start_date=${start_date:="2000-01-01T00:00:00Z"}
            end_date=${end_date:="${sim_time}"}

calendar="360 day year"

# restart intervals
checkpoint_interval="P100D"
restart_interval="PT1H"

# output intervals
output_interval="PT1H"
output_interval_3d="PT3H"
file_interval="P1H"
output_interval_day="P1D"
file_interval_day_avg="P10D"

# namelist files
atmo_namelist=NAMELIST_${EXPNAME}_atm
#--------------------------------------------------------------------------------------------------
# (2) Output control
# ------------------
# asynchronous diagnostic output processes
# ----------------------------------------
mpi_atm_io_procs=0                 # for atmosphere (2d and 3d, lnd and dyamond) (8+14
# output file selection
# ---------------------
# Note that "mpi_atm_io_procs" must match the number of output files
#
# output_<xyz>=yes : yes --> output files for <xyz>, any other value --> no files for <xyz>

output_atm_vgrid=no                # produces 1 atm file
output_atm_debug=no                # produces 1 atm file
output_atm_3d=no                   # produces 2 atm file
output_atm_2d=no                   # produces 4 atm file
output_phy_3d=no                   # produces 1 atm file
output_trc_3d=no                   # produces 1 atm file

#--------------------------------------------------------------------------------------------------
# (3) Define the model configuration

# atmospheric dynamics and physics
# --------------------------------
cat > ${atmo_namelist} << EOF
!
&parallel_nml
 nproma           = ${nproma}
 num_io_procs      = ${mpi_atm_io_procs}
! num_restart_procs = 1
 io_process_stride = 1
 io_proc_chunk_size = 31
/
&grid_nml
 dynamics_grid_filename = ${atmo_dyn_grids}
/
&run_nml
 num_lev          = 90          ! number of full levels
 modelTimeStep    = "PT15M"
 ltestcase        = .TRUE.      ! run testcase
 ldynamics        = .TRUE.      ! dynamics
 ltransport       = .TRUE.      ! transport
 iforcing         = 2           ! 0: none, 1: HS, 2: AES, 3: NWP
 output           = 'nml'
 msg_level        = 15          ! level of details report during integration
! restart_filename = "${EXPNAME}_restart_atm_<rsttime>.nc"
 activate_sync_timers = .TRUE.
/
&nh_testcase_nml
 nh_test_name     = 'APE_aes'
 ape_sst_case     = 'sst_qobs'
 zp_ape           = 101325
 ztmc_ape         = 25.006
/
&nonhydrostatic_nml
 ndyn_substeps    = 5           ! dtime/dt_dyn
 damp_height      = 50000.      ! [m]
 rayleigh_coeff   = 0.10
 vwind_offctr     = 0.2

divdamp_fac      = 0.004
/
&interpol_nml
 rbf_scale_mode_ll = 1
/
&sleve_nml
 min_lay_thckn    = 25.         ! [m]
 max_lay_thckn    = 400.   ! maximum layer thickness below htop_thcknlimit
 htop_thcknlimit  = 14000  ! this implies that the upcoming COSMO-EU nest will have 60 levels
 top_height       = 75000.      ! [m]
 stretch_fac      = 0.9
 decay_scale_1    = 4000.       ! [m]
 decay_scale_2    = 2500.       ! [m]
 decay_exp        = 1.2
 flat_height      = 16000.      ! [m]
/
&initicon_nml
 pinit_seed       = 0           ! seed for perturbation of initial model state. no perturbation by default
 pinit_amplitude  = 0.          ! amplitude of perturbation
/
&diffusion_nml
/
&transport_nml
 tracer_names     = 'hus','clw','cli', 'qr', 'qs', 'qg'
 ivadv_tracer     =    3 ,   3 ,   3 ,   3 ,   3 ,   3
 itype_hlimit     =    3 ,   4 ,   4 ,   4 ,   4 ,   4
 ihadv_tracer     =   20 ,  20 ,  20 ,  20 ,  20 ,  20
/
&aes_phy_nml
!
! domain 1
! --------
!
! atmospheric phyiscs (""=never)
 aes_phy_config(1)%dt_rad = "PT60M"
 aes_phy_config(1)%dt_vdf = "PT15M"
 aes_phy_config(1)%dt_mig = "PT15M"
!
! surface (.TRUE. or .FALSE.)
 aes_phy_config(1)%ljsb   = .FALSE.
 aes_phy_config(1)%lamip  = .FALSE.
 aes_phy_config(1)%lice   = .FALSE.
 aes_phy_config(1)%lmlo   = .FALSE.
 aes_phy_config(1)%llake  = .FALSE.
/
&aes_rad_nml
!
! domain 1
! --------
!
 aes_rad_config(1)%isolrad    =  2
 aes_rad_config(1)%cecc       = 0.0
 aes_rad_config(1)%cobld      = 0.0
 aes_rad_config(1)%irad_h2o   =  1
 aes_rad_config(1)%irad_co2   =  2
 aes_rad_config(1)%irad_ch4   =  0
 aes_rad_config(1)%irad_n2o   =  0
 aes_rad_config(1)%irad_o3    =  4
 aes_rad_config(1)%irad_o2    =  0
 aes_rad_config(1)%irad_cfc11 =  0
 aes_rad_config(1)%irad_cfc12 =  0
 aes_rad_config(1)%irad_aero  =  0
/
&aes_vdf_nml
 aes_vdf_config(1)%pr0        =  0.7
 aes_vdf_config(1)%turb       =  2     ! Smagorinsky
/
&aes_cop_nml
 aes_cop_config(1)%cn1lnd     =  50.0
 aes_cop_config(1)%cn2lnd     = 220.0
 aes_cop_config(1)%cn1sea     =  50.0
 aes_cop_config(1)%cn2sea     = 100.0
 aes_cop_config(1)%cinhomi    =   1.0
 aes_cop_config(1)%cinhoml1   =   0.66
 aes_cop_config(1)%cinhoml2   =   0.66
 aes_cop_config(1)%cinhoml3   =   0.66
/
&aes_mig_nml
 aes_mig_config(1)%mu_rain        = 0.5
 aes_mig_config(1)%rain_n0_factor = 0.1
 aes_mig_config(1)%v0snow         = 25.
! aes_mig_config(1)%zvz0i          = 3.29  ! Terminal fall velocity of ice  (original value of Heymsfield+Donner 1990: 3.29)
/
&aes_cov_nml
 ! aes_cov_config(1)%icov       = 0     ! 0/1 cloud cover based on cloud water and ice
 aes_cov_config(1)%cqx        = 1.e-6
/
EOF

#--------------------------------------------------------------------------------------------------

# (4) Define the input

# model files
#
add_link_file ${basedir}/externals/rte-rrtmgp/rrtmgp/data/rrtmgp-data-lw-g128-210809.nc         ./coefficients_lw.nc
add_link_file ${basedir}/externals/rte-rrtmgp/rrtmgp/data/rrtmgp-data-sw-g112-210809.nc         ./coefficients_sw.nc
add_link_file ${basedir}/data/ECHAM6_CldOptProps_rrtmgp_lw.nc           ./rrtmgp-cloud-optics-coeffs-lw.nc
add_link_file ${basedir}/data/ECHAM6_CldOptProps_rrtmgp_sw.nc           ./rrtmgp-cloud-optics-coeffs-sw.nc

cp ${atmo_namelist} ${basedir}/run/${atmo_namelist}
# namelist files
#
add_required_file ${basedir}/run/${atmo_namelist}                       ./

# dictionary file for output variable names
#
dict_file="dict.${EXPNAME}"
cat ${basedir}/run/dict.iconam.mpim  > ${dict_file}
cp ${dict_file} ${basedir}/run/${dict_file}
add_required_file ${basedir}/run/${dict_file}                           ./

# boundary conditions
#
# range of years for yearly files
# assume start_date and end_date have the format yyyy-...
start_year=$(( ${start_date%%-*} - 1 ))
end_year=$(( ${end_date%%-*} + 1 ))
#
# - well mixed greenhouse gases
#datadir=${icon_data_poolFolder_gases}/greenhouse_gases
#add_link_file $datadir/greenhouse_historical_plus.nc                    ./bc_greenhouse_gases.nc
#
# - ozone
datadir=${basedir}/ozone/
add_link_file ${datadir}/bc_ozone_ape.nc                                ./bc_ozone.nc

#--------------------------------------------------------------------------------------------------

# (5) Define the output

# Parameters for all output files
# -------------------------------
cat >> ${atmo_namelist} << EOF
&io_nml
 output_nml_dict  = "${dict_file}"
 netcdf_dict      = "${dict_file}"
 lnetcdf_flt64_output = .TRUE.
 itype_pres_msl   = 4
 restart_file_type= 5
 restart_write_mode = "joint procs multifile"    !not necessary/useful in default r2b4 setup
/
EOF

# Define output files
# -------------------
#
# output_<xyz>=yes : yes --> output files for <xyz>, any other value --> no files for <xyz>
#
# 3-dimensional files include 'ps' and 'pfull' to allow the vertical
# interpolation to pressure levels by cdo ap2pl.
#
if [[ "$output_atm_vgrid" == "yes" ]]; then
  #
  cat >> ${atmo_namelist} << EOF
&output_nml
 output_filename  = "${EXPNAME}_atm_vgrid"
 filename_format  = "<output_filename>_<levtype_l>"
 filetype         = 5
 remap            = 0
 output_grid      = .TRUE.
 output_start     = "${start_date}"       ! output_start = output_end
 output_end       = "${start_date}"       ! --> write once only irrespective of
 output_interval  = "${output_interval}"  !     the output interval and
 file_interval    = "${file_interval}"    !     the file interval
 ml_varlist       = 'zghalf'  , 'zg'      , 'dzghalf'
/
EOF
fi
#
if [[ "$output_atm_3d" == "yes" ]]; then
  #
  cat >> ${atmo_namelist} << EOF
&output_nml
 output_filename  = "${EXPNAME}_atm_3d"
 filename_format  = "<output_filename>_<levtype_l>_<datetime2>"
 filetype         = 5
 remap            = 0
 reg_lon_def      = -180.,3600,180.
 reg_lat_def      = -90.,1800,90.
 m_levels         = "30...(nlev-1)"
 output_grid      = .TRUE.
 output_start     = "${start_date}"
 output_end       = "${end_date}"
 output_interval  = "${output_interval_3d}"
 file_interval    = "${file_interval}"
 include_last     = .FALSE.
 ml_varlist       = 'pfull'   , 'zg'      ,
                    'rho'     , 'ta'      ,
                    'ua'      , 'va'      , 'wap'     ,
                    'hus'     , 'clw'     , 'cli'     ,
                    'qr'      , 'qs'      , 'qg'      ,
                    'cl'      ,
/
&output_nml
 output_filename  = "${EXPNAME}_dayavg_atm_3d"
 filename_format  = "<output_filename>_<levtype_l>_<datetime2>"
 filetype         = 5
 remap            = 0
 operation        = 'mean'
 output_grid      = .FALSE.
 output_start     = "${start_date}"
 output_end       = "${end_date}"
 output_interval  = "${output_interval_day}"
 file_interval    = "${file_interval_day_avg}"
 include_last     = .FALSE.
 ml_varlist       = 'pfull' , 'rho'   , 'ta'  ,
                    'ua'    , 'va'    , 'wap' ,
                    'hus'   , 'clw'   , 'cli' ,
                    'qr'    , 'qs'    , 'qg'  ,
                    'cl'    , 'cptgz' , 'rsd' ,
                    'rsu'   , 'rld'   , 'rlu' ,
/
EOF
fi
#
if [[ "$output_atm_2d" == "yes" ]]; then
  #
  cat >> ${atmo_namelist} << EOF
&output_nml
 output_filename  = "${EXPNAME}_dayavg_atm_2d"
 filename_format  = "<output_filename>_<levtype_l>_<datetime2>"
 filetype         = 5
 remap            = 0
 operation        = 'mean'
 output_grid      = .FALSE.
 output_start     = "${start_date}"
 output_end       = "${end_date}"
 output_interval  = "${output_interval_day}"
 file_interval    = "${file_interval_day_avg}"
 include_last     = .FALSE.
 ml_varlist       = 'ps'      , 
                    'uas'     , 'vas'     , 'tas'     ,
/
EOF
fi
#
if [[ "$output_phy_3d" == "yes" ]]; then
  #
  cat >> ${atmo_namelist} << EOF
&output_nml
 output_filename  = "${EXPNAME}_phy_3d"
 filename_format  = "<output_filename>_<levtype_l>_<datetime2>"
 filetype         = 5
 remap            = 0
 output_grid      = .TRUE.
 output_start     = "${start_date}"
 output_end       = "${end_date}"
 output_interval  = "${output_interval}"
 file_interval    = "${file_interval}"
 include_last     = .FALSE.
 ml_varlist       = 'ps'           , 'pfull'        , 'zg'           ,
                    'tend_ta'      , 'tend_ta_dyn'  , 'tend_ta_phy'  ,
                    'tend_ta_rlw'  , 'tend_ta_rsw'  ,
                    'tend_ta_vdf'  , 'tend_ta_gwd'  ,
                    'tend_ta_cnv'  , 'tend_ta_cld'  ,
                    'tend_ua'      , 'tend_ua_dyn'  , 'tend_ua_phy'  ,
                    'tend_ua_vdf'  , 'tend_ua_gwd'  ,
                    'tend_ua_cnv'  ,
                    'tend_va'      , 'tend_va_dyn'  , 'tend_va_phy'  ,
                    'tend_va_vdf'  , 'tend_va_gwd'  ,
                    'tend_va_cnv'  ,
                    'tend_qhus'    , 'tend_qhus_dyn', 'tend_qhus_phy',
                    'tend_qhus_cld', 'tend_qhus_cnv', 'tend_qhus_vdf',
                    !'tend_qhus_mox'
/
EOF
fi
#
if [[ "$output_trc_3d" == "yes" ]]; then
  #
  cat >> ${atmo_namelist} << EOF
&output_nml
 output_filename  = "${EXPNAME}_trc_3d"
 filename_format  = "<output_filename>_<levtype_l>_<datetime2>"
 filetype         = 5
 remap            = 0
 output_grid      = .TRUE.
 output_start     = "${start_date}"
 output_end       = "${end_date}"
 output_interval  = "${output_interval}"
 file_interval    = "${file_interval}"
 include_last     = .FALSE.
 ml_varlist       = 'ps'        , 'pfull'     , 'zg'        ,
                    'mairvi_phy',
                    'mdryvi_phy',
                    'mh2ovi_phy',
                    'qhus_phy'  , 'mhusvi_phy', 'tend_mhusvi_phy',
                    'qclw_phy'  , 'mclwvi_phy', 'tend_mclwvi_phy',
                    'qcli_phy'  , 'mclivi_phy', 'tend_mclivi_phy',
/
EOF
fi

#--------------------------------------------------------------------------------------------------
#!/bin/ksh
#=============================================================================
#
# This section of the run script prepares and starts the model integration. 
#
# MODEL and START must be defined as environment variables or
# they must be substituted with appropriate values.
#
# Marco Giorgetta, MPI-M, 2010-04-21
#
#-----------------------------------------------------------------------------
final_status_file=${basedir}/run/${job_name}.final_status
rm -f ${final_status_file}
#-----------------------------------------------------------------------------
#
# directories definition
#
RUNSCRIPTDIR=${basedir}/run
if [ x$grids_folder = x ] ; then
   HGRIDDIR=${basedir}/grids
else
   HGRIDDIR=$grids_folder
fi

make_and_change_to_experiment_dir

#-----------------------------------------------------------------------------
final_status_file=${RUNSCRIPTDIR}/${job_name}.final_status
rm -f ${final_status_file}

#-----------------------------------------------------------------------------
# set up the model lists if they do not exist
# this works for single model runs
# for coupled runs the lists should be declared explicilty
if [ x$namelist_list = x ]; then
#  minrank_list=(        0           )
#  maxrank_list=(     65535          )
#  incrank_list=(        1           )
  minrank_list[0]=0
  maxrank_list[0]=65535
  incrank_list[0]=1
  if [ x$atmo_namelist != x ]; then
    # this is the atmo model
    namelist_list[0]="$atmo_namelist"
    modelname_list[0]="atmo"
    modeltype_list[0]=1
    run_atmo="true"
  elif [ x$ocean_namelist != x ]; then
    # this is the ocean model
    namelist_list[0]="$ocean_namelist"
    modelname_list[0]="oce"
    modeltype_list[0]=2
  elif [ x$psrad_namelist != x ]; then
    # this is the psrad model
    namelist_list[0]="$psrad_namelist"
    modelname_list[0]="psrad"
    modeltype_list[0]=3
  elif [ x$hamocc_namelist != x ]; then
    # this is the hamocc model
    namelist_list[0]="$hamocc_namelist"
    modelname_list[0]="hamocc"
    modeltype_list[0]=4
  elif [ x$jsbach_namelist != x ]; then
    # this is the jsbach standalone model
    namelist_list[0]="$jsbach_namelist"
    modelname_list[0]="jsbach"
    modeltype_list[0]=5
    run_jsbach_standalone="true"
  elif [ x$testbed_namelist != x ]; then
    # this is the testbed model
    namelist_list[0]="$testbed_namelist"
    modelname_list[0]="testbed"
    modeltype_list[0]=99
  else
    check_error 1 "No namelist is defined"
  fi 
fi

#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# set some default values and derive some run parameteres
restart=${restart:=".false."}
restartSemaphoreFilename='isRestartRun.sem'
#AUTOMATIC_RESTART_SETUP:
if [ -f ${restartSemaphoreFilename} ]; then
  restart=.true.
  #  do not delete switch-file, to enable restart after unintended abort
  #[[ -f ${restartSemaphoreFilename} ]] && rm ${restartSemaphoreFilename}
fi
#END AUTOMATIC_RESTART_SETUP
#
# wait 5min to let GPFS finish the write operations
if [ "x$restart" != 'x.false.' -a "x$submit" != 'x' ]; then
  if [ x$(df -T ${EXPDIR} | cut -d ' ' -f 2) = gpfs ]; then
    sleep 10;
  fi
fi
# fill some checks

run_atmo=${run_atmo="false"}
if [ x$atmo_namelist != x ]; then
  run_atmo="true"
  run_jsbach_standalone="false"
fi
run_jsbach=${run_jsbach="false"}
if [ x$jsbach_namelist != x ]; then
  run_jsbach="true"
fi
run_ocean=${run_ocean="false"}
if [ x$ocean_namelist != x ]; then
  run_ocean="true"
fi
run_psrad=${run_psrad="false"}
if [ x$psrad_namelist != x ]; then
  run_psrad="true"
fi
run_hamocc=${run_hamocc="false"}
if [ x$hamocc_namelist != x ]; then
  run_hamocc="true"
fi

#-----------------------------------------------------------------------------
# add grids to required files
all_grids="${atmo_dyn_grids} ${atmo_rad_grids} ${ocean_grids}"
for gridfile in ${all_grids}; do
  #
  gridfile=${gridfile//\'/} # strip all ' in case ' is used to delimit the grid names
  gridfile=${gridfile//\"/} # strip all " in case " is used to delimit the grid names
  gridfile=${gridfile//\,/} # strip all , in case , is used to separate the grid names
  #
  grfinfofile=${gridfile%.nc}-grfinfo.nc
  #
  ls -l ${HGRIDDIR}/$gridfile
  check_error $? "${HGRIDDIR}/$gridfile does not exist."
  add_link_file ${HGRIDDIR}/${gridfile} ./
  if [ -f ${HGRIDDIR}/${grfinfofile} ]; then    
    add_link_file ${HGRIDDIR}/${grfinfofile} ./
  fi
done
#-----------------------------------------------------------------------------
# print_required_files
copy_required_files
link_required_files


#-----------------------------------------------------------------------------
# get restart files

if  [ x$restart_atmo_from != "x" ] ; then
  rm -f restart_atm_DOM01.nc
#  ln -s ${basedir}/experiments/${restart_from_folder}/${restart_atmo_from} ${EXPDIR}/restart_atm_DOM01.nc
  cp ${basedir}/experiments/${restart_from_folder}/${restart_atmo_from} cp_restart_atm.nc
  ln -s cp_restart_atm.nc restart_atm_DOM01.nc
  restart=".true."
fi
if  [ x$restart_ocean_from != "x" ] ; then
  rm -f restart_oce.nc
#  ln -s ${basedir}/experiments/${restart_from_folder}/${restart_ocean_from} ${EXPDIR}/restart_oce.nc
  cp ${basedir}/experiments/${restart_from_folder}/${restart_ocean_from} cp_restart_oce_DOM01.nc
  ln -s cp_restart_oce_DOM01.nc restart_oce_DOM01.nc
  restart=".true."
fi
#-----------------------------------------------------------------------------

read_restart_namelists=${read_restart_namelists:=".true."}

#-----------------------------------------------------------------------------
#
# create ICON master namelist
# ------------------------
# For a complete list see Namelist_overview and Namelist_overview.pdf

#-----------------------------------------------------------------------------
# create master_namelist
if [ -z "$dont_create_icon_master_namelist" ]; then
  master_namelist=icon_master.namelist

  calendar=${calendar:="proleptic gregorian"}
  calendar_type=${calendar_type:=1}
  {
    echo "&master_nml"
    echo " lrestart               =  $restart"
    echo " read_restart_namelists =  $read_restart_namelists"
    echo "/"

    if [ -z "$nsteps" ]; then
      echo "&master_time_control_nml"
      echo " calendar             = '$calendar'"
      echo " experimentStartDate  = '$start_date'"
      # echo " restartTimeIntval    = '$restart_interval'"
      # echo " checkpointTimeIntval = '$checkpoint_interval'"
      if [ -n "$end_date" ]; then
        echo " experimentStopDate = '$end_date'"
      fi
      echo "/"

      echo "&time_nml"
      echo " is_relative_time     = .false."
      echo "/"

    else # $nsteps is set -> use time_nml:ini_datetime_string
      echo "&time_nml"
      echo " calendar             =  $calendar_type"
      echo " ini_datetime_string  = '$start_date'"
      echo " dt_restart           =  $dt_restart"
      echo "/"
    fi
  } > $master_namelist

fi
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# add model component to master_namelist
add_component_to_master_namelist()
{
  model_namelist_filename=$1
  if [ x${dont_create_icon_master_namelist+set} != xset ]; then
    model_name=$2
    model_type=$3
    model_min_rank=$4
    model_max_rank=$5
    model_inc_rank=$6
    model_rank_group_size=$7
    cat >> $master_namelist << EOF
&master_model_nml
  model_name="$model_name"
  model_namelist_filename="$model_namelist_filename"
  model_type=$model_type
  model_min_rank=$model_min_rank
  model_max_rank=$model_max_rank
  model_inc_rank=$model_inc_rank
  model_rank_group_size=$model_rank_group_size
/
EOF
  fi

  #-----------
  #get namelist
  if [ -f ${RUNSCRIPTDIR}/$model_namelist_filename ] ; then
    mv -f ${RUNSCRIPTDIR}/$model_namelist_filename ${EXPDIR}
    check_error $? "mv -f ${RUNSCRIPTDIR}/$model_namelist_filename ${EXPDIR}"
  else
    check_error 1 "${RUNSCRIPTDIR}/$model_namelist_filename does not exist"
  fi
}
#-----------------------------------------------------------------------------


no_of_models=${#namelist_list[*]}
echo "no_of_models=$no_of_models"

rank_group_size=1
j=0
while [ $j -lt ${no_of_models} ]
do
  add_component_to_master_namelist "${namelist_list[$j]}" "${modelname_list[$j]}" ${modeltype_list[$j]} ${minrank_list[$j]} ${maxrank_list[$j]} ${incrank_list[$j]} ${rank_group_size}
  j=`expr ${j} + 1`
done

#-----------------------------------------------------------------------------
# Add JSBACH part to master_namelist
# For several domains, $jsbach_namelist is only the basename for each domain's jsbach namelist;
#   the actual namelist files are appended by suffixes '_d1', '_d2', etc.

if [[ $run_jsbach == @(yes|true) ]]; then
  cat >> $master_namelist << EOF
&jsb_control_nml
 is_standalone      = .${run_jsbach_standalone:=false}.
 restart_jsbach     = ${restart}
 debug_level        = 0
 timer_level        = 0
EOF
if [[ ${run_jsbach_standalone} == true ]]; then
  cat >> $master_namelist << EOF
 l_force_from_obs   = .${l_force_from_obs:-false}.
EOF
fi
  cat >> $master_namelist << EOF
/
EOF
#
if [[ -n ${atmo_dyn_grids} ]]; then
  no_of_domains=${#atmo_dyn_grids[@]}
else
  no_of_domains=1
fi
echo "no_of_domains=$no_of_domains"
domain=""
domain_suffix=""
j=1
while [ $j -le ${no_of_domains} ]
do
  if [[ $no_of_domains -gt 1 ]]; then
    # no_of_domains < 10 !
    domain=" DOM0${j}"
    domain_suffix="_d${j}"
  fi
  cat >> $master_namelist << EOF
&jsb_model_nml
 model_id = $j
 model_name = "JSBACH${domain}"
 model_shortname = "jsb${domain_suffix}"
 model_description = 'JSBACH land surface model'
 model_namelist_filename = "${jsbach_namelist}${domain_suffix}"
/
EOF
  if [[ ${run_jsbach_standalone} != true ]]; then
    if [[ -f ${RUNSCRIPTDIR}/${jsbach_namelist}${domain_suffix} ]] ; then
      mv ${RUNSCRIPTDIR}/${jsbach_namelist}${domain_suffix} ${EXPDIR}
      check_error $? "mv ${RUNSCRIPTDIR}/${jsbach_namelist}${domain_suffix}"
    else
      check_error 1 "${RUNSCRIPTDIR}/${jsbach_namelist}${domain_suffix} does not exist"
    fi
  fi
  j=`expr ${j} + 1`
done
fi
#
#  get model
#
ls -l ${MODEL}
check_error $? "${MODEL} does not exist?"
#
ldd ${MODEL}
#
#-----------------------------------------------------------------------------
#
# start experiment
#

rm -f finish.status
#
date
set -x
# ${START} ${MODEL} 2>&1 | tee -a $output_file || exit 1
${START} ${MODEL}
set +x
date
#
if [ -r finish.status ] ; then
  check_final_status 0 "${START} ${MODEL}"
else
  check_final_status -1 "${START} ${MODEL}"
fi
#
#-----------------------------------------------------------------------------
#
finish_status=`cat finish.status`
echo $finish_status
echo "============================"
echo "Script run successfully: $finish_status"
echo "============================"

# Do not restart regardless of the finish status
finish_status="DO NOT RESTART"

#-----------------------------------------------------------------------------
# rm output_schedule_steps*
#-----------------------------------------------------------------------------
if [[ "x$use_hamocc" = "xyes" ]]; then
# store HAMOCC log file
strg="$(ls -rt ${EXPNAME}_hamocc_EU*.nc* | tail -1 )"
prefx="${EXPNAME}_hamocc_EU_tendencies"
foo=${strg##${prefx}}
foo=${foo%%.*}
bgcout_file="bgcout_${foo}"
mv bgcout $bgcout_file
fi
#-----------------------------------------------------------------------------
namelist_list=""
#-----------------------------------------------------------------------------
# check if we have to restart, ie resubmit
#   Note: this is a different mechanism from checking the restart
if [ "$finish_status" = "RESTART" ] ; then
  echo "restart next experiment..."
  this_script="${RUNSCRIPTDIR}/${job_name}"
  echo 'this_script: ' "$this_script"
  touch ${restartSemaphoreFilename}
  cd ${RUNSCRIPTDIR}
  ${submit} $this_script $run_param_0
else
  [[ -f ${restartSemaphoreFilename} ]] && rm ${restartSemaphoreFilename}
fi

#-----------------------------------------------------------------------------
# automatic call/submission of post processing if available
if [ "x${autoPostProcessing}" = "xtrue" ]; then
  # check if there is a postprocessing is available
  cd ${RUNSCRIPTDIR}
  targetPostProcessingScript="./post.${EXPNAME}.run"
  [[ -x $targetPostProcessingScript ]] && ${submit} ${targetPostProcessingScript}
  cd -
fi

#-----------------------------------------------------------------------------

cd $RUNSCRIPTDIR

#-----------------------------------------------------------------------------

	
exit 0
#
# vim:ft=sh
#-----------------------------------------------------------------------------