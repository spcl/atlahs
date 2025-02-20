import os
import yaml
import argparse

# Script that builds all the benchmarks

def print_warning(message: str):
    """
    Prints a warning message in color orange.
    """
    CSTART = '\033[93m'
    CEND = '\033[0m'
    print(f"{CSTART}[WARNING] {message}{CEND}")


def print_error(message: str):
    """
    Prints an error message in color red.
    """
    CSTART = '\033[91m'
    CEND = '\033[0m'
    print(f"{CSTART}[ERROR] {message}{CEND}")


def print_success(message: str):
    """
    Prints a success message in color green.
    """
    CSTART = '\033[92m'
    CEND = '\033[0m'
    print(f"{CSTART}[SUCCESS] {message}{CEND}")



def build_lulesh(jobs: int = 1,
                 verbose: bool = False) -> bool:
    """
    Clones and builds the LULESH benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building LULESH...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("lulesh"):
        print_error("Directory 'lulesh' does not exist. Make sure to clone the submodule")
        return False
    
    # Changes the directory to lulesh
    os.chdir("lulesh")
    # Removes the build directory if it exists
    if os.path.exists("build"):
        if os.system("rm -rf build") != 0:
            print_error("Failed to remove the build directory")
            return False

    # Runs the command "mkdir build" to create a build directory
    if os.system("mkdir build") != 0:
        print_error("Failed to create the build directory")
        return False
    
    # Changes the directory to build
    os.chdir("build")

    # Runs the command "cmake" to configure the build
    command = f"cmake .. -DWITH_MPI=ON -DWITH_OPENMP=ON"
    
    if os.system(command) != 0:
        print_error("Failed to configure the build")
        return False

    # Runs the command "make" to build the benchmark
    if os.system(f"make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    assert os.path.exists("lulesh2.0"), "lulesh executable does not exist"

    # Changes the directory to the parent directory
    os.chdir("../../")

    return True


def build_npb(jobs: int = 1,
              verbose: bool = False) -> bool:
    """
    Clones and builds the NPB benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building NPB benchmark...")
    
    npb_name = f"NPB3.4.3"
    download_link = "https://www.nas.nasa.gov/assets/npb/NPB3.4.3.tar.gz"

    # Makes sure that the directory is already cloned
    if not os.path.exists(npb_name):
        # Removes the tar file if it exists
        if verbose:
            print(f"[INFO] Downloading the NPB benchmark from '{download_link}'...")
        # Downloads the NPB benchmark from the given URL
        if os.system(f"wget {download_link}") != 0:
            print_error("Failed to download the NPB benchmark")
            return False

        # Extracts the tar file to a directory named `npb_name`
        if os.system(f"tar -xf {npb_name}.tar.gz") != 0:
            print_error("Failed to extract the NPB benchmark")
            return False
    
    # Changes the directory to `{npb_name}`
    os.chdir(npb_name)
    # Changes the directory to the subdirectory that ends in "-MPI"
    os.chdir([d for d in os.listdir() if d.endswith("-MPI")][0])

    # Copies the configuration 'make.def.template' to 'make.def'
    if os.system("cp config/make.def.template config/make.def") != 0:
        print_error("Failed to copy the configuration file")
        return False
    
    # Copies the suite configuration file to the current directory
    if os.system(f"cp config/suite.def.template config/suite.def") != 0:
        print_error("Failed to copy the suite configuration file")
        return False
    
    benchmarks = ["ft", "mg", "sp", "lu", "bt", "is", "ep", "cg", "ua"]
    # Updates the suite configuration file with the benchmarks
    CLASS = "C"
    with open("config/suite.def", "w") as file:
        for benchmark in benchmarks:
            file.write(f"{benchmark}\t{CLASS}\n")

    # Compiles ft separately
    if os.system(f"make ft CLASS={CLASS}") != 0:
        print_error("Failed to build the benchmark 'ft'")
        return False
        
    # Executes the command 'make clean'
    if os.system("make clean") != 0:
        print_error("Failed to clean the directory")
        return False

    # Executes the command to build the benchmark
    if os.system(f"make suite -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    # Changes the directory to the parent directory
    os.chdir("../../")

    return True


def build_hpcg(jobs: int = 1,
               verbose: bool = False) -> bool:
    """
    Clones and builds the HPCG benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building HPCG...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("hpcg"):
        print_error(f"Directory 'hpcg' does not exist. Make sure to clone the submodule")
        return False
    
    # Changes the directory to `HPCG-benchmark`
    os.chdir("hpcg")

    # Deletes the build directory if it exists
    if os.path.exists("build"):
        if os.system("rm -rf build") != 0:
            print_error("Failed to remove the build directory")
            return False
    
    # Creates the build directory
    if os.system("mkdir build") != 0:
        print_error("Failed to create the build directory")
        return False
    
    arch = "MPI_GCC_OMP"
    print_warning(f"Make sure that all parameters are specified in the file setup/Make.{arch}")


    # Changes the directory to `build`
    os.chdir("build")
    
    # Runs the configure script
    if os.system(f"../configure {arch}") != 0:
        print_error("Failed to run the configure script")
        return False
    
    if os.system(f"CXX=mpicxx make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False

    # Changes the directory to the parent directory
    os.chdir("../../")

    return True


def build_lammps(jobs: int = 1,
                 verbose: bool = False) -> bool:
    """
    Builds the LAMMPS application.
    """
    print("[INFO] Building LAMMPS...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("lammps"):
        print_error("Directory 'lammps' does not exist. Make sure to clone the submodule")
        return False
    
    # Changes the directory to `lammps`
    os.chdir("lammps")

    # Makes sure that the build directory does not exist
    if os.path.exists("build"):
        if os.system("rm -rf build") != 0:
            print_error("Failed to remove the build directory")
            return False
    
    # Creates the build directory
    if os.system("mkdir build") != 0:
        print_error("Failed to create the build directory")
        return False

    # Changes the directory to `build`
    os.chdir("build")

    # Runs the command "cmake" to configure the build
    if os.system("cmake ../cmake -DBUILD_MPI=yes -DBUILD_OMP=yes -DPKG_MANYBODY=yes") != 0:
        print_error("Failed to configure the build")
        return False

    # Runs the command "make" to build the benchmark
    if os.system(f"make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    # Asserts that the 'lmp' executable exists
    assert os.path.exists("lmp"), "lmp executable does not exist"

    # Changes the directory to the parent directory
    os.chdir("../../")
    return True


def build_milc(jobs: int = 1,
               verbose: bool = False) -> bool:
    """
    Builds the MILC application.
    """
    print("[INFO] Building MILC...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("milc_qcd"):
        print_error("Directory 'milc_qcd' does not exist. Make sure to clone the submodule")
        return False
    
    os.chdir("milc_qcd")

    # Checks if the scidac directory exists
    if os.path.exists("scidac"):
        # Removes the scidac directory
        if os.system("rm -rf scidac") != 0:
            print_error("Failed to remove the scidac directory")
            return False
    
    # Creates the scidac directory
    if os.system("mkdir scidac") != 0:
        print_error("Failed to create the scidac directory")
        return False

    print("[INFO] Building SCIDAC dependencies")
    # Changes the directory to `scidac`
    os.chdir("scidac")
    # Clones the https://github.com/usqcd-software/qmp.git repository
    if os.system("git clone https://github.com/usqcd-software/qmp.git") != 0:
        print_error("Failed to clone the qmp repository")
        return False
    
    # Changes the directory to `qmp`
    os.chdir("qmp")
    # Runs the command "autoreconf -f -i" to generate the configure script
    if os.system("autoreconf -f -i") != 0:
        print_error("Failed to generate the configure script")
        return False

    # Runs the command "./configure --prefix=$PWD/install" to configure the build
    if os.system("./configure --prefix=$PWD/install CC=mpicc --with-qmp-comms-type=MPI") != 0:
        print_error("Failed to configure the build")
        return False
    
    # Runs the command "make" to build the benchmark
    if os.system(f"make -j {jobs}") != 0:
        print_error("Failed to build QMP")
        return False

    # Runs the command "make install" to install the benchmark
    if os.system("make install") != 0:
        print_error("Failed to install QMP")
        return False

    # Asserts that 'qmp-config' exists
    assert os.path.exists("install/bin/qmp-config"), "qmp-config does not exist"

    # Changes the directory to the ROOT directory of MILC
    os.chdir("../../")

    # Replaces line 403 in the generic/com_qmp.c file with the following:
    # '  if(provided < required){'
    if os.system("sed -i '403s/.*/  if(provided < required){/' generic/com_qmp.c") != 0:
        print_error("Failed to replace line 402 in the com_qmp.c file")
        return False

    # Replaces line 28 in the Makefile with the following:
    # MPP ?= true
    if os.system("sed -i '28s/.*/MPP ?= true/' Makefile") != 0:
        print_error("Failed to replace line 28 in the Makefile")
        return False
    
    # Replaces line 107 in the Makefile with the following:
    # OMP ?= true
    if os.system("sed -i '107s/.*/OMP ?= true/' Makefile") != 0:
        print_error("Failed to replace line 107 in the Makefile")
        return False

    # Replaces line 293 in the Makefile with the following:
    # WANTQMP ?= true
    if os.system("sed -i '293s/.*/WANTQMP ?= true/' Makefile") != 0:
        print_error("Failed to replace line 293 in the Makefile")
        return False
    
    # Replaces line 301 in the Makefile with the following:
    # SCIDAC = $(shell pwd)/../scidac
    if os.system("sed -i '301s/.*/SCIDAC = $(shell pwd)\/..\/scidac/' Makefile") != 0:
        print_error("Failed to replace line 301 in the Makefile")
        return False

    # Replaces line 302 in the Makefile with the following:
    # TAG=/install
    if os.system("sed -i '302s/.*/TAG=\/install/' Makefile") != 0:
        print_error("Failed to replace line 302 in the Makefile")   
        return False

    # Copies the Makefile to the directory ks_imp_dyn
    if os.system("cp Makefile ks_imp_dyn/") != 0:
        print_error("Failed to copy the Makefile to the ks_imp_dyn directory")
        return False
    
    # Changes the directory to `ks_imp_dyn`
    os.chdir("ks_imp_dyn")

    # Checks if reunitarize_ks.o already exists in the file Make_template
    # If it does not exist, inserts the following line to line 46 of Make_template
    # '  reunitarize_ks.o \'
    if os.system("grep -q reunitarize_ks.o Make_template || sed -i '46i\\  reunitarize_ks.o \\\\' Make_template") != 0:
        print_error("Failed to insert the line 'reunitarize_ks.o' to Make_template")
        return False

    # Runs the command "make clean"
    if os.system("make clean") != 0:
        print_error("Failed to clean the directory")
        return False

    # Runs the command "make su3_rmd" to build the application
    if os.system(f"make su3_rmd -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False

    assert os.path.exists("su3_rmd"), "su3_rmd executable does not exist"

    os.chdir("../")

    print("[INFO] Downloading dataset...")
    # Checks if the directory "lattices" exists
    if os.path.exists("lattices"):
        # Removes the directory "lattices"
        if os.system("rm -rf lattices") != 0:
            print_error("Failed to remove the lattices directory")
            return False
    
    os.mkdir("lattices")
    # Changes the directory to "lattices"
    os.chdir("lattices")

    # Downloads the lattice file from
    # https://portal.nersc.gov/project/m888/apex/MILC_lattices/16x16x16x16.chklat
    if os.system("wget https://portal.nersc.gov/project/m888/apex/MILC_lattices/16x16x16x16.chklat") != 0:
        print_error("Failed to download the lattice file")
        return False
    # Stores the absolute path of the lattice file in a variable
    lattice_file = os.path.abspath("16x16x16x16.chklat")
    assert os.path.exists("16x16x16x16.chklat"), "Lattice file does not exist"
    os.chdir("../")
    # Prints the current working directory
    print(f"[INFO] Current working directory: {os.getcwd()}")
    # Inserts a '\' in front of every '/' in lattice_file
    lattice_file = lattice_file.replace("/", "\/")
    # Changes line 25 of the configuration file ../../validation/milc/milc.in
    # to the following:
    # reload_parallel $lattice_file
    if os.system(f"sed -i '25s/.*/reload_parallel {lattice_file}/' ../input/milc/milc.in") != 0:
        print_error("Failed to replace line 25 in the configuration file")
        return False

    os.chdir("../../")
    return True

def build_icon_dependencies(jobs: int = 1,
                            verbose: bool = False) -> bool:
    """
    Builds the dependencies for ICON, namely HDF5, NetCDF-C and NetCDF-Fortran.
    """
    print_warning("Make sure to source the environment variables in env.sh")
    
    def hdf5_installed() -> bool:
        """
        Checks if HDF5 is installed.
        """
        return os.system("h5pfc --version") == 0

    def netcdf_c_installed() -> bool:
        """
        Checks if NetCDF-C is installed.
        """
        return os.system("nc-config --version") == 0
    
    def netcdf_fortran_installed() -> bool:
        """
        Checks if NetCDF-Fortran is installed.
        """
        return os.system("nf-config --version") == 0

    print("[INFO] Building ICON dependencies...")
    # Makes sure that the directory is already cloned
    os.chdir("deps")
    
    hdf5_install_dir = os.environ.get("HDF5_INSTALL_DIR", os.path.abspath("hdf5/install"))
    netdf_c_install_dir = os.environ.get("NETCDF_C_INSTALL_DIR", os.path.abspath("netcdf-c/install"))
    netcdf_fortran_install_dir = os.environ.get("NETCDF_FORTRAN_INSTALL_DIR", os.path.abspath("netcdf-fortran/install"))

    if hdf5_installed():
        print_success("HDF5 is already installed")
    else:
        print("[INFO] Building HDF5...")
        # Changes the directory to `hdf5`
        if not os.path.exists("hdf5"):
            print_error("Directory 'hdf5' does not exist. Make sure to clone the submodule")
            return False
        
        os.chdir("hdf5")

        if os.system("./autogen.sh") != 0:
            print_error("Failed to run the autogen script")
            return False
        
        config_cmd = f"CFLAGS=-fPIC ./configure --enable-tests=no --enable-shared --enable-parallel --enable-fortran --enable-fortran2003 --prefix=`pwd`/install"

        if os.system(config_cmd) != 0:
            print_error("Failed to configure the build")
            return False
        
        if os.system(f"make clean && make -j {jobs}") != 0:
            print_error("Failed to build HDF5")
            return False
        
        if os.system("make install") != 0:
            print_error("Failed to install HDF5")
            return False
        
        os.environ["HDF5_INSTALL_DIR"] = hdf5_install_dir
        os.environ["PATH"] = f"{hdf5_install_dir}/bin:{os.environ['PATH']}"

        # Makes sure that HDF5 is actually installed
        if os.system("h5pfc -v") != 0:
            print_error("Failed to verify the installation of HDF5")
            return False

        os.chdir("../")
        print_success("Built HDF5")

    # Changes the directory to `netcdf-c`
    
    if netcdf_c_installed():
        print_success("NetCDF-C is already installed")
    else:
            
        print("[INFO] Building NetCDF-C...")
        if not os.path.exists("netcdf-c"):
            print_error("Directory 'netcdf-c' does not exist. Make sure to clone the submodule")
            return False

        os.chdir("netcdf-c")

        config_cmd = f"CC=mpicc LDFLAGS=-L{hdf5_install_dir}/lib LIBS=-lhdf5 CPPFLAGS=-I{hdf5_install_dir}/include ./configure --enable-parallel-tests --prefix={netdf_c_install_dir} --disable-libxml2"

        if os.system(config_cmd) != 0:
            print_error("Failed to configure the build")
            return False
        
        if os.system(f"make clean && make -j {jobs}") != 0:
            print_error("Failed to build NetCDF-C")
            return False

        if os.system("make install") != 0:
            print_error("Failed to install NetCDF-C")
            return False
        
        os.environ["NETCDF_C_INSTALL_DIR"] = netdf_c_install_dir
        os.environ["PATH"] = f"{netdf_c_install_dir}/bin:{os.environ['PATH']}"

        # Makes sure that NetCDF-C is actually installed
        if os.system("nc-config --version") != 0:
            print_error("Failed to verify the installation of NetCDF-C")
            return False
        
        os.chdir("../")

        print_success("Built NetCDF-C")

    if netcdf_fortran_installed():
        print_success("NetCDF-Fortran is already installed")
        os.chdir("../")
        return True
    # Changes the directory to `netcdf-fortran`
    print("[INFO] Building NetCDF-Fortran...")
    if not os.path.exists("netcdf-fortran"):
        print_error("Directory 'netcdf-fortran' does not exist. Make sure to clone the submodule")
        return False

    os.chdir("netcdf-fortran")

    config_cmd = f"CC=mpicc FC=mpifort LIBS=-lnetcdf CPPFLAGS=-I{netdf_c_install_dir}/include LDFLAGS=-L{netdf_c_install_dir}/lib ./configure --enable-parallel-tests --prefix={netcdf_fortran_install_dir}"

    if os.system(config_cmd) != 0:
        print_error("Failed to configure the build")
        return False
    
    if os.system(f"make clean && make -j {jobs}") != 0:
        print_error("Failed to build NetCDF-Fortran")
        return False

    if os.system("make install") != 0:
        print_error("Failed to install NetCDF-Fortran")
        return False
    
    os.environ["NETCDF_FORTRAN_INSTALL_DIR"] = netcdf_fortran_install_dir
    os.environ["PATH"] = f"{netcdf_fortran_install_dir}/bin:{os.environ['PATH']}"

    # Makes sure that NetCDF-Fortran is actually installed
    if os.system("nf-config --version") != 0:
        print_error("Failed to verify the installation of NetCDF-Fortran")
        return False

    os.chdir("../../")
    return True



def build_icon(jobs: int = 1,
               verbose: bool = False) -> bool:
    """
    Builds the ICON application.
    """
    if not build_icon_dependencies(jobs, verbose):
        print_error("Failed to build the dependencies for ICON")
        return False

    print("[INFO] Building ICON...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("icon"):
        # Print current working directory
        print_error("Directory 'icon' does not exist. Make sure to clone the submodule")
        return False
    
    # Changes the directory to `icon`
    os.chdir("icon")

    # Checks if the environment variable 'NETCDF_C_INSTALL_DIR' and 'NETCDF_FORTRAN_INSTALL_DIR' exist
    if "NETCDF_C_INSTALL_DIR" not in os.environ:
        os.environ["NETCDF_C_INSTALL_DIR"] = os.path.abspath("../deps/netcdf-c/install")
    
    if "NETCDF_FORTRAN_INSTALL_DIR" not in os.environ:
        os.environ["NETCDF_FORTRAN_INSTALL_DIR"] = os.path.abspath("../deps/netcdf-fortran/install")

    # Configures the build
    config_cmd = './configure --disable-loop-exchange --disable-jsbach --enable-mpi '\
    '--disable-gpu CFLAGS="-I${NETCDF_C_INSTALL_DIR}/include" FCFLAGS="-g -fallow-argument-mismatch '\
    '-I${NETCDF_FORTRAN_INSTALL_DIR}/include -I${NETCDF_C_INSTALL_DIR}/include" '\
    'LDFLAGS="-L${NETCDF_C_INSTALL_DIR}/lib -L${NETCDF_FORTRAN_INSTALL_DIR}/lib" '\
    'LIBS="-lnetcdff -lnetcdf -lopenblas" CC=mpicc FC=mpif90 '\
    '--disable-rte-rrtmgp --disable-mpi-rget --disable-coupling --enable-openmp'
    print(f"[INFO] Config command: {config_cmd}")
    if os.system(config_cmd) != 0:
        print_error("Failed to configure the build")
        return False

    mpi_src_dir = "../input/icon/src"
    # Copies the file from '../../case-studies/icon/icon-src/mo_mpi.f90' to
    # src/mo_mpi.f90
    if os.system(f"cp {mpi_src_dir}/mo_mpi.f90 src/parallel_infrastructure/mo_mpi.f90") != 0:
        print_error("Failed to copy the file 'mo_mpi.f90'")
        return False
    
    # Copies the file from '../../case-studies/icon/icon-src/mo_atmo_nonhydrostatic.f90' to
    # src/drivers/mo_atmo_nonhydrostatic.f90
    if os.system(f"cp {mpi_src_dir}/mo_atmo_nonhydrostatic.f90 src/drivers/mo_atmo_nonhydrostatic.f90") != 0:
        print_error("Failed to copy the file 'mo_atmo_nonhydrostatic.f90'")
        return False

    # Runs 'make clean'
    if os.system("make clean") != 0:
        print_error("Failed to clean the directory")
        return False

    # Builds the application
    if os.system(f"make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    assert os.path.exists("bin/icon"), "icon executable does not exist"

    # Records the current working directory
    icon_dir = os.getcwd()

    # Checks if the directory 'data' exists
    if os.path.exists("data"):
        # Removes the directory 'data'
        if os.system("rm -rf data") != 0:
            print_error("Failed to remove the data directory")
            return False
    os.mkdir("data")

    # Checks if the directory 'results' exists
    if os.path.exists("results"):
        # Removes the directory 'results'
        if os.system("rm -rf results") != 0:
            print_error("Failed to remove the results directory")
            return False
    os.mkdir("results")


    # Checks if the directory "ozone" exists
    if os.path.exists("ozone"):
        # Removes the directory "ozone"
        if os.system("rm -rf ozone") != 0:
            print_error("Failed to remove the ozone directory")
            return False
    os.mkdir("ozone")

    data_dir = "../input/icon/data"
    # Copies the grid files from ../../case-studies/icon/data to the appropriate directories
    if os.system(f"cp {data_dir}/icon_grid* data/") != 0:
        print_error("Failed to copy the grid files")
        return False

    if os.system(f"cp {data_dir}/bc_ozone* ozone/") != 0:
        print_error("Failed to copy the boundary condition files")
        return False

    # Changes all the '/' in icon_dir to '\/' in the path
    icon_dir = icon_dir.replace("/", "\/")
    # Changes the directory to the validation directory
    os.chdir("../input/icon")

    print_warning("Make sure to change the paths and parameters in the run-icon.sh file")
    assert os.path.exists("run-icon.sh"), "run-icon.sh does not exist"

    # Changes line 125 of the file run-icon.sh to the following:
    # 'basedir=$icon_dir'
    if os.system(f"sed -i '125s/.*/basedir={icon_dir}/' run-icon.sh") != 0:
        print_error("Failed to replace line 106 in the file run-icon.sh")
        return False
    
    # Changes line 129 of the file run-icon.sh to the following:
    # 'experiments_dir=$basedir\/results'
    if os.system(f"sed -i '129s/.*/experiments_dir=$basedir\/results/' run-icon.sh") != 0:
        print_error("Failed to replace line 110 in the file run-icon.sh")
        return False
    
    # Changes line 131 of the file run-icon.sh to the following:
    # 'icon_data_rootFolder=$basedir\/data'
    if os.system(f"sed -i '131s/.*/icon_data_rootFolder=$basedir\/data/' run-icon.sh") != 0:
        print_error("Failed to replace line 112 in the file run-icon.sh")
        return False

    # Changes the directory to the parent directory
    os.chdir("../")
    return True



# The build functions for each benchmark
build_funcs = {
    "lulesh": build_lulesh,
    "npb": build_npb,
    "hpcg": build_hpcg,
    "milc": build_milc,
    "lammps": build_lammps,
    "icon": build_icon,
}


def build_app(app: str,
              verbose: bool = False,
              jobs: int = 1):
    """
    Build the given application.
    """
    print(f"[INFO] Build application: {app}")


    if app == "all":
        apps = [ "lulesh", "npb", "hpcg", "lammps", "milc", "icon" ]
    else:
        apps = [app]

    for app in apps:
        if app not in build_funcs:
            print_warning(f"Unknown benchmark '{app}'. Skipping...")
            continue
        
        if not build_funcs[app](jobs, verbose):
            print_error(f"Failed to build the benchmark '{app}'")
            return False
        
        print_success(f"Built the application '{app}'")
    
    print(f"[INFO] Build {app}: COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build all the benchmarks')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--app', default='all', help='The application to build, options are [lulesh, npb, hpcg, lammps, icon]')
    parser.add_argument('-j', '--jobs', type=int, default=32, help='Number of jobs to run in parallel while compiling')
    args = parser.parse_args()

    build_app(app=args.app, verbose=args.verbose, jobs=args.jobs)