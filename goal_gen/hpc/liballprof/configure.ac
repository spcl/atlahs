#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(liballprof, use VERSION !, <empty>)
AC_CONFIG_AUX_DIR(config)
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE(liballprof, 1.1)
AM_CONFIG_HEADER([config.h])


# Checks for programs.
NG_WITH_MPI
AC_PROG_CC
AC_PROG_RANLIB
NBC_CHECK_NBC

# Checks for libraries.
AC_CHECK_LIB([c], [sysctl])

# Checks for header files.
AC_CHECK_HEADERS([mpi.h])
CFLAGS="${CFLAGS} -Wno-missing-prototypes"

# Checks for library functions.
AC_PROG_GCC_TRADITIONAL
AC_FUNC_MALLOC

# defines F77 wrapper function macro F77_FUNC (name, NAME)
AC_F77_WRAPPERS

AM_DISABLE_SHARED
AM_ENABLE_STATIC
AC_PROG_LIBTOOL

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

