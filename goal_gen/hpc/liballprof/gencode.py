############################################################################################
# liballprof MPIP Wrapper generator script 
# 
# generates C and F77 wrappers for all functions in mpi.h (which was
# originally stolen from OMPI)
# timos: now from mpich because ompi is not const-correct
#
# Copyright: Indiana University
# Author: Torsten Hoefler <htor@cs.indiana.edu>
#
############################################################################################

import sys, os, re, string

# erases the spaces at the beginning of the string/line
def stripspaces(str):
  # erase whitespaces at beginning of line 
  p = re.compile("^[\s]*");
  str = p.sub( '', str)
  return str

# erase * in pointer arguments :)
def stripasterisk(str):
      p = re.compile("^[\*]+");
      str = p.sub( '', str)
      return str

# prints an array of strings comma separated
def printstrings(params):
  str = ""
  for k in range(0,len(params)):
    str = str + stripspaces(params[k])
    if(k != len(params)-1):
        str = str + ", "
  return str      

##################################################################
# ROUTINES to print the different argument types in C bindings
##################################################################
# generates C-Code that prints a pointer to the bufptr ...
def printpointer(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

# generates C-Code that prints an integer to the bufptr ...
def printint(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)'+ptr+');'
  return str

# generates C-Code that prints an MPI_Datatype to the bufptr ...
def printdatatype(ptr):
  str = ""
  #str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  str = str + """
bufptr += printdatatype("""+ptr+""", (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size;
  MPI_Aint extent;
  PMPI_Type_size ("""+ptr+""", &size );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  PMPI_Type_extent("""+ptr+""", &extent );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}
"""
  
  return str

# generates C-Code that prints an MPI_Comm to the bufptr ...
def printcomm(ptr):
  str = ""
  str = str + """
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)"""+ptr+""");
{
  int i;
  PMPI_Comm_rank("""+ptr+""", &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  PMPI_Comm_size("""+ptr+""", &i);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}
"""
  return str

# generates C-Code that prints a {send,recv}count to the bufptr ...
def printcounts(ptr):
  str = ""
  str = str + """
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)"""+ptr+""");
{
  int i,p;
  PMPI_Comm_size(comm, &p);
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", """+ptr+"""[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", """+ptr+"""[i]);
  }
}
"""
  return str

# generates C-Code that prints an MPI_Op to the bufptr ...
def printop(ptr):
  str = ""
  str = str + 'bufptr += printop('+ptr+', (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));'
  return str

# generates C-Code that prints an MPI_Win to the bufptr ...
def printwin(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

# generates C-Code that prints an MPI_Info to the bufptr ...
def printinfo(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

# generates C-Code that prints an MPI_Group to the bufptr ...
def printgroup(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

# generates C-Code that prints an MPI_Errhandler to the bufptr ...
def printerrhndl(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

# generates C-Code that prints an MPI_File to the bufptr ...
def printfile(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

# generates C-Code that prints an MPI_Offset to the bufptr ...
def printoffset(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

# generates C-Code that prints an MPI_Request to the bufptr ...
def printrequest(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

# generates C-Code that prints an MPI_Status to the bufptr ...
def printstatus(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

##################################################################
# ROUTINES to print the different argument types in F77 bindings
##################################################################
# generates C-Code that prints a pointer to the bufptr ...
def printf77pointer(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)'+ptr+');'
  return str

# generates C-Code that prints an integer to the bufptr ...
def printf77int(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%li", (long)*'+ptr+');'
  return str

# generates C-Code that prints an MPI_Datatype to the bufptr ...
def printf77datatype(ptr):
  str = ""
  #str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  str = str + """
bufptr += printdatatype(MPI_Type_f2c(*"""+ptr+"""), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));
{
  int size, iierr;
  MPI_Aint extent;
  F77_FUNC(pmpi_type_size, PMPI_TYPE_SIZE) ("""+ptr+""", &size, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", size);
  F77_FUNC(pmpi_type_extent, PMPI_TYPE_EXTENT)("""+ptr+""", &extent, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%lu", (VOLATILE unsigned long)extent);
}
"""
  return str

# generates C-Code that prints an MPI_Comm to the bufptr ...
def printf77comm(ptr):
  str = ""
  str = str + """
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *"""+ptr+""");
{
  int i, iierr;
  F77_FUNC(pmpi_comm_rank, PMPI_COMM_RANK)("""+ptr+""", &i, &iierr );
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)("""+ptr+""", &i, &iierr);
  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", i);
}
"""
  return str

# generates C-Code that prints a {send,recv}count to the bufptr ...
def printf77counts(ptr):
  str = ""
  str = str + """
bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%lu", (VOLATILE unsigned long)"""+ptr+""");
{
  int i,p,iierr;
  F77_FUNC(pmpi_comm_size, PMPI_COMM_SIZE)(comm, &p, &iierr );
  for(i=0;i<p;i++) {
    if(i==0)
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", """+ptr+"""[i]);
    else
      bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ",%i", """+ptr+"""[i]);
  }
}
"""
  return str


# generates C-Code that prints an MPI_Op to the bufptr ...
def printf77op(ptr):
  str = ""
  #str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  str = str + 'bufptr += printop(MPI_Op_f2c(*'+ptr+'), (char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf));'
  return str

# generates C-Code that prints an MPI_Win to the bufptr ...
def printf77win(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  return str

# generates C-Code that prints an MPI_Info to the bufptr ...
def printf77info(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  return str

# generates C-Code that prints an MPI_Group to the bufptr ...
def printf77group(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  return str

# generates C-Code that prints an MPI_Errhandler to the bufptr ...
def printf77errhndl(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  return str

# generates C-Code that prints an MPI_File to the bufptr ...
def printf77file(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  return str

# generates C-Code that prints an MPI_Offset to the bufptr ...
def printf77offset(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  return str

# generates C-Code that prints an MPI_Request to the bufptr ...
def printf77request(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  return str

# generates C-Code that prints an MPI_Status to the bufptr ...
def printf77status(ptr):
  str = ""
  str = str + 'bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%i", *'+ptr+');'
  return str

def remove_const(params):
  for k in range(0,len(params)):
    p = re.compile("^[\s]*const ");
    params[k] = p.sub('', params[k])
  return params

# generates the function profiling code for C bindings...
def gencfunc(name, ret, params):
  str = ret + " " + stripspaces(name) + "("
  str = str + printstrings(params)
  str = str + ") { \n"
  str = str + "  " + ret + " ret;\n"

  params = remove_const(params)

  ##### special MPI_Type_free handling for boost::mpi -- see C comment!! #######
  if(stripspaces(name) == "MPI_Type_free" ):
    str = str + """\n  if(!mpi_initialized) {
    // this is weird because boost::mpi seems to call MPI_Type_free
    // after MPI finalize when it is profiled ????
    return MPI_SUCCESS;\n  }\n\n"""

  ##### special MPI_Finalize handling #######
  if(stripspaces(name) == "MPI_Finalize" ):
    str = str + "  mpi_finalize();\n"
  else:
    str = str + "  check();\n"

  str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "'+stripspaces(name)+'");\n'
  # can't call MPI_Wtime before MPI_Init ...
  if(stripspaces(name) == "MPI_Init" or stripspaces(name) == "MPI_Init_thread"):
    str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-");\n'
  else:
    str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);\n'
  
  #####################################################
  # call to PMPI backend ...
  #####################################################
  str = str + "\n"
  str = str + "  ret = P" + stripspaces(name) + "("
  # forward call arguments to pmpi routine ...
  for k in range(0,len(params)):
    # split arguments
    p = re.compile(' ');
    j = p.split(stripspaces(params[k]))
    
    if(len(j) > 1):
      # delete array arguments [...] ...
      p = re.compile("\[.*\]+");
      j[1] = p.sub('', j[1])

      str = str + stripspaces(stripasterisk(j[1]))
    else:  
      # don't add void parameter ...
      if(stripspaces(j[0]) != "void"):
        str = str + stripspaces(j[0])
    
    # add comma
    if(k != len(params)-1):
        str = str + ", "

  str = str + ");\n\n"

  ################ special MPI_Init handling
  if(stripspaces(name) == "MPI_Init" or stripspaces(name) == "MPI_Init_thread"):
    str = str + "\n  {\n    if(!mpi_initialized) mpi_initialize();\n print_banner(world_rank, \"C\", \""+stripspaces(name)+"\", world_size);\n  }\n"
  ################ end of special MPI_Init handling

  #####################################################
  # profiling - dependent on variable type ...
  #####################################################
  for k in range(0,len(params)):

    # split arguments
    p = re.compile(' ');
    j = p.split(stripspaces(params[k]))
    
    if(len(j) > 1): # this filters "void" and "..."
      # see if it's a pointer
      p = re.compile(".*\[.*\].*"); # search for array brackets "[]" ...
      if( p.match(stripspaces(j[1]))):
        p = re.compile("\[.*\]+"); # delete array brackets "[]"
        j[1] = p.sub('', j[1])
        str = str + '  ' + printpointer(stripasterisk(j[1])) + '\n'
        continue
      # see if it's a pointer
      p = re.compile('\*'); # search for "*"
      if( p.match(stripspaces(j[1]))):
        # the operations Alltoallv, Alltoallw, Allgatherv, Gatherv,
        # Scatterv and Reduce_scatter have {send,recv}counts argumens
        # that are arrays of the size of the comm and contain the count
        # arguments. We filter for the *names* here and call a special
        # routine to print those arguments
        if(j[1] == '*sendcounts' or j[1] == '*recvcounts' or 
           j[1] == '*displs' or j[1] == '*rdispls' or j[1] == '*sdispls'):
          str = str + '  ' + printcounts(stripasterisk(j[1])) + '\n'
        else:
          str = str + '  ' + printpointer(stripasterisk(j[1])) + '\n'
        continue
      # see if it's an integer
      if(j[0] == 'int' or j[0] == 'MPI_Aint' or j[0] == 'MPI_Fint'):
        str = str + '  ' + printint(j[1]) + '\n'
        continue
      # see if it's an MPI_Datatype
      if(j[0] == 'MPI_Datatype'):
        str = str + '  ' + printdatatype(j[1]) + '\n'
        continue
      # see if it's an MPI_Comm
      if(j[0] == 'MPI_Comm'):
        str = str + '  ' + printcomm(j[1]) + '\n'
        continue
      # see if it's an MPI_Op
      if(j[0] == 'MPI_Op'):
        str = str + '  ' + printop(j[1]) + '\n'
        continue
      # see if it's an MPI_Win
      if(j[0] == 'MPI_Win'):
        str = str + '  ' + printwin(j[1]) + '\n'
        continue
      # see if it's an MPI_Info
      if(j[0] == 'MPI_Info'):
        str = str + '  ' + printinfo(j[1]) + '\n'
        continue
      # see if it's an MPI_Group
      if(j[0] == 'MPI_Group'):
        str = str + '  ' + printgroup(j[1]) + '\n'
        continue
      # see if it's an MPI_Errhandler
      if(j[0] == 'MPI_Errhandler'):
        str = str + '  ' + printerrhndl(j[1]) + '\n'
        continue
      # see if it's an MPI_File
      if(j[0] == 'MPI_File'):
        str = str + '  ' + printfile(j[1]) + '\n'
        continue
      # see if it's an MPI_Offset
      if(j[0] == 'MPI_Offset'):
        str = str + '  ' + printoffset(j[1]) + '\n'
        continue
      # see if it's an MPI_Request
      if(j[0] == 'MPI_Request'):
        str = str + '  ' + printrequest(j[1]) + '\n'
        continue
      # see if it's an NBC_Handle 
      if(j[0] == 'NBC_Handle'):
        str = str + '  ' + printrequest(j[1]) + '\n'
        continue
      # see if it's an MPI_Status
      if(j[0] == 'MPI_Status'):
        str = str + '  ' + printstatus(j[1]) + '\n'
        continue



      print j[0] + ' not caught for ' + name
      sys.exit(1)
      
  str = str + "\n"
  # can't use MPI_Wtime after MPI_Finalize!
  if(stripspaces(name) == "MPI_Finalize" ):
    str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-\\n");\n'
  else:
    str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\\n", PMPI_Wtime()*1e6);\n'
  str = str + "\n"
  
  if(stripspaces(name) == "MPI_Finalize" ):
    str = str + "  fputs((char*)curbuf, fp);; fclose(fp);"
    
  str = str + "  return ret;"
  str = str + "\n}\n"
  return str

# generates the function profiling code for Fortran ...
def genfortfunc(name, ret, params):

  params = remove_const(params)

  # ompi does not have fortran-bidings for *f2c* *c2f* 
  p = re.compile('.*f2c.*');
  if( p.match(name)):
    return ""
  p = re.compile('.*c2f.*');
  if( p.match(name)):
    return ""
  
  str = ""
  ######################################################
  # generate function prototype for PMPI function ...
  ######################################################
  # several functions that are called internally are defned statically
  # at the beginning and not generated - all others are directly added
  # to the code 
  if (stripspaces(name).lower() != "mpi_type_extent" and 
      stripspaces(name).lower() != "mpi_type_size" and
      stripspaces(name).lower() != "mpi_comm_size" and
      stripspaces(name).lower() != "mpi_comm_rank"):
    pname = "P" + stripspaces(name);
    str = "void F77_FUNC("+stripspaces(pname.lower())+","+stripspaces(pname.upper()) + ")(" 
    
    # every argument is an integer pointer :-/
    for k in range(0,len(params)):
      # split arguments
      p = re.compile(' ');
      j = p.split(stripspaces(params[k]))
      if(len(j) > 1):
        if(stripspaces(j[0]) == "MPI_Aint"):
          str = str + "MPI_Aint *" + stripasterisk(stripspaces(j[1]))
        else:
          str = str + "int *" + stripasterisk(stripspaces(j[1]))
      else:
        # don't add void parameter ...
        if(stripspaces(j[0]) != "void"):
          str = str + stripspaces(j[0])
        else:
          # fools the next if into thinking that this is last ... and not adding a ", "
          k = len(params);
      
      # add comma
      if(k != len(params)):
          str = str + ", "
      
    
    str = str + "int *ierr); \n"
  
  ######################################################
  # generate function header for MPI function ...
  ######################################################
  str = str + "void F77_FUNC("+stripspaces(name.lower())+","+stripspaces(name.upper()) + ")(" 
  
  # every argument is an integer pointer :-/
  for k in range(0,len(params)):
    # split arguments
    p = re.compile(' ');
    j = p.split(stripspaces(params[k]))
    if(len(j) > 1):
      if(stripspaces(j[0]) == "MPI_Aint"):
        str = str + "MPI_Aint *" + stripasterisk(stripspaces(j[1]))
      else:
        str = str + "int *" + stripasterisk(stripspaces(j[1]))
    else:
      # don't add void parameter ...
      if(stripspaces(j[0]) != "void"):
        str = str + stripspaces(j[0])
      else:
        # fools the next if into thinking that this is last ... and not adding a ", "
        k = len(params);
    
    # add comma
    if(k != len(params)):
        str = str + ", "
    
  
  str = str + "int *ierr) { \n"
  
  ##### special MPI_Finalize handling #######
  if(stripspaces(name) == "MPI_Finalize" ):
    str = str + "  mpi_finalize();"
  else:
    str = str + "  check();\n"

  str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), "%s", "'+stripspaces(name)+'");\n'
  # can't use MPI_Wtime before MPI_Init
  if(stripspaces(name) == "MPI_Init" or stripspaces(name) == "MPI_Init_thread"):
    str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-");\n'
  else:
    str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f", PMPI_Wtime()*1e6);\n'

  #####################################################
  # call to PMPI backend ...
  #####################################################
  str = str + "\n"
  pname = "P" + stripspaces(name);
  str = str + "  F77_FUNC("+stripspaces(pname.lower())+","+stripspaces(pname.upper()) + ")("
  # forward call arguments to pmpi routine ...
  for k in range(0,len(params)):
    # split arguments
    p = re.compile(' ');
    j = p.split(stripspaces(params[k]))
    
    if(len(j) > 1):
      # delete array arguments [...] ...
      p = re.compile("\[.*\]+");
      j[1] = p.sub('', j[1])

      str = str + stripspaces(stripasterisk(j[1]))
    else:  
      # don't add void parameter ...
      if(stripspaces(j[0]) != "void"):
        str = str + stripspaces(j[0])
      else:
        # fools the next if into thinking that this is last ... and not adding a ", "
        k = len(params);
    
    # add comma
    if(k != len(params)):
        str = str + ", "

  str = str + "ierr);\n\n"

  ################ special MPI_Init handling
  if(stripspaces(name) == "MPI_Init" or stripspaces(name) == "MPI_Init_thread"):
    str = str + "\n  {\n    if(!mpi_initialized) mpi_initialize();\n print_banner(world_rank, \"F77\", \""+stripspaces(name)+"\", world_size);\n  }\n"
  ################ end of special MPI_Init handling

  #####################################################
  # profiling - dependent on variable type ...
  #####################################################
  for k in range(0,len(params)):
    # split arguments
    p = re.compile(' ');
    j = p.split(stripspaces(params[k]))
    
    if(len(j) > 1): # this filters "void" and "..." TODO not really, i.e. MPI_Pcontrol(level, ...)
      # see if it's a pointer
      p = re.compile(".*\[.*\].*"); # search for array brackets "[]" ...
      if( p.match(stripspaces(j[1]))):
        p = re.compile("\[.*\]+"); # delete array brackets "[]"
        j[1] = p.sub('', j[1])
        str = str + '  ' + printpointer(stripasterisk(j[1])) + '\n'
        continue

      # see if it's a pointer
      p = re.compile('\*');
      if( p.match(stripspaces(j[1]))):
        # the operations Alltoallv, Alltoallw, Allgatherv, Gatherv,
        # Scatterv and Reduce_scatter have {send,recv}counts argumens
        # that are arrays of the size of the comm and contain the count
        # arguments. We filter for the *names* here and call a special
        # routine to print those arguments
        if(j[1] == '*sendcounts' or j[1] == '*recvcounts' or 
           j[1] == '*displs' or j[1] == '*rdispls' or j[1] == '*sdispls'):
          str = str + '  ' + printf77counts(stripasterisk(j[1])) + '\n'
        else:
          str = str + '  ' + printpointer(stripasterisk(j[1])) + '\n'
        continue
      # see if it's an integer
      if(j[0] == 'int' or j[0] == 'MPI_Aint' or j[0] == 'MPI_Fint'):
        str = str + '  ' + printf77int(j[1]) + '\n'
        continue
      # see if it's an MPI_Datatype
      if(j[0] == 'MPI_Datatype'):
        str = str + '  ' + printf77datatype(j[1]) + '\n'
        continue
      # see if it's an MPI_Comm
      if(j[0] == 'MPI_Comm'):
        str = str + '  ' + printf77comm(j[1]) + '\n'
        continue
      # see if it's an MPI_Op
      if(j[0] == 'MPI_Op'):
        str = str + '  ' + printf77op(j[1]) + '\n'
        continue
      # see if it's an MPI_Win
      if(j[0] == 'MPI_Win'):
        str = str + '  ' + printf77win(j[1]) + '\n'
        continue
      # see if it's an MPI_Info
      if(j[0] == 'MPI_Info'):
        str = str + '  ' + printf77info(j[1]) + '\n'
        continue
      # see if it's an MPI_Group
      if(j[0] == 'MPI_Group'):
        str = str + '  ' + printf77group(j[1]) + '\n'
        continue
      # see if it's an MPI_Errhandler
      if(j[0] == 'MPI_Errhandler'):
        str = str + '  ' + printf77errhndl(j[1]) + '\n'
        continue
      # see if it's an MPI_File
      if(j[0] == 'MPI_File'):
        str = str + '  ' + printf77file(j[1]) + '\n'
        continue
      # see if it's an MPI_Offset
      if(j[0] == 'MPI_Offset'):
        str = str + '  ' + printf77offset(j[1]) + '\n'
        continue
      # see if it's an MPI_Request
      if(j[0] == 'MPI_Request'):
        str = str + '  ' + printf77request(j[1]) + '\n'
        continue
      # see if it's an NBC_Handle
      if(j[0] == 'NBC_Handle'):
        str = str + '  ' + printf77request(j[1]) + '\n'
        continue
      # see if it's an MPI_Status
      if(j[0] == 'MPI_Status'):
        str = str + '  ' + printf77status(j[1]) + '\n'
        continue



      print j[0] + ' not caught'
      sys.exit(1)
      
  str = str + "\n"
  # can't use MPI_Wtime after MPI_Finalize!
  if(stripspaces(name) == "MPI_Finalize" ):
    str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":-\\n");\n'
  else:
    str = str + '  bufptr += snprintf((char*)bufptr, BUFSIZE-((VOLATILE unsigned long)bufptr-(VOLATILE unsigned long)curbuf), ":%.0f\\n", PMPI_Wtime()*1e6);\n'
  str = str + "\n"
  
  # flush curbuf and close fp in Finalize
  if(stripspaces(name) == "MPI_Finalize" ):
    str = str + "  fputs((char*)curbuf, fp);; fclose(fp);"
    
  str = str + "\n}\n"
  return str


#####################################################
# reads a file into a string
#####################################################
def fileread(name):
  try:
    f=open(name, 'r')
  except IOError:
    print "Error: can\'t find file" + name + "\n"
  file = ""
  text = "1"
  while text:
    text = f.readline()
    file = file + text

  return file

#####################################################
# MAIN DRIVER ROUTINE 
#####################################################

f77 = 0
c = 0
if(len(sys.argv) > 1 and sys.argv[1] == "f77"):
  f77 = 1
elif (len(sys.argv) > 1 and sys.argv[1] == "c"):
  c = 1
else:  
  print "usage: " + sys.argv[0] + " [f77|c]" 
  sys.exit(1)
  

str = fileread("template.c")

# read full file into 'file'
file = fileread("mpi.h_mpich")

# split function-wise at ';'
functions = string.split(file, ";")

for i in functions:
  # erase newlines
  p = re.compile('\n');
  i = p.sub( '', i)
  # erase double whitespaces
  p = re.compile("[\s]+");
  i = p.sub( ' ', i)
  i = stripspaces(i)

  if len(i) > 0:
    str = str + "\n/* parsing >" + i + "< */\n"

    # see if it's an NBC function ...
    p = re.compile(".*NBC_.*");
    nbcfunc=0
    if(p.match(i)):
      nbcfunc=1

    # get function return type
    p = re.compile(" ");
    j = p.split(i)
    funcreturn = j[0]

    # put remaining string in i
    i=""
    for k in range(1,len(j)):
      i = i+" "+j[k]
 
    # get function name
    p = re.compile('\(');
    j = p.split(i)
    funcname = j[0]

    # put remaining string in i
    i=j[1]

    # erase closing ")"
    p = re.compile('\)');
    i = p.sub( '', i)

    p = re.compile(",");
    params = p.split(i)

    if(nbcfunc==1):
      str = str + "#ifdef HAVE_NBC\n"

    if(c == 1):
      str = str + gencfunc(funcname, funcreturn, params)
    if (f77 == 1):  
      str = str + genfortfunc(funcname, funcreturn, params)

    if(nbcfunc==1):
      str = str + "#endif\n"

str = str + """#ifdef __cplusplus
}
#endif
"""

print str
