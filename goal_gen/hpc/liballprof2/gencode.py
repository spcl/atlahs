#! /usr/bin/env python3

import re
import clang.cindex
import argparse
from collections import defaultdict
import yaml
import pathlib
import os

class AllprofCodegen:

    def __init__(self):
        self.semantics = {}

    def parse_semantics(self, semfile):
        with open(semfile, "r") as stream:
            try:
                self.semantics = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    def get_param_names(self, func):
        names = []
        params = self.semantics[func]['params']
        for p in params:
            names += [p['name']]
        return names
    
    def get_param_argtype(self, argname, funcname):
        for p in self.semantics[funcname]['params']:
            if p['name'] == argname:
                return p['type']
        return None
    
    def is_ptr_arg(self, argname, funcname):
        if self.is_ptr(self.get_param_argtype(argname, funcname)):
            return True
        else:
            return False
  
    def is_inttype(self, typestr):
        """ Check if typestr is a type which we can cast to a long long int without loosing precision """
        if typestr in ['int', 'MPI_Aint', 'MPI_Offset', 'MPI_Count']:
            return True
        return False
    
    def is_ptr(self, typestr):
        if typestr in [None, ""]:
            return True
        if ("*" in typestr) or ("[" in typestr):
            return True
        return False            
        

    def deref_args(self, expr, funcname):
        """
        We use C snippets (which might reference arguments) to describe how many elements arrays contain. 
        For example a function f might have two arguments, arr and count. And we know (in C syntax) that arr contains count elements.
        This function translates expr to Fortran-compatible arguments, where all simple args are pointers, and thus need to be
        dereferenced.
        """
        params = self.get_param_names(funcname)
        for p in params:
            if self.is_ptr_arg(p, funcname):
                continue
            expr_new = ""
            m = True
            while m:
                m = re.match(f"(.*?\W|^){p}($|\W.*)", expr)
                if m is None:
                    break
                expr_new += m.group(1)+f"(*{p})"
                expr = m.group(2)
            expr = expr_new + expr
        return expr


    def fortranize_prolog(self, expr, funcname):
        expr = self.deref_args(expr, funcname)
        expr = re.sub(r"PMPI_Cartdim_get\((.+?),(.+?)\)", r"PMPI_Cartdim_get( MPI_Comm_f2c(\1), \2)", expr)
        expr = re.sub(r"PMPI_Comm_size\((.+?),(.+?)\)", r"PMPI_Comm_size( MPI_Comm_f2c(\1), \2)", expr)
        expr = re.sub(r"PMPI_Comm_rank\((.+?),(.+?)\)", r"PMPI_Comm_rank( MPI_Comm_f2c(\1), \2)", expr)
        expr = re.sub(r"PMPI_Dist_graph_neighbors_count\((.+?),(.+?),(.+?),(.+?)\)", r"PMPI_Dist_graph_neighbors_count( MPI_Comm_f2c(\1), \2, \3, \4)", expr)
        return expr

    def write_prolog(self, mode='c'):
        if mode == 'fortran':
            self.outfile.write("#include \"fc_mangle.h\"\n")
        source_path = pathlib.Path(__file__).resolve()
        source_dir = source_path.parent
        with open(os.path.join(source_dir, "tracer_main.c")) as f:
            c_code = f.readlines()
            for l in c_code:
                if re.match("int\s+main\s*\(.*", l):
                    return
                self.outfile.write(l)

    def write_tracer_prolog(self, func, mode):
        """ Write a tracers prolog code, which writes starttime and function name to trace. Return value is expected in pmpi_retval!"""
        prolog_code = ""
        if mode == 'c':
            prolog_code += f"  {self.semantics[func]['return_type']} pmpi_retval;\n"
        prolog_code += f"  lap_check();\n"
        prolog_code += f"  WRITE_TRACE(\"%s\", \"{func}:\");\n"
        prolog_code += f"  WRITE_TRACE(\"%0.2f:\", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);\n"
        self.outfile.write(prolog_code)

    def write_tracer_epilog(self, func, mode):
        """ Write a tracers epilog code, which writes endtime and backtrace """
        code = ""
        code += f"  WRITE_TRACE(\"%0.2f\", lap_mpi_initialized ? PMPI_Wtime()*1e6 : 0.0);"
        code += f"  if (lap_backtrace_enabled) {{\n"
        code += f"    lap_get_full_backtrace(lap_backtrace_buf, LAP2_BACKTRACE_BUF_SIZE);\n"
        code += f"    WRITE_TRACE(\"  # backtrace [%s]\", lap_backtrace_buf);\n"
        code += f"  }}\n"
        code += f"  WRITE_TRACE(\"%s\", \"\\n\");\n"
        self.outfile.write(code)

    def write_pmpi_call(self, func, mode):
        """ Write the PMPI call. """
        sem = self.semantics[func]
        args = []
        for arg in self.semantics[func]['params']:
            args.append(arg['name'])
        if mode == 'fortran':
            args.append("ierr")
        argstr = ", ".join(args)
        pfunc = f"P{func}"
        #self.outfile.write(f"fprintf(stderr, \"before {pfunc}\\n\");\n")
        if mode == 'c':
            self.outfile.write(f"  pmpi_retval = {pfunc}({argstr});\n")
        elif mode == 'fortran':
            # in fortran mpi calls return void
            self.outfile.write(f" FortranCInterface_GLOBAL({pfunc.lower()},{pfunc.upper()})({argstr});\n")
        #self.outfile.write(f"fprintf(stderr, \"after {pfunc}\\n\");\n")
        if func in ["MPI_Abort", "MPI_Finalize"]:
            self.outfile.write("  lap_mpi_initialized = 0;\n")


    def split_type(self, typestr):
        """ Type descriptions like int[] cannot simply be prepended to an argument name foo, it must be int foo[]. This function seperates the base type and the [] part. """
        m = re.match("(.+?)(\[.*\])", typestr)
        if m:
            return (m.group(1), m.group(2))
        else:
            return (typestr, "")
        
    def get_basetype(self, typestr):
        """ Remove any attribute such as const, *, [] from typestr """
        m = re.match("^(.+?)(\[.*)", typestr)
        if m:
            typestr = m.group(1)
        elems = typestr.split(" ")
        if elems[-1] == "*":
            elems = elems[:-1]
        return elems[-1]
        
 
    
    def trace_inttype(self, name, typestr, basetype, is_ptr, sep):
        deref = ""
        if is_ptr:
            deref = "*"
        return f"  WRITE_TRACE(\"%lli{sep}\", (long long int) {deref}({name}));\n"

    def is_mpiobj(self, typestr):
        if typestr in ['MPI_Comm', 'MPI_Group', 'MPI_Win', 
                       'MPI_Info', 'MPI_Message', 'MPI_Op', 
                       'MPI_Request', 'MPI_Status', 'MPI_Datatype', 
                       'MPI_File', 'MPI_Errhandler']:
            return True
        return False

    def trace_mpiobj(self, name, typestr, basetype, is_ptr, sep):
        deref = ""
        ref = "&"
        if is_ptr:
            deref = "*"
            ref = ""
        if basetype == "MPI_Datatype": #the mpi forum really values consistency - except when they don't :)
            basetype = "MPI_Type"
        if basetype == "MPI_Status":
            if is_ptr: 
                return f"if (({name} == MPI_STATUSES_IGNORE) || ({name} == MPI_STATUSES_IGNORE)) {{WRITE_TRACE(\"%lli\", (long long int) {name});}} else {{WRITE_TRACE(\"%p:[%i,%i,%i]{sep}\", {name}, ({ref}{name})->MPI_SOURCE, ({ref}{name})->MPI_TAG, ({ref}{name})->MPI_ERROR);}}\n"
            else:
                return f"WRITE_TRACE(\"%p:[%i,%i,%i]{sep}\", {name}, ({ref}{name})->MPI_SOURCE, ({ref}{name})->MPI_TAG, ({ref}{name})->MPI_ERROR);\n"
            
        return f"  WRITE_TRACE(\"%lli{sep}\", (long long int) P{basetype}_c2f({deref}{name}));\n"


    def tracer_for_simple_arg(self, name, typestr, func, sep=":"):
        # strip const, shouldn't make a difference how we trace
        if typestr.startswith("const "):
            typestr = typestr[6:]
        # handle this special case first
        if typestr == "int[3]":
            return f"WRITE_TRACE(\"[%i,%i,%i]{sep}\", {name}[0], {name}[1], {name}[2]);\n"
        basetype = self.get_basetype(typestr)
        is_ptr = self.is_ptr(typestr)
        if self.is_inttype(basetype):
            return self.trace_inttype(name, typestr, basetype, is_ptr, sep)
        elif self.is_mpiobj(basetype):
            return self.trace_mpiobj(name, typestr, basetype, is_ptr, sep)
        elif typestr == "char":
            return f"WRITE_TRACE(\"%c{sep}\", {name});\n"
        elif typestr == "void *":
            return f"WRITE_TRACE(\"%p{sep}\", {name});\n"
        elif typestr.startswith("char *"):
            return f"WRITE_TRACE(\"%p{sep}\", {name});\n"
        elif typestr.endswith("_function *"):
            return f"WRITE_TRACE(\"%p{sep}\", {name});\n"
        else:
            print(f"{typestr} tracer not implemmented (appears in {func}, basetype is {basetype})")
        return ""

    def tracer_for_simple_arg_fortran(self, name, typestr, func, sep=":"):
        # strip const, shouldn't make a difference how we trace
        if typestr.startswith("const "):
            typestr = typestr[6:]
        # handle this special case first
        if typestr == "int[3]":
            return f"  WRITE_TRACE(\"[%i,%i,%i]{sep}\", *(&({name})+0), *(&({name})+1), *(&({name})+2));\n"
        basetype = self.get_basetype(typestr)
        if not self.is_ptr_arg(name, func):
            return f"  WRITE_TRACE(\"%lli{sep}\", (long long int) *{name});\n"
        else:
            if "[" in name:
                return f"  WRITE_TRACE(\"%lli{sep}\", (long long int) {name});\n"
            else:
                return f"  WRITE_TRACE(\"%lli{sep}\", (long long int) *{name});\n"


    def write_argument_tracers(self, func, mode):
        # collect all needed prologs (code that we need to decide how many elemnts are in an array, like getting comm size)
        prologs = []
        for sem_param in self.semantics[func]['params']:
            if ('prolog_elem_count' in sem_param) and (sem_param['prolog_elem_count'] is not None):
                if mode == 'c':
                    prologs.append(sem_param['prolog_elem_count'])
                if mode == 'fortran':
                    prolog = sem_param['prolog_elem_count']
                    prolog = self.fortranize_prolog(prolog, func)
                    prologs.append(prolog)
        prologs = list(set(prologs))
        if len(prologs) > 0:
            self.outfile.write("\n".join(prologs)+"\n//end of prologs\n")

        for sem_param in self.semantics[func]['params']:
            #print(sem_param)
            if ('elem_count' in sem_param):
                # this argument contains multiple elements
                # let the user configure at runtime if elements need to be traced (replace 0 with env-bases expr)
                elem_count_expr = "0"
                if ('elem_count' in sem_param) and (sem_param['elem_count'] is not None):
                    elem_count_expr = sem_param['elem_count']
                    if (mode == 'fortran'):
                        elem_count_expr = self.deref_args(elem_count_expr, func)
                self.outfile.write(f"  WRITE_TRACE(\"%p,%i[\", (void*) {sem_param['name']}, (int) {elem_count_expr});\n")
                self.outfile.write(f"  if (lap_elem_tracing_enabled == 0) {{  }} else {{ \n")
                self.outfile.write(f"    for (int trace_elem_idx=0; trace_elem_idx<{elem_count_expr}; trace_elem_idx++) "+"{\n")
                
                # emit the tracer for the simplified arg, use ; to seperate elems
                name = sem_param['name'] + "[trace_elem_idx]"
                basetype, brackets = self.split_type(sem_param['type'])
                typestr = basetype + brackets[2:]
                if mode == 'c':
                    code = self.tracer_for_simple_arg(name, typestr, func, sep=";")
                else:
                    code = self.tracer_for_simple_arg_fortran(name, typestr, func, sep=";")
                self.outfile.write("    "+code)
                self.outfile.write("  }\n")  
                self.outfile.write("  WRITE_TRACE(\"]%s\", \":\");\n")
                self.outfile.write("}\n")               
            else:
                if mode == 'c':
                    code = self.tracer_for_simple_arg(sem_param['name'], sem_param['type'], func, sep=":")
                else:
                    code = self.tracer_for_simple_arg_fortran(sem_param['name'], sem_param['type'], func, sep=":")
                self.outfile.write("  " + code)


    def produce_fortran_pmpi_prototypes(self):
        for func in self.semantics:
            param_signatures = []
            for param in self.semantics[func]['params']:
                argtype = "int*"
                if "char" in self.get_param_argtype(param['name'], func):
                    argtype = "char*"
                param_signatures.append(f"{argtype} {param['name']}")
            param_signatures.append("int* ierr")
            params = ", ".join(param_signatures)
            pfunc = f"P{func}"
            self.outfile.write(f"void FortranCInterface_GLOBAL({pfunc.lower()},{pfunc.upper()}) ({params})"+";\n")
        self.outfile.write("\n\n")


    def produce_pmpi_only_if_tracing_disabled(self, func, mode):
        self.outfile.write(f"  if (lap_tracing_enabled == 0) {{ \n")
        self.outfile.write(f"    {self.semantics[func]['return_type']} pmpi_retval;")
        self.write_pmpi_call(func, mode)
        if mode == 'c':
            self.outfile.write(f"    return pmpi_retval;\n")
        if mode == 'fortran':
            self.outfile.write(f"    return;\n")
        self.outfile.write(f"  }}\n")


    def produce_pcontrol(self, mode):
        deref = ""
        if mode == 'fortran':
            deref = "*"
        self.outfile.write(f"  if ({deref}level == 0) {{ lap_tracing_enabled = 0; lap_elem_tracing_enabled = 0; lap_backtrace_enabled = 0; }}\n")
        self.outfile.write(f"  if ({deref}level == 1) {{ lap_tracing_enabled = 1; lap_elem_tracing_enabled = 0; lap_backtrace_enabled = 0; }}\n")
        self.outfile.write(f"  if ({deref}level == 2) {{ lap_tracing_enabled = 1; lap_elem_tracing_enabled = 1; lap_backtrace_enabled = 0; }}\n")
        self.outfile.write(f"  if ({deref}level == 3) {{ lap_tracing_enabled = 1; lap_elem_tracing_enabled = 1; lap_backtrace_enabled = 1; }}\n")
        self.outfile.write(f"  WRITE_TRACE(\"# pcontrol with value / epoch %i)\\n\", {deref}level);\n")
        if mode == 'c':
            self.outfile.write(f"  return MPI_SUCCESS;\n")
    

    def produce_tracers(self, mode='c'):
        for func in self.semantics:

            # do not trace f2c funcs in fortran mode - they don't exist?
            if mode == 'fortran' and (('f2c' in func) or ('c2f' in func)):
                return

            delay_pmpi = False # usually we write the trace after the pmpi call, however, if the function frees some of its arguments we want to do it before
            if func.endswith("_free") or func.endswith("Free_mem") or ("_delete_" in func) or (func == "MPI_Finalize"):
                delay_pmpi = True

            param_signatures = []
            for param in self.semantics[func]['params']:
                if mode == 'c':
                    type_prefix, type_suffix = self.split_type(param['type'])
                    param_signatures.append(f"{type_prefix} {param['name']}{type_suffix}")
                elif mode == 'fortran':
                    argtype = "int*"
                    if "char" in self.get_param_argtype(param['name'], func):
                        argtype = "char*"
                    param_signatures.append(f"{argtype} {param['name']}")
            if mode == 'fortran':
                param_signatures.append("int* ierr")
            params = ", ".join(param_signatures)

            if mode == 'c':
                self.outfile.write(f"{self.semantics[func]['return_type']} {func} ({params})"+" {\n")
            elif mode == 'fortran':
                self.outfile.write(f"void FortranCInterface_GLOBAL({func.lower()},{func.upper()}) ({params})"+" {\n")
            if func == "MPI_Pcontrol":
                self.produce_pcontrol(mode)
                self.outfile.write("}\n\n")
                continue
            if func != "MPI_Finalize":
                self.produce_pmpi_only_if_tracing_disabled(func, mode)
            self.write_tracer_prolog(func, mode)
            if not delay_pmpi:
                self.write_pmpi_call(func, mode)
            self.write_argument_tracers(func, mode)
            if delay_pmpi and (func != "MPI_Finalize"):
                self.write_pmpi_call(func, mode)
            if func == "MPI_Finalize":
                self.write_tracer_epilog(func, mode)
                self.outfile.write("lap_collect_traces();\n")
                self.write_pmpi_call(func, mode)
            else:
                self.write_tracer_epilog(func, mode)
            if mode == 'c':
                self.outfile.write("  return pmpi_retval;\n")
            self.outfile.write("}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='liballprof_gencode',
                    description='Generates wrappers for the MPI functions present in the supplied MPI header file. The wrappers output in liballprof2 trace format.',
                    epilog='')
    parser.add_argument('-s', '--semantics-file',       default='mpi_sem.yml',              help="Name of the file that specifies the tracer semantics (default: mpi-sem.yml)")
    parser.add_argument('-c', '--c-output-file',        default="mpi_c_wrapper.c",          help="Name of the generated C wrapper file (default: mpi_c_wrapper.c)")
    parser.add_argument('-f', '--fortran-output-file',  default="mpi_f_wrapper.c",          help="Name of the generated FORTRAN wrapper file (default: mpi_f_wrapper.c)")
    args = parser.parse_args()

    codegen = AllprofCodegen()
    codegen.parse_semantics(args.semantics_file)
    codegen.outfile = open(args.c_output_file, "w")
    codegen.write_prolog(mode='c')
    codegen.produce_tracers(mode='c')
    codegen.outfile.close()

    # modify the semantics for fortran
    codegen.semantics["MPI_Init"]['params'] = []
    codegen.semantics["MPI_Init_thread"]['params'] = []
    codegen.semantics["MPI_Pcontrol"]['params'].pop()

    codegen.outfile = open(args.fortran_output_file, "w")
    codegen.write_prolog(mode='fortran')
    codegen.produce_fortran_pmpi_prototypes()
    codegen.produce_tracers(mode='fortran')
    codegen.outfile.close()

