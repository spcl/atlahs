program helloworld
 implicit none
 include 'mpif.h'
 
 integer :: ierr, me, nproc
 double precision :: val

 call MPI_INIT(ierr)
 call MPI_COMM_RANK(MPI_COMM_WORLD,me,ierr)
 call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)

 call RANDOM_NUMBER(val)
 
 write(*,*) 'before', me, val
 call MPI_ALLREDUCE(MPI_IN_PLACE, val, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
 &                  MPI_COMM_WORLD, ierr)
 write(*,*) 'after', me, val
     
 call MPI_FINALIZE(ierr);

end program
