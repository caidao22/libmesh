#!/usr/bin/sh

#set -x
source $LIBMESH_DIR/examples/run_common.sh

example_name=navier_stokes

#run_example "$example_name" 


### --------------------------------- gmres/fgmres --------------------------------------

# works! commandline used for test. convergence faster with larger restart number!
#mpirun -np 4 ./ns-opt -ksp_type gmres -ksp_gmres_restart 80 -pc_type asm -sub_pc_type ilu -ksp_view -ksp_monitor -ksp_converged_reason -log_summary


# works! fgmres converges fast, but the residual norm is comparatively large?
# in addition ksp pc_type reduce the iteration steps, but increase the total time.
#mpirun -np 3 ./ns-opt -ksp_type fgmres -pc_type ksp -sub_pc_type ilu -ksp_view -ksp_monitor -ksp_converged_reason -log_summary

#mpirun -np 4 ./ns-opt -ksp_type gmres -ksp_gmres_restart 80 -pc_type asm -sub_pc_type ilu -log_summary

# simple solve
$PETSC_DIR/$PETSC_ARCH/bin/mpirun -np 2 ./ns-$METHOD -ts_type beuler -ksp_type preonly -pc_type lu -dm_view
# -ts_monitor
# -ksp_monitor -ksp_view

