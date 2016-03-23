// The libMesh Finite Element Library.
// Copyright (C) 2002-2014 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

#include "libmesh/libmesh_common.h"
#include "libmesh/petsc_vector.h"
#include "libmesh/petsc_matrix.h"  // for PetscMatrix
#include "libmesh/dof_map.h"
#include "libmesh/petscdmlibmesh.h"
#include "libmesh/solver_configuration.h"

#ifdef LIBMESH_HAVE_PETSC

// C++ includes

// Local Includes
#include "petsc_ts_system.h"
#include "petsc_ts_solver.h"

namespace libMesh
{

// Functions with C linkage to pass to PETSc.  PETSc will call these
// methods as needed.
//
// Since they must have C linkage they have no knowledge of a namespace.
// Give them an obscure name to avoid namespace pollution.
extern "C"
{

  // this function is called by PETSc at the end of each step
  PetscErrorCode
  libmesh_petsc_ts_monitor (TS ts, PetscInt step, PetscReal time, Vec x, void *ctx)
  {
    START_LOG("Monitor()", "PetscTSSolver");

    libmesh_assert(x); // make sure x is non-NULL

    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem *tssys = (PetscTSSystem *)ctx;

    // Wrap PETSc Vec as a libMesh::PetscVector<Number> .
    PetscVector<Number> X (x,tssys->comm());

    // Call the monitor on the PetscVector X, along with the other arguments.
    tssys->monitor(step,time,X);

    STOP_LOG("Monitor()", "PetscTSSolver");
    return 0;
  }

  /*
  // this function is called by TS to evaluate the rhs function at X
  PetscErrorCode
  libmesh_petsc_ts_rhsfunction (TS ts, PetscReal time, Vec x, Vec r, void *ctx)
  {
    START_LOG("RHSFunction()", "PetscTSSolver");

    libmesh_assert(x);
    libmesh_assert(r);
    libmesh_assert(ctx);

    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem *tssys = (PetscTSSystem *)ctx;
    PetscVector<Number> X (x,tssys->comm());
    PetscVector<Number> R (r,tssys->comm());
    tssys->RHSFunction(time,X,R);
    R.close();

    STOP_LOG("RHSFunction()", "PetscTSSolver");
    return 0;
  }

  // this function is called by PETSc to evaluate the RHS Jacobian at X ant time t
  PetscErrorCode
  libmesh_petsc_ts_rhsjacobian(TS ts, PetscReal time, Vec x, Mat jac, Mat jacpre, void *ctx)
  {
    START_LOG("RHSJacobian()", "PetscTSSolver");
    PetscErrorCode ierr=0;

    libmesh_assert(x);
    libmesh_assert(jac);
    libmesh_assert(ctx);

    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem *tssys = (PetscTSSystem *)ctx;
    PetscVector<Number> X     (x,     tssys->comm());
    PetscMatrix<Number> Jac   (jac,   tssys->comm());
    PetscMatrix<Number> Jacpre(jacpre,tssys->comm());

    // What do we do if jacpre is NULL?  Should we switch to passing
    // PetscVector* and PetscMatrix* to TSSystem::RHSJacobian() etc.?
    tssys->RHSJacobian(time,X,Jac,Jacpre);

    STOP_LOG("RHSJacobian()", "PetscTSSolver");
    return ierr;
  }
  */

  PetscErrorCode
  libmesh_petsc_ts_ifunction (TS ts, PetscReal time, Vec x, Vec xdot, Vec f, void *ctx)
  {
    START_LOG("IFunction()", "PetscTSSolver");
    PetscErrorCode ierr = 0;

    libmesh_assert(x);
    libmesh_assert(xdot);
    libmesh_assert(f);
    libmesh_assert(ctx);

    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem & tssys = *(static_cast<PetscTSSystem *>(ctx));

    PetscVector<Number> X     (x,     tssys.comm());
    PetscVector<Number> Xdot  (xdot,  tssys.comm());
    PetscVector<Number> F     (f,     tssys.comm());

    // Use the system's update() to get a good local version of the
    // parallel solution.  This operation does not modify the incoming
    // "x" vector, it only localizes information from "x" into
    // sys.current_local_solution.
    PetscVector<Number> & X_sys = *cast_ptr<PetscVector<Number> *>(tssys.solution.get());
    X.swap(X_sys);
    tssys.update();
    X.swap(X_sys);
    // Enforce constraints (if any) exactly on the
    // current_local_solution.  This is the solution vector that is
    // actually used in the computation of the residual below, and is
    // not locked by debug-enabled PETSc the way that "x" is.
    //tssys.get_dof_map().enforce_constraints_exactly(tssys, tssys.current_local_solution.get());
    //if (solver->_zero_out_residual)
    //  F.zero();

    // Localize the potentially parallel vector
    UniquePtr<NumericVector<Number> > local_xdot = NumericVector<Number>::build(tssys.comm());
    local_xdot->init(Xdot.size(),false);
    Xdot.localize (*local_xdot, tssys.get_dof_map().get_send_list());

    // evaluate the ifunction
    tssys.IFunction(time,*tssys.current_local_solution.get(),*local_xdot,F);

    //tssys.IFunction(time,X,Xdot,F);

    F.close();
    STOP_LOG("IFunction()", "PetscTSSolver");

    // ---------------------- view the vector f ---------------------------
    //PetscViewer vec_viewer;
    //ierr = PetscPrintf(tssys.comm().get() ,"View Vec info: \n");
    //ierr = PetscViewerCreate(tssys.comm().get(), &vec_viewer);
    //ierr = PetscViewerSetType(vec_viewer, PETSCVIEWERASCII);
    //ierr = VecView(f, vec_viewer);
    //ierr = PetscViewerDestroy(&vec_viewer);
    // --------------------------------------------------------------------
    return ierr;
  }

  //---------------------------------------------------------------
  PetscErrorCode
  libmesh_petsc_ts_ijacobian (TS ts, PetscReal time, Vec x, Vec xdot, PetscReal shift,Mat ijac, Mat ijacpre, void *ctx)
  {
    START_LOG("IJacobian()", "PetscTSSolver");
    PetscErrorCode ierr = 0;

    libmesh_assert(x);
    libmesh_assert(xdot);
    libmesh_assert(ijac);
    libmesh_assert(ctx);

    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem & tssys = *(static_cast<PetscTSSystem *>(ctx));

    PetscVector<Number> X     (x,       tssys.comm());
    PetscVector<Number> Xdot  (xdot,    tssys.comm());
    PetscMatrix<Number> IJ    (ijac,    tssys.comm());
    PetscMatrix<Number> IJpre (ijacpre, tssys.comm());

    PetscVector<Number> & X_sys = *cast_ptr<PetscVector<Number> *>(tssys.solution.get());
    // Set the dof maps
    IJpre.attach_dof_map(tssys.get_dof_map());
    IJ.attach_dof_map(tssys.get_dof_map());
    // Use the systems update() to get a good local version of the parallel solution
    X.swap(X_sys);
    tssys.update();
    X.swap(X_sys);
    // Enforce constraints (if any) exactly on the
    // current_local_solution.  This is the solution vector that is
    // actually used in the computation of the residual below, and is
    // not locked by debug-enabled PETSc the way that "x" is.
    //tssys.get_dof_map().enforce_constraints_exactly(tssys, tssys.current_local_solution.get());
    //if (solver->_zero_out_jacobian)
    //  IJpre.zero();
    // evaluate the matrices
    tssys.IJacobian(time,*tssys.current_local_solution.get(),Xdot,shift,IJ,IJpre);
   
    //tssys.IJacobian(time,X,Xdot,shift,IJ,IJpre);
    IJ.close();
    IJpre.close(); 
    STOP_LOG("IJacobian()", "PetscTSSolver");
    
    // ---------------------- view the matrix ijac ---------------------------
    //PetscViewer mat_viewer;
    //ierr = PetscPrintf(tssys.comm().get(),"View Mat info: \n"); CHKERRABORT(tssys.comm().get(), ierr);
    //ierr = PetscViewerCreate(tssys.comm().get(), &mat_viewer);  CHKERRABORT(tssys.comm().get(), ierr);
    //ierr = PetscViewerSetType(mat_viewer, PETSCVIEWERASCII);   CHKERRABORT(tssys.comm().get(), ierr);
    //ierr = MatView(ijac, mat_viewer);                          CHKERRABORT(tssys.comm().get(), ierr);
    //ierr = PetscViewerDestroy(&mat_viewer);                    CHKERRABORT(tssys.comm().get(), ierr);
    // -----------------------------------------------------------------------
    
    return ierr;
  }

} // end extern "C"


// PetscTSSolver methods
template <typename T>
PetscTSSolver<T>::PetscTSSolver (sys_type& tssys, const char* name) :
  ParallelObject(tssys),
  _ts(NULL),
  _system(tssys),
  _name(name),
  _initialized(false),
  _reason(TS_CONVERGED_ITERATING  /*==0*/ ), // Arbitrary initial value...
  _initial_time(0.),
  _max_time(0.),
  _dt(0.),
  _max_steps(0),
  _n_linear_iterations(0),
  _current_nonlinear_iteration_number(0)
{
  // do nothing
}


template <typename T>
PetscTSSolver<T>::~PetscTSSolver ()
{
  this->clear ();
}


// PetscTSSolver members
#if defined(LIBMESH_HAVE_PETSC)
template <typename T>
UniquePtr<PetscTSSolver<T> > PetscTSSolver<T>::build(sys_type& s)
{
  // Build the appropriate solver
  return UniquePtr<PetscTSSolver<T> >(new PetscTSSolver<T>(s));
}
  
#else // LIBMESH_HAVE_PETSC
template <typename T>
UniquePtr<PetscTSSolver<T> > PetscTSSolver<T>::build(sys_type& s)
{
  libmesh_not_implemented_msg("ERROR: libMesh was compiled without PETSc TS solver support");
}
#endif


template <typename T>
void PetscTSSolver<T>::clear ()
{
  if (this->_initialized)
  {
    this->_initialized = false;

    PetscErrorCode ierr = 0;
    ierr = TSDestroy(&_ts);   LIBMESH_CHKERR(ierr);
    delete _name;
//    delete _R;
//    delete _J;
//    delete _Jpre;

    // Reset the nonlinear iteration counter.  This information is only relevant
    // *during* the solve().  After the solve is completed it should return to
    // the default value of 0.
    _initial_time                       = 0.;
    _max_time                           = 0.;
    _dt                                 = 0.;
    _max_steps                          = 0;
    _n_linear_iterations                = 0;
    _current_nonlinear_iteration_number = 0;
    _n_cost_functions                   = 0;
    _do_adjoint                         = false;
  }
}

// Initialize the data structures for TS solver.
// It is automatically called in PetscTSSystem::solve() before TSSolve is called.
template <typename T>
void PetscTSSolver<T>::init ()
{
  START_LOG("init()", "PetscTSSolver");
  
  // Initialize the data structures if not done so already.
  if (!this->_initialized)
  {
    this->_initialized  = true;
    PetscErrorCode ierr = 0;

    // Create TS
    ierr = TSCreate(this->comm().get(), &_ts); LIBMESH_CHKERR(ierr);
    ierr = TSSetProblemType(_ts, TS_NONLINEAR); LIBMESH_CHKERR(ierr);

    // Attaching a DM to TS.
    DM dm;
    ierr = DMCreate(this->comm().get(), &dm); LIBMESH_CHKERR(ierr);
    ierr = DMSetType(dm,DMLIBMESH); LIBMESH_CHKERR(ierr);
    ierr = DMlibMeshSetSystem(dm,this->system()); LIBMESH_CHKERR(ierr);

    if (_name)
    {
      std::string prefix = std::string(_name)+std::string("_");
      ierr = TSSetOptionsPrefix(_ts, prefix.c_str()); LIBMESH_CHKERR(ierr);
      ierr = DMSetOptionsPrefix(dm, _name); LIBMESH_CHKERR(ierr);
    }
    ierr = DMSetFromOptions(dm); LIBMESH_CHKERR(ierr);
    ierr = DMSetUp(dm); LIBMESH_CHKERR(ierr);
    ierr = TSSetDM(this->_ts, dm); LIBMESH_CHKERR(ierr);
    // TS now owns the reference to dm
    ierr = DMDestroy(&dm); LIBMESH_CHKERR(ierr);
    ierr = TSMonitorSet(_ts, libmesh_petsc_ts_monitor, &this->system(), NULL); LIBMESH_CHKERR(ierr);

    // Build the vector and matrices
    // TODO: how can we ensure it's a PetscVector*? Is cast_ptr enough?
//    NumericVector<Number> *R    = NumericVector<Number>::build(this->comm()).release();
//    _R = cast_ptr<PetscVector<Number>*>(R);
//    SparseMatrix<Number>  *J    = SparseMatrix<Number>::build(this->comm()).release();
//    _J = cast_ptr<PetscMatrix<Number>*>(J);
//    SparseMatrix<Number>  *Jpre = SparseMatrix<Number>::build(this->comm()).release();
//    _Jpre = cast_ptr<PetscMatrix<Number>*>(Jpre);

    // Use the solution of the system as the working vector. (This is not safe since the system may be modified when PETSc has the control)
    // PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(this->system().solution.get());
    //ierr = TSSetIFunction(_ts,X_sys.vec(), libmesh_petsc_ts_ifunction, &this->system()); LIBMESH_CHKERR(ierr);
    //ierr = TSSetSolution(_ts,X_sys.vec());

    // PETSc creates the working vector.
    // Thus the working vector can be cast into PETSc vector safely.
    ierr = TSSetIFunction(_ts,NULL,libmesh_petsc_ts_ifunction,&this->system());

    // Set the IJacobian
    PetscMatrix<Number>& Jac_sys = *cast_ptr<PetscMatrix<Number>*>(this->system().matrix);

    if (this->system().request_matrix("Preconditioner"))
    {
      this->system().request_matrix("Preconditioner")->close();
      PetscMatrix<Number>& PC_sys = *cast_ptr<PetscMatrix<Number>*>( this->system().request_matrix("Preconditioner") );
      ierr = TSSetIJacobian(_ts,Jac_sys.mat(),PC_sys.mat(),libmesh_petsc_ts_ijacobian,&this->system()); LIBMESH_CHKERR(ierr);
    }
    else
    {
      ierr = TSSetIJacobian(_ts,Jac_sys.mat(),Jac_sys.mat(),libmesh_petsc_ts_ijacobian,&this->system()); LIBMESH_CHKERR(ierr);
    }

    ierr = TSSetDuration(_ts, _max_steps, _max_time); LIBMESH_CHKERR(ierr);
    ierr = TSSetExactFinalTime(_ts,TS_EXACTFINALTIME_STEPOVER); LIBMESH_CHKERR(ierr);
    ierr = TSSetInitialTimeStep(_ts, _initial_time, _dt); LIBMESH_CHKERR(ierr);

    if (_do_adjoint)
    {
      // Tell TS to save trajectory in the forward run.
      ierr = TSSetSaveTrajectory(_ts);LIBMESH_CHKERR(ierr);
    }
    ierr = TSSetFromOptions(_ts); LIBMESH_CHKERR(ierr);

  } // end if

  STOP_LOG("init()", "PetscTSSolver");
}

//---------------------------------------------------------------------
template <typename T>
void PetscTSSolver<T>::solve ()
{
  START_LOG("solve()", "PetscTSSolver");
  PetscErrorCode ierr=0;

  // Set the solution
  PetscVector<Number>* PETScX  = cast_ptr<PetscVector<Number>*>(this->system().solution.get());
  ierr = TSSolve (_ts,PETScX->vec()); LIBMESH_CHKERR(ierr);

  // Get and store the reason for convergence
  PetscReal         ftime;
  PetscInt          nsteps;
  ierr = TSGetSolveTime(_ts,&ftime);      LIBMESH_CHKERR(ierr);
  ierr = TSGetTimeStepNumber(_ts,&nsteps);LIBMESH_CHKERR(ierr);
  TSGetConvergedReason(_ts, &_reason);    LIBMESH_CHKERR(ierr);

  PetscPrintf(this->comm().get(),"************* %s at time %g after %D steps\n",
              TSConvergedReasons[_reason], (double)ftime, nsteps);

  STOP_LOG("solve()", "PetscTSSolver");

  //this->system().update();
}


//// ... PetscTSSolver::step(), etc.
//
//TSConvergedReason PetscTSSolver::get_converged_reason()
//{
//  PetscErrorCode ierr=0;
//
//  if (this->_initialized)
//    {
//      ierr = TSGetConvergedReason(_ts, &_reason);
//      LIBMESH_CHKERR(ierr);
//    }
//
//  return _reason;
//}

template <typename T>
void PetscTSSolver<T>::adjoint_init()
{
  START_LOG("adjoint_init()", "PetscTSSolver");
  PetscErrorCode ierr=0;

  PetscVector<Number>* adjoint_solution  = cast_ptr<PetscVector<Number>*>(&this->system().get_adjoint_solution(1));
  _lambda.push_back(adjoint_solution->vec());
  _n_cost_functions = 1;
  ierr = TSSetCostGradients(_ts,_n_cost_functions,&_lambda[0],NULL);LIBMESH_CHKERR(ierr);
  STOP_LOG("adjoint_init()", "PetscTSSolver");
}

template <typename T>
void PetscTSSolver<T>::adjoint_solve ()
{
  START_LOG("adjointsolve()", "PetscTSAdjointSolver");
  PetscErrorCode ierr=0;

  // Set the solution
  //PetscVector<Number>* PETScX  = cast_ptr<PetscVector<Number>*>(this->system().solution.get());
  ierr = TSAdjointSolve (_ts); LIBMESH_CHKERR(ierr);

  STOP_LOG("adjointsolve()", "PetscTSAdjointSolver");

  //this->system().update();
}

//------------------------------------------------------------------
// Explicit instantiations
template class PetscTSSolver<Number>;

} // namespace libMesh

#endif // #ifdef LIBMESH_HAVE_PETSC
