#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dof_map.h"

#include "petsc_ts_system.h"
//#include "navier_stokes_assemble.h"

namespace libMesh
{

PetscTSSystem::PetscTSSystem(EquationSystems& es,
                             const std::string& name_in,
                             const unsigned int number_in):
  Parent   (es, name_in, number_in),
  // Init TS solver
  ts_solver(UniquePtr<PetscTSSolver<Number> >(new PetscTSSolver<Number>(*this))),
  time(0),
  timestep(0)
{
}

PetscTSSystem::~PetscTSSystem ()
{
  this->clear();
}

void PetscTSSystem::clear ()
{
  Parent::clear();
}

void PetscTSSystem::init ()
{
  Parent::init();
}

void PetscTSSystem::reinit ()
{
  // Initialize parent data
  Parent::reinit();

  // Re-initialize the TS solver interface
  ts_solver->clear();
}

void PetscTSSystem::solve ()
{
  // Log how long the nonlinear solve takes.
  START_LOG("solve()", "PetscTSSystem");

  // What parameters can we set for TS solver? Copy them to TS solver.
  this->set_solver_parameters();

  // There is solver constructed through build(), but not init()
  ts_solver->init();

  // Call TS solver to solve the system
  ts_solver->solve();

  time = ts_solver->get_time();
  timestep = ts_solver->get_time_step_number();

  // Stop logging the nonlinear solve
  STOP_LOG("solve()", "PetscTSSystem");

  // Update the system after the solve
 // this->update();
}

// F(t,U,U_t) in DAE form. Required by PETSc implicit TS methods
void PetscTSSystem::IFunction (Real time,
                               NumericVector<Number>& X,
                               NumericVector<Number>& Xdot,
                               NumericVector<Number>& F)
{
}

// Compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t) is IFunction
void PetscTSSystem::IJacobian (Real time,
                               NumericVector<Number>& X,
                               NumericVector<Number>& Xdot,
                               Real shift,
                               SparseMatrix<Number>& IJ,
                               SparseMatrix<Number>& IJpre)
{
}

void PetscTSSystem::monitor (int step, Real time,
                             NumericVector<Number>& X)
{
}

// Set up adjoint solver
void PetscTSSystem::petsc_adjoint_init(int n_cost_functions)
{
  START_LOG("petsc_adjoint_init()", "PetscTSSystem");
  ts_solver->adjoint_init(n_cost_functions);
  STOP_LOG("petsc_adjoint_init()", "PetscTSSystem");
}

void PetscTSSystem::petsc_adjoint_solve ()
{
  // Log how long the adjoint solve takes.
  START_LOG("petsc_adjoint_solve()", "PetscTSSystem");

  // call ts solver to solve the system
  ts_solver->adjoint_solve();

  // Stop logging the nonlinear solve
  STOP_LOG("petsc_adjoint_solve()", "PetscTSSystem");

  // Update the system after the solve
 // this->update();
}

// Override existing settings for TS solver.
void PetscTSSystem::set_solver_parameters ()
{
  // Get a reference to the EquationSystems
  const EquationSystems& es = this->get_equation_systems();

  // Get the user-specified parameters.
  const Real   t0 = es.parameters.get<Real>("initial time");
  const Real maxt = es.parameters.get<Real>("final time");
  const Real   dt = es.parameters.get<Real>("dt");
  const unsigned int nsteps = es.parameters.get<unsigned int>("time steps");
  const bool do_adjoint = es.parameters.get<bool>("adjoint");

  // Set the parameters for TS solver
  if (ts_solver.get() )
  {
    ts_solver->set_duration(t0, maxt);
    ts_solver->set_timestep (dt,nsteps);
    ts_solver->set_adjoint(do_adjoint);
  }
}

} // end of namespace
