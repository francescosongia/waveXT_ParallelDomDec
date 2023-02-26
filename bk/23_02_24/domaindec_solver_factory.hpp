#ifndef HH_DOMAINDEC_SOLVER_FACTORY_HH
#define HH_DOMAINDEC_SOLVER_FACTORY_HH
#include "decomposition.hpp"
#include "domaindec_solver_base.hpp"
#include "ras.hpp"
#include <limits>
#include <memory>
#include <string>
#include <utility>

class DomainDecSolverFactory {
  /*
  @traits: options and data for the solvers
  @operator(): calls the solver of the method passed by string
  @createSolver(): creates a solver object and return an unique pointer to it
  */
private:
  Domain domain;
  Decomposition DataDD;
  SpMat A;

public:
  DomainDecSolverFactory(Domain dom,Decomposition dec,const SpMat& A) : domain(dom),DataDD(std::move(dec)),A(A){};
  //pass also A and initialize localA

  Eigen::VectorXd operator()(const std::string &, const SpMat &b, unsigned int max_it, double tol);

  template <typename... Args>
  std::unique_ptr<DomainDecSolverBase> createSolver(std::string const &name,
                                                    Args... args);



};

#endif
