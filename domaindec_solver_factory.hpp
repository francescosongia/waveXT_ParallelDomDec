#ifndef HH_DOMAINDEC_SOLVER_FACTORY_HH
#define HH_DOMAINDEC_SOLVER_FACTORY_HH
#include "decomposition.hpp"
#include "domaindec_solver_base.hpp"
#include "ras.hpp"
#include "ras_pipelined.hpp"
#include <limits>
#include <memory>
#include <string>
#include <utility>

class DomainDecSolverFactory {
  /*
  @operator(): calls the solver of the method passed by string
  @createSolver(): creates a solver object and return an unique pointer to it
  */
private:
  Domain domain;
  Decomposition DataDD;

public:
  DomainDecSolverFactory(Domain dom,Decomposition dec) : domain(dom),DataDD(std::move(dec)){};

  Eigen::VectorXd operator()(const std::string &, const SpMat& A,const SpMat &b, SolverTraits traits);

  template <typename... Args>
  std::unique_ptr<DomainDecSolverBase> createSolver(std::string const &name,
                                                    Args... args);
};

#endif
