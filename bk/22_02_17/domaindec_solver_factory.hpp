#ifndef HH_DOMAINDEC_SOLVER_FACTORY_HH
#define HH_DOMAINDEC_SOLVER_FACTORY_HH
#include "decomposition.hpp"
#include "domaindec_solver_base.hpp"
#include "ras.hpp"
#include <limits>
#include <memory>
#include <string>

class DomainDecSolverFactory {
  /*
  @traits: options and data for the solvers
  @operator(): calls the solver of the method passed by string
  @createSolver(): creates a solver object and return an unique pointer to it
  */
private:
  Decomposition DataDD;

public:
  DomainDecSolverFactory(Decomposition d) : DataDD(d){};

  double operator()(const std::string &);

  template <typename... Args>
  std::unique_ptr<DomainDecSolverBase> createSolver(std::string const &name,
                                                    Args... args);
};

#endif
