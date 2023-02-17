#include "domaindec_solver_factory.hpp"
#include <iostream>
double DomainDecSolverFactory::operator()(const std::string &method) {
  auto method_ptr = createSolver(
      method, DataDD); // each method obj is initialized with traits
  if (method_ptr == nullptr)
    return std::numeric_limits<double>::quiet_NaN();
  return method_ptr->solve();
}

template <typename... Args> // Args are passed to the costructor of the solver
std::unique_ptr<SolverBase>
DomainDecSolverFactory::createSolver(std::string const &name, Args... args) {
  if (name == "RAS")
    return std::make_unique<RAS>(std::forward<Args>(args)...);
  /*
  if(name == "Bisection")
    return std::make_unique<Bisection>(std::forward<Args>(args)...);
  if(name == "Secant")
    return std::make_unique<Secant>(std::forward<Args>(args)...);
  */
  return std::unique_ptr<SolverBase>(nullptr);
}
