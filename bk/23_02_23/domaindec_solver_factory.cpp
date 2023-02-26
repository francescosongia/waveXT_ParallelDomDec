#include "domaindec_solver_factory.hpp"
#include <iostream>
Eigen::VectorXd DomainDecSolverFactory::operator()(const std::string &method, const SpMat &b, unsigned int max_it, double tol) {
  auto method_ptr = createSolver(method, domain,DataDD,A); // each method obj is initialized with traits

  if (method_ptr == nullptr)
    return Eigen::VectorXd::Zero(1);//std::numeric_limits<double>::quiet_NaN();
  return method_ptr->solve(A,b,max_it,tol);
}

template <typename... Args> // Args are passed to the costructor of the solver
std::unique_ptr<DomainDecSolverBase>
DomainDecSolverFactory::createSolver(std::string const &name, Args... args) {
  if (name == "RAS") {
      return std::make_unique<Ras>(std::forward<Args>(args)...);
  }
  /*
  if(name == "Bisection")
    return std::make_unique<Bisection>(std::forward<Args>(args)...);
  if(name == "Secant")
    return std::make_unique<Secant>(std::forward<Args>(args)...);
  */
  return {nullptr};
}


