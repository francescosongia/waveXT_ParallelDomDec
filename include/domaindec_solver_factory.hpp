#ifndef HH_DOMAINDEC_SOLVER_FACTORY_HH
#define HH_DOMAINDEC_SOLVER_FACTORY_HH
#include "decomposition.hpp"
#include "domaindec_solver_base.hpp"
#include "ras_pipelined.hpp"
#include "ras.hpp"
#include <limits>
#include <memory>
#include <string>
#include <utility>

template< class P,class LA>
class DomainDecSolverFactory {
  
private:
  Domain domain;
  Decomposition DataDD;
  LocalMatrices<LA> local_matrices;
  SolverTraits traits_;

public:
  
  DomainDecSolverFactory(Domain dom,Decomposition dec, LocalMatrices<LA> local_mat,const SolverTraits& traits) :
       domain(dom),DataDD(std::move(dec)), local_matrices(local_mat), traits_(traits)
       {};

  SolverResults operator()(const std::string& method, const SpMat& A,const SpMat &b){
    auto method_ptr = this->createSolver(method, domain,DataDD,local_matrices,traits_); // each method obj is initialized with traits
    
    if (method_ptr == nullptr)
      return SolverResults(Eigen::VectorXd::Zero(1),0,0,traits_,DataDD);//std::numeric_limits<double>::quiet_NaN();
    
    return method_ptr->solve(A,b);
  };

  template <typename... Args>
  std::unique_ptr<DomainDecSolverBase<P,LA>> createSolver(std::string const &name,Args... args)
  {
    if (name == "RAS") {
      return std::make_unique<Ras<P,LA>>(std::forward<Args>(args)...);
    }
    if(name == "PIPE")
      return std::make_unique<RasPipelined<P,LA>>(std::forward<Args>(args)...);

    return {nullptr};
  };
};

#endif
