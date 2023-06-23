#ifndef RAS_HPP_
#define RAS_HPP_

#include <utility>

#include "domaindec_solver_base.hpp"


class Ras : public DomainDecSolverBase {
public:
  Ras(Domain dom, const Decomposition& dec,const SpMat& A,int np, int current_rank=0) : DomainDecSolverBase(dom,dec,A, np,current_rank){};

  Eigen::VectorXd solve(const SpMat& A, const SpMat& b, SolverTraits traits) override;
        //ritornare tupla con u,w,obj perf

private:
  Eigen::VectorXd precondAction(const SpMat& x);
};

#endif
