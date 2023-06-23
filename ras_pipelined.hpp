#ifndef RAS_PIPELINED_HPP_
#define RAS_PIPELINED_HPP_

#include <utility>

#include "domaindec_solver_base.hpp"


class RasPipelined : public DomainDecSolverBase {
public:
  RasPipelined(Domain dom, const Decomposition& dec,const SpMat& A, int np, int current_rank=0) : 
                DomainDecSolverBase(dom,dec,A,np,current_rank),solves_(0),subt_sx_(1),zone_(2),it_waited_(0){};

  Eigen::VectorXd solve(const SpMat& A, const SpMat& b, SolverTraits traits) override;
        //ritornare tupla con u,w,obj perf

private:
    unsigned int solves_;
    unsigned int subt_sx_;
    unsigned int zone_;
    unsigned int it_waited_;
    Eigen::VectorXd precondAction(const SpMat& x,SolverTraits traits);
    unsigned int check_sx(const Eigen::VectorXd& v,double tol_sx);
};

#endif
