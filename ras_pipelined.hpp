#ifndef RAS_PIPELINED_HPP_
#define RAS_PIPELINED_HPP_

#include <utility>

#include "domaindec_solver_base.hpp"


class RasPipelined : public DomainDecSolverBase {
public:
  RasPipelined(Domain dom, const Decomposition& dec, LocalMatrices local_matrices) : 
                DomainDecSolverBase(dom,dec,local_matrices),solves_(0),subt_sx_(1),zone_(2),it_waited_(0), matrix_domain_(dec.nsub_t(),dec.nsub_x())
                {
                  Eigen::VectorXi list=Eigen::VectorXi::LinSpaced(dec.nsub(),1,dec.nsub());  
                  matrix_domain_=list.transpose().reshaped(dec.nsub_t(),dec.nsub_x());  //serve traspose?
                };

  Eigen::VectorXd solve(const SpMat& A, const SpMat& b, SolverTraits traits) override;
        //ritornare tupla con u,w,obj perf

private:
    unsigned int solves_;
    unsigned int subt_sx_;
    unsigned int zone_;
    unsigned int it_waited_;
    Eigen::MatrixXi matrix_domain_;

    Eigen::VectorXd precondAction(const SpMat& x,SolverTraits traits);
    unsigned int check_sx(const Eigen::VectorXd& v,double tol_sx);
};

#endif
