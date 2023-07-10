#ifndef POLICIES_HPP_
#define POLICIES_HPP_
/*
#include <iostream>
#include "domaindec_solver_base.hpp"
#include "ras.hpp"

struct Parallel : public Ras<Parallel>
{
  public:
    Eigen::VectorXd 
    precondAction(const SpMat& x) 
    {
      Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
      Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);

      // questo for devo prendere i k giusti che vede quel core, mi appoggio su sub_assingment
      auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
      for(unsigned int k : sub_division_vec){    
          Eigen::SparseLU<SpMat > lu;
          lu.compute(local_mat.getAk(k));
          auto temp = local_mat.getRk(k);
          uk = lu.solve(temp.first*x);
          z=z+(temp.second.transpose())*uk;
          }

      MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return z;
  }
};


struct Sequential : public Ras<Sequential>
{
public:
  Eigen::VectorXd precondAction(const SpMat& x) {
    Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    //Eigen::VectorXd xk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    for(unsigned int k=1;k<=DataDD.nsub();++k){

        Eigen::SparseLU<SpMat > lu;
        lu.compute(local_mat.getAk(k));

        auto temp = local_mat.getRk(k);

        uk = lu.solve(temp.first*x);

        z=z+(temp.second.transpose())*uk;
    }
    return z;
}

};
*/

#endif
