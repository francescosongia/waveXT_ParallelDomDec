#ifndef DOMAINDEC_SOLVER_BASE_HPP_
#define DOMAINDEC_SOLVER_BASE_HPP_

#include "local_matrices.hpp"

#include "solver_traits.h"
#include "solver_results.hpp"
#include <Eigen/Sparse>
#include <utility>
#include <iostream>
#include <chrono>
#include <cassert>
#include "mpi.h"



//typedef Eigen::SparseMatrix<double,Eigen::RowMajor>
typedef Eigen::SparseMatrix<double>
        SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

template <class P, class LA>
class DomainDecSolverBase {

public:

  DomainDecSolverBase(Domain dom,const Decomposition&  dec,const LocalMatrices<LA>& local_matrices, const SolverTraits& traits) :
  domain(dom),DataDD(std::move(dec)),local_mat(std::move(local_matrices)), traits_(traits),
  localLU(local_mat.get_size_vector_localmat())
  {
    int count{0};
    auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
    if(local_mat.sub_assignment().custom_matrix()==false){
      for(unsigned int k : sub_division_vec){    
          // Eigen::SparseLU<SpMat > lu;
          // lu.compute(local_mat.getAk(k));
          localLU[count].compute(local_mat.getAk(k));
          count++;
      }
    }
    else{
      for(unsigned int k : sub_division_vec){    
          // Eigen::SparseLU<SpMat > lu;
          // lu.compute(local_mat.getAk(k));
          localLU[k-1].compute(local_mat.getAk(k));
      }
    }
  };  
  
  virtual SolverResults solve(const SpMat& A, const SpMat& b) = 0;  
 
  SolverTraits traits() const { return traits_;};

  const Eigen::SparseLU<SpMat>& get_LU_k(unsigned int k) const
  {
    assert(k<=this->DataDD.nsub());
    auto local_k= this->local_mat.sub_assignment().idxSub_to_LocalNumbering(k, this->local_mat.rank());
    //k = (this->local_mat.local_num()) ? k-this->local_mat.rank_group_la()*this->local_mat.get_size_vector_localmat() : k;
    return this->localLU[local_k-1];
  };
  
  virtual ~DomainDecSolverBase() = default;

protected:
  virtual Eigen::VectorXd precondAction(const SpMat& x) = 0;
  
  Domain domain;
  Decomposition DataDD;
  LocalMatrices<LA> local_mat;
  SolverTraits traits_;
  std::vector<Eigen::SparseLU<SpMat>> localLU;
};

#endif
