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


typedef Eigen::SparseMatrix<double>
        SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


/*
SOLVER BASE VIRTUAL CLASS
solve method will be overridden by the different policies
*/
template <class P, class LA>
class DomainDecSolverBase {

public:

  DomainDecSolverBase(Domain dom,const Decomposition&  dec,const LocalMatrices<LA>& local_matrices, const SolverTraits& traits) :
  domain(dom),DataDD(std::move(dec)),local_mat(std::move(local_matrices)), traits_(traits),
  localLU(local_mat.get_size_vector_localmat())
  {
    // precomputed the LU factorization for all local matrices
    int count{0};
    auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
    if(local_mat.sub_assignment().custom_matrix()==false){
      for(unsigned int k : sub_division_vec){    
          localLU[count].compute(local_mat.getAk(k));
          count++;
      }
    }
    else{
      for(unsigned int k : sub_division_vec)  
          localLU[k-1].compute(local_mat.getAk(k));
    }
  };  
  
  virtual SolverResults solve(const SpMat& A, const SpMat& b) = 0;  
  

  //getter for traits
  SolverTraits traits() const { return traits_;};

  //getter for LU factorization
  const Eigen::SparseLU<SpMat>& get_LU_k(unsigned int k) const
  {
    assert(k<=this->DataDD.nsub());
    auto iscustom = this->local_mat.sub_assignment().custom_matrix();
    auto local_k = (!iscustom)? this->local_mat.sub_assignment().idxSub_to_LocalNumbering(k, this->local_mat.rank()):k;
    
    return this->localLU[local_k-1];
  };
  
  virtual ~DomainDecSolverBase() = default;

protected:
  
  Domain domain;
  Decomposition DataDD;
  LocalMatrices<LA> local_mat;
  SolverTraits traits_;
  std::vector<Eigen::SparseLU<SpMat>> localLU;
};

#endif
