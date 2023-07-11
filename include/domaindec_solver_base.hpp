#ifndef DOMAINDEC_SOLVER_BASE_HPP_
#define DOMAINDEC_SOLVER_BASE_HPP_

#include "domain.hpp"
#include "decomposition.hpp"
#include "local_matrices.hpp"
#include "sub_assignment.hpp"

#include "solver_traits.h"
#include "solver_results.hpp"
#include <Eigen/Sparse>
#include <utility>
#include <iostream>
#include <chrono>
#include "mpi.h"

typedef Eigen::SparseMatrix<double>
        SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

template <class P, class LA>
class DomainDecSolverBase {

public:
  DomainDecSolverBase(Domain dom,Decomposition  dec, LocalMatrices<LA> local_matrices, const SolverTraits& traits) :
  domain(dom),DataDD(std::move(dec)),local_mat(std::move(local_matrices)), traits_(traits)
  {};  //avoid copies with move, da capire!

  virtual SolverResults solve(const SpMat& A, const SpMat& b) = 0;  //Eigen::VectorXd
 
  virtual ~DomainDecSolverBase() = default;

protected:
  virtual Eigen::VectorXd precondAction(const SpMat& x) = 0;
  
  Domain domain;
  Decomposition DataDD;
  LocalMatrices<LA> local_mat;
  SolverTraits traits_;
};

#endif
