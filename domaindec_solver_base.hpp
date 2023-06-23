#ifndef DOMAINDEC_SOLVER_BASE_HPP_
#define DOMAINDEC_SOLVER_BASE_HPP_

#include "domain.hpp"
#include "decomposition.hpp"
#include "sub_assignment.hpp"
#include "local_matrices.hpp"
#include "solver_traits.h"
#include <Eigen/Sparse>
#include <utility>
#include <iostream>

typedef Eigen::SparseMatrix<double>
        SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class DomainDecSolverBase {

public:
  DomainDecSolverBase(Domain dom,Decomposition  dec, LocalMatrices local_matrices) :
  domain(dom),DataDD(std::move(dec)),local_mat(std::move(local_matrices))
  {  };  //avoid copies with move, da capire!

  virtual Eigen::VectorXd solve(const SpMat& A, const SpMat& b, SolverTraits traits) = 0;
 
  virtual ~DomainDecSolverBase() = default;

protected:
  Domain domain;
  Decomposition DataDD;
  LocalMatrices local_mat;

};

#endif
