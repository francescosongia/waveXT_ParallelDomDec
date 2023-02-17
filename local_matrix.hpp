#ifndef LOCAL_MATRIX_HPP_
#define LOCAL_MATRIX_HPP_

#include "solver_base.hpp"

#include <Eigen/Sparse>
#include <iostream>
#include <vector>

typedef Eigen::SparseMatrix<double>
    SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class LocalMatrix {
public:
  LocalMatrix(Decomposition d) : DataDD(d){};

  SpMat createLocalMat(unsigned int k);

private:
  Decomposition DataDD;
};

#endif
