#ifndef SOLVERRESULTS_HPP_
#define SOLVERRESULTS_HPP_

#include "decomposition.hpp"
#include "solver_traits.h"
#include <iostream>

class SolverResults {

public:
  SolverResults(Eigen::VectorXd uw, unsigned int solves, double time, SolverTraits straits, Decomposition dec)
      : uw_(uw), solves_(solves), time_(time), max_it_(straits.max_it()), tol_(straits.tol()), tol_pipe_sx_(straits.tol_pipe_sx()),
        it_wait_(straits.it_wait()), sub_sizes_(2), max_overlaps_(2)
      {
        auto dec_info = dec.basic_info_decomposition();
        nsub_x_ = dec_info[0];
        nsub_t_ = dec_info[1];
        sub_sizes_[0] = dec_info[2];
        sub_sizes_[1] = dec_info[3];
        max_overlaps_[0] = dec_info[4];
        max_overlaps_[1] = dec_info[5];
      };

private:
  Eigen::VectorXd uw_;
  unsigned int solves_;
  double time_;

  //solver_traits
  unsigned int max_it_;
  double tol_;
  double tol_pipe_sx_;
  unsigned int it_wait_;

  //decomposition
  int nsub_x_;
  int nsub_t_;
  std::vector<int> sub_sizes_;
  std::vector<int> max_overlaps_;

public:
  Eigen::VectorXd getUW() const {return uw_;};
  Eigen::VectorXd getU() const {return uw_(Eigen::seq(0,uw_.size()/2 -1));};
  Eigen::VectorXd getW() const {return uw_(Eigen::seq(uw_.size()/2, uw_.size()-1));};

  //funzione che creano file testo con le info
  //void getSolverInfo() const;

  //postprocessing functions to plot u,w 
  

};


#endif
