#ifndef DECOMPOSITION_HPP_
#define DECOMPOSITION_HPP_

#include "Eigen/Dense"
#include "domain.hpp"
#include <vector>

class Decomposition {

public:

  Decomposition(Domain& dom, unsigned int nsub_x, unsigned int nsub_t,
                double theta=0.5)
      : domain(dom), nsub_x_(nsub_x), nsub_t_(nsub_t), theta_(theta),
        sub_sizes_(dom.d()+1), overlap_back_(dom.d()+1, nsub_t * nsub_x),
        overlap_forw_(dom.d()+1, nsub_t * nsub_x),
        start_elem_(nsub_t * nsub_x){this->createDec(-1,-1);};

  Decomposition(Domain& dom, unsigned int nsub_x, unsigned int nsub_t, int n,
                int m)
      : domain(dom), nsub_x_(nsub_x), nsub_t_(nsub_t), theta_(0.5),
        sub_sizes_(dom.d()+1), overlap_back_(dom.d()+1, nsub_t * nsub_x),
        overlap_forw_(dom.d()+1, nsub_t * nsub_x),
        start_elem_(nsub_t * nsub_x){this->createDec(n, m);};


private:
  Domain domain;
  unsigned int nsub_x_;
  unsigned int nsub_t_;
  double theta_;                        // overlap weight
  std::vector<unsigned int> sub_sizes_; // [n,m] [dim_subx, dim_subt]

  // overlap for all elements in both directions. First row for space, second for time
  Eigen::MatrixXi overlap_back_;
  Eigen::MatrixXi overlap_forw_;
  
  // root (lower left) element number for each subdomain 
  std::vector<unsigned int> start_elem_; 

public:
  //getters
  auto nsub_x() const { return nsub_x_; };
  auto nsub_t() const { return nsub_t_; };
  auto sub_sizes() const {return sub_sizes_;};
  auto theta() const { return theta_; };
  auto nsub() const { return nsub_t_ * nsub_x_; };
  auto overlap_forw() const {return overlap_forw_;};
  auto overlap_back() const {return overlap_back_;};
  auto start_elem() const {return start_elem_;};

  // create the decomposition and all structures above
  void createDec(double, double); 
                            
  // collect recap info
  std::tuple<unsigned int, unsigned int, unsigned int,unsigned int,unsigned int> get_info_subK(unsigned int k) const;
  std::vector<int> basic_info_decomposition() const;
};

#endif
