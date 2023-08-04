#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

class Domain {

public:
  Domain(unsigned int nx, unsigned int nt, double x, double t, unsigned int nln)
      : nx_(nx), nt_(nt), X_(x), T_(t), nln_(nln), d_(1){};

private:
  // finite elements 
  unsigned int nx_;
  unsigned int nt_;
  
  // domain size. [0,X_] x [0,T_]
  double X_;
  double T_;

  // dof for each finite elem
  unsigned int nln_;

  // space size
  unsigned int d_;

// simple getters
public:
  auto nx() const { return nx_; };
  auto nt() const { return nt_; };
  auto X() const { return X_; };
  auto T() const { return T_; };
  auto nln() const { return nln_; };
  auto d() const { return d_; };
};

#endif
