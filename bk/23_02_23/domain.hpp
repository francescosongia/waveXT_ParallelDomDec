#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

class Domain {

public:
  Domain(unsigned int nx, unsigned int nt, double x, double t, unsigned int nln)
      : nx_(nx), nt_(nt), X_(x), T_(t), nln_(nln), d_(1){};
  // X deve essere un vettore di d dimensione, inizializzo campo X in base a d:
  // se d=1 allora X è double altrimenti sarà vettore

private:
  unsigned int nx_;
  unsigned int nt_;
  double X_;
  double T_;
  unsigned int nln_;
  unsigned int d_;
  // struttura femregion

public:
  auto nx() const { return nx_; };
  auto nt() const { return nt_; };
  auto X() const { return X_; };
  auto T() const { return T_; };
  auto nln() const { return nln_; };
  auto d() const { return d_; };
};

#endif
