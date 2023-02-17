#ifndef DECOMPOSITION_HPP_
#define DECOMPOSITION_HPP_

#include "domain.hpp"
#include <vector>

class Decomposition {

public:
  Decomposition(Domain dom, unsigned int nsub_x, unsigned int nsub_t,
                double theta)
      : domain(dom), nsub_x(nsub_x), nsub_t(nsub_t), theta(theta){};
  Decomposition(Domain dom, unsigned int nsub_x, unsigned int nsub_t)
      : domain(dom), nsub_x(nsub_x), nsub_t(nsub_t), theta(0.5){};

private:
  Domain domain;
  double nsub_x;
  double nsub_t;
  double theta;
  std::vector<unsigned int> sub_sizes; //[n,m]
  // inserire una struttura che dia l'informazione di come sono numerati i sub,
  // in che direzione

  // inizializzare dimensione
  std::vector<unsigned int> ot_forw;
  std::vector<unsigned int> ot_back;
  std::vector<unsigned int> ox_forw;
  std::vector<unsigned int> ox_back;

  std::vector<unsigned int> start_elem;

public:
  auto nsub_x() const { return nsub_x; };
  auto nsub_t() const { return nsub_t; };
  auto theta() const { return theta; };
  auto nsub() const { return nsub_t * nsub_x; };
  // completare con altri getters

  void createDec(); // crea i vettori sopra, prima decide la decomposizione e
                    // crea m, n
};

#endif
