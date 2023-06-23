#ifndef SUB_ASSIGNMENT_HPP_
#define SUB_ASSIGNMENT_HPP_

#include "decomposition.hpp"

class SubAssignment {

public:
  SubAssignment(int np, Decomposition dec)
      : np_(np), nsub_x_(dec.nsub_x()), nsub_t_(dec.nsub_t()), sub_division_(np,dec.nsub_t())
      {this->createSubDivision();};

  //per ora assumiamo solo parallelizzazione in tempo, poi si puo fare i casi con strategie diverse
  // comunque si tratta solo di fornire una strategia per dividere i sub tra i vari core, potrei creare
  // costruttore che legge una matrice grande come domain diviso e ogni elemnto è un numero che identifica core

private:
  int np_;
  int nsub_x_;
  int nsub_t_;

  Eigen::MatrixXi sub_division_;

public:
  auto np() const { return np_; };
  auto nsub_x() const { return nsub_x_; };
  auto nsub_t() const { return nsub_t_; };
  auto sub_division() const { return sub_division_; };

  void createSubDivision();
  
};

#endif
