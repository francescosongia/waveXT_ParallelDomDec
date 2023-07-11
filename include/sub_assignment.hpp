#ifndef SUB_ASSIGNMENT_HPP_
#define SUB_ASSIGNMENT_HPP_

#include "decomposition.hpp"
#include "policyLA.hpp"
#include <iostream>

template<class LA>
class SubAssignment {

public:
  SubAssignment(int np, const Decomposition& dec)
      : np_(np), nsub_x_(dec.nsub_x()), nsub_t_(dec.nsub_t()), sub_division_(np,dec.nsub_t()), sub_division_vec_(np)
      {this->createSubDivision();};

  //per ora assumiamo solo parallelizzazione in tempo, poi si puo fare i casi con strategie diverse
  // comunque si tratta solo di fornire una strategia per dividere i sub tra i vari core, potrei creare
  // costruttore che legge una matrice grande come domain diviso e ogni elemnto è un numero che identifica core

  // con l'aggiunta di sub_division_Vec perde di importanza la matrice sub_division (che potrei pensare solo come 
  // argomento per una versione di un costrcuttore). meglio salvarsi cosi un vector per ogni processore
private:
  int np_;
  int nsub_x_;
  int nsub_t_;

  Eigen::MatrixXi sub_division_;
  std::vector<Eigen::VectorXi> sub_division_vec_;

public:
  auto np() const { return np_; };
  auto nsub_x() const { return nsub_x_; };
  auto nsub_t() const { return nsub_t_; };
  auto sub_division() const { return sub_division_; };
  auto sub_division_vec() const { return sub_division_vec_; };

  void createSubDivision(){
    LA func_wrapper;
    return func_wrapper.createSubDivision(np_,nsub_x_,nsub_x_,sub_division_,sub_division_vec_);
  };
  
};


#endif
