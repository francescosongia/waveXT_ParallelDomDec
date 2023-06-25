#ifndef LOCAL_MATRICES_HPP_
#define LOCAL_MATRICES_HPP_

#include "domain.hpp"
#include "decomposition.hpp"
#include "sub_assignment.hpp"
#include "solver_traits.h"
#include <Eigen/Sparse>
#include <utility>
#include <iostream>

typedef Eigen::SparseMatrix<double>
        SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class LocalMatrices {

public:
  LocalMatrices(Domain dom,Decomposition  dec,const SpMat& A, int np, int current_rank_=0) :
  domain(dom),DataDD(std::move(dec)),R_(DataDD.nsub()),R_tilde_(DataDD.nsub()), localA_(DataDD.nsub()),localA_created_(0),
  sub_assignment_(SubAssignment(np,dec)),current_rank(current_rank_)
  {
      this->createRMatrices();
      this->createAlocal(A);
      std::cout<<"local matrices created"<<std::endl;
  };  //avoid copies with move, da capire!

 
  std::pair<SpMat, SpMat> createRK(unsigned int k);
  void createRMatrices();  
  void createAlocal(const SpMat& A);  
  //nel parallelo considerare che ogni core si crei le sue local mat
  // PASSAGGIO SUCC: non allocare per R_ ecc un vettore di dimensione nsub totali ma solo grande quanto i sottodimini gestiti da quel core
  //                 quando creo la divisione in sub_assignment creo vettore di dimensione piccola (il numero di sub gestiti dal core) e creo 
  //                 relazione tra il primo elem di questo vettore e il numero del sottodominio


  std::pair<SpMat, SpMat> getRk(unsigned int k) const;
  SpMat getAk(unsigned int k) const;

  auto localA_created() {return localA_created_;};
  auto rank() {return current_rank;};
  auto sub_assignment() {return sub_assignment_;};

private:
  Domain domain;
  Decomposition DataDD; 
  std::vector<SpMat> R_;
  std::vector<SpMat> R_tilde_;
  std::vector<SpMat> localA_;
  int localA_created_;
  SubAssignment sub_assignment_;
  int current_rank;



  // poi piu avanti posso salvarmi gli indici e prelevare da un vettore
  // direttamente le componenti che mi interessano: in uqesto caso basta
  // salavarmi un vettore di coppie (i,j).
  // Rimane da capire come ottenere Aj senza moltiplicazione con matrici R,
  // anche qui devo capire queli (ij) tenermi pero poi dovranno essere messi nel
  // punto giusto di una matrice e quindi devo sapaere anche la pos finale di
  // dove mettelri, non è solo un vettore

  // comunque non è detto convenga, sono già ben organizzate con matrici sparse, 
  // già una moltiplicazione tra loro è ottimizzata per prendere comp diverse da zero


};

#endif
