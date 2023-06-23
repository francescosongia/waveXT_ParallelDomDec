#ifndef DOMAINDEC_SOLVER_BASE_HPP_
#define DOMAINDEC_SOLVER_BASE_HPP_

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

class DomainDecSolverBase {

public:
  DomainDecSolverBase(Domain dom,Decomposition  dec,const SpMat& A, int np, int current_rank_) :
  domain(dom),DataDD(std::move(dec)),R_(DataDD.nsub()),R_tilde_(DataDD.nsub()), localA_(DataDD.nsub()),localA_created(0),sub_assignment(SubAssignment(np,dec)),current_rank(current_rank_)
  {
      this->createRMatrices();
      this->createAlocal(A);
      std::cout<<"local matrices created"<<std::endl;
  };  //avoid copies with move, da capire!

  // scelgo di passare A come param in modo da creare subito le localA. lo sto facendo in solver base,
  // se poi vorrò accedere a queste matrices tramite il puntatore dovro creare metodo nella factory. Per ora lascio
  // cosi. Ha perso anche senso il controllo con la flag localA_created.
  // va bene lasciare cosi. poi se volgio accedere a localA o localR dovrò creare oggetto Ras e usare metodi di questa classe


  virtual Eigen::VectorXd solve(const SpMat& A, const SpMat& b, SolverTraits traits) = 0;
 
  std::pair<SpMat, SpMat> createRK(unsigned int k);
  void createRMatrices();  
  void createAlocal(const SpMat& A);  
  //nel parallelo considerare che ogni core si crei le sue local mat
  // PASSAGGIO SUCC: non allocare per R_ ecc un vettore di dimensione nsub totali ma solo grande quanto i sottodimini gestiti da quel core
  //                 quando creo la divisione in sub_assignment creo vettore di dimensione piccola (il numero di sub gestiti dal core) e creo 
  //                 relazione tra il primo elem di questo vettore e il numero del sottodominio


  std::pair<SpMat, SpMat> getRk(unsigned int k) const;
  SpMat getAk(unsigned int k) const;


  virtual ~DomainDecSolverBase() = default;

protected:
  Domain domain;
  Decomposition DataDD;
  std::vector<SpMat> R_;
  std::vector<SpMat> R_tilde_;
  std::vector<SpMat> localA_;
  int localA_created;
  SubAssignment sub_assignment;
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
