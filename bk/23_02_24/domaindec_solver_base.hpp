#ifndef DOMAINDEC_SOLVER_BASE_HPP_
#define DOMAINDEC_SOLVER_BASE_HPP_

#include "domain.hpp"
#include "decomposition.hpp"
#include <Eigen/Sparse>
#include <utility>

typedef Eigen::SparseMatrix<double>
        SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class DomainDecSolverBase {

public:
  DomainDecSolverBase(Domain dom,const Decomposition& dec,const SpMat& A) :
  domain(dom),DataDD(dec),R_(DataDD.nsub()),R_tilde_(DataDD.nsub()), localA_(DataDD.nsub()),localA_created(0)
  {
      this->createRMatrices();
      this->createAlocal(A);
  };  //avoid copies with move, da capire!

  // scelgo di passare A come param in mod oda creare subito le localA. lo sto facendo in solver base,
  // se poi vorrò accedere a queste matrices tramite il puntatore dovro creare metodo nella factory. Per ora lascio
  // cosi. Ha perso anche senso il controllo con la flag localA_created


  virtual Eigen::VectorXd solve(const SpMat& A, const SpMat& b, unsigned int max_it, double tol) = 0;

  std::pair<SpMat, SpMat> createRK(unsigned int k);
  void createRMatrices();  //nel parallelo considerare che ogni core si crei le sue local mat
  void createAlocal(const SpMat& A);

  std::pair<SpMat, SpMat> getRk(unsigned int k);
  SpMat getAk(unsigned int k);


  virtual ~DomainDecSolverBase() = default;

protected:
  Domain domain;
  Decomposition DataDD;
  std::vector<SpMat> R_;
  std::vector<SpMat> R_tilde_;
  std::vector<SpMat> localA_;
  int localA_created;




  // creare domain decomposition, una struttura con tutte le info per sub k
  // (m,n,overlaps). Pensare a come collegare ogni elem a un sub. ok che nel
  // progetto avevamo solo cartesiana e quindi basta dire chi è l'elemnto di
  // sinistra e poi mi muovo con m e n ma per generalizzare posso pensare alla
  // creazione di una 'femregion' per i sub.
  // NON posso creare struct per collegare ogni elem a il sub perchè ce ne
  // sarebbero 5000. Qiuindi rimanere cartesiani e va bene spostarsi con il lato
  // sx.

  // inserisco collegamento con local matrix che va a creare tutte le local
  // matrices che salvo qua. poi usare pointer?
  // inizio a creare local matric come matrici sparse e le raccolgo in vettore.
  // poi piu avanti posso salvarmi gli indici e prelevare da un vettore
  // direttamente le componenti che mi interessano: in uqesto caso basta
  // salavarmi un vettore di coppie (i,j).
  // Rimane da capire come ottenere Aj senza moltiplicazione con matrici R,
  // anche qui devo capire queli (ij) tenermi pero poi dovranno essere messi nel
  // punto giusto di una matrice e quindi devo sapaere anche la pos finale di
  // dove mettelri, non è solo un vettore
};

#endif
