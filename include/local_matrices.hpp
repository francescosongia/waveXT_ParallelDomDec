#ifndef LOCAL_MATRICES_HPP_
#define LOCAL_MATRICES_HPP_

#include "decomposition.hpp"
#include "sub_assignment.hpp"
#include "solver_traits.h"
#include <Eigen/Sparse>
#include <utility>
#include <iostream>


//typedef Eigen::SparseMatrix<double,Eigen::RowMajor>
typedef Eigen::SparseMatrix<double>
        SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

// ABBIAMO SEMPRE USATO COLUMN MAJOR MATRICES. NELLA PARALLELIZZAZIONE LA CI SERVE PERÒ
// ACCEDERE CON MIDDLEROWS AD ALCUNE RIGHE. CAPIRE SE È PIU EFFICIENTE SALVARE LA MATRICI 
// COME ROW-MAJOR. IN QUESTO CASO CONTROLLARE CHE NON CAMBI NIENTE NELLE ALTRE IMPLEMENTAZIONI

template<class LA>
class LocalMatrices {

private:
  Domain domain;
  Decomposition DataDD; 
  std::vector<SpMat> R_;
  std::vector<SpMat> R_tilde_;
  std::vector<SpMat> localA_;
  int localA_created_;
  int current_rank_;
  int rank_group_la_;
  bool local_numbering;
  LA sub_assignment_;
  

  std::pair<SpMat, SpMat> createRK(unsigned int k)
  {
    unsigned int start_elem,ox_forw,ot_forw,ox_back,ot_back;
    std::tie(start_elem,ox_forw,ot_forw,ox_back,ot_back) = this->DataDD.get_info_subK(k);
    unsigned int m,n,nt,nx,nln;
    double theta;
    m = this->DataDD.sub_sizes()[1];
    n = this->DataDD.sub_sizes()[0];
    nt = this->domain.nt();
    nx = this->domain.nx();
    nln = this->domain.nln(); 
    theta = this->DataDD.theta();

    Eigen::ArrayXi indexcol = Eigen::ArrayXi::Zero(nln*m*n*2);
    Eigen::ArrayXd values = Eigen::ArrayXd::Zero(nln*m*n*2);

    for (size_t q=0;q<=n-1;++q) {
        auto temp = Eigen::VectorXi::LinSpaced(nln * m, q * nt * nln + (start_elem - 1) * nln,
                                               (q * nt + m) * nln + (start_elem - 1) * nln-1);
        indexcol(Eigen::seq(q * nln * m, nln * m * (q + 1)-1)) << temp;
    }
    indexcol(Eigen::seq(nln*m*n,nln*m*n*2 - 1))=indexcol(Eigen::seq(0,nln*m*n-1))+(nt*nx*nln); //extend for w

    auto temp1 = Eigen::ArrayXd::Constant(ot_back*nln,(1-theta)/2);
    auto temp2 = Eigen::ArrayXd::Constant((m-ot_forw-ot_back)*nln,0.5);
    auto temp3 = Eigen::ArrayXd::Constant(ot_forw*nln,theta/2);
    for(size_t q=0;q<ox_back;++q){
        values(Eigen::seq(q*nln*m,nln*m*(q+1)-1))<<temp1,temp2,temp3;
    }
    for(size_t q=ox_back;q<n-ox_forw;++q){
        values(Eigen::seq(q*nln*m,nln*m*(q+1)-1))<<temp1*2,temp2*2,temp3*2;
    }
    for(size_t q=n-ox_forw;q<n;++q){
        values(Eigen::seq(q*nln*m,nln*m*(q+1)-1))<<temp1,temp2,temp3;
    }
    values(Eigen::seq(nln*m*n,nln*m*n*2 - 1))=values(Eigen::seq(0,nln*m*n-1)); //extend for w

    std::vector<T> tripletList;
    tripletList.reserve(nln*m*n*2);
    std::vector<T> tripletList_tilde;
    tripletList_tilde.reserve(nln*m*n*2);
    for(size_t i=0;i<nln*m*n*2;++i){
        tripletList.emplace_back(i,indexcol[i],1.);
        tripletList_tilde.emplace_back(i,indexcol[i],values[i]);
    }
    SpMat Rk(nln*m*n*2,nln*nt*nx*2),Rk_tilde(nln*m*n*2,nln*nt*nx*2);
    Rk.setFromTriplets(tripletList.begin(), tripletList.end());
    Rk_tilde.setFromTriplets(tripletList_tilde.begin(), tripletList_tilde.end());  //already compressed

    return std::make_pair(Rk, Rk_tilde);
  };

  void createAlocal(const SpMat& A)
  {
    unsigned int m,n,nt,nx,nln;
    nt = this->domain.nt();
    nx = this->domain.nx();
    nln = this->domain.nln();
    m = this->DataDD.sub_sizes()[1];
    n = this->DataDD.sub_sizes()[0];
    SpMat temp(nln*m*n*2,nln*nt*nx*2);
    auto sub_division_vec = this->sub_assignment_.sub_division_vec()[this->current_rank_];
    for(unsigned int k : sub_division_vec){    
        k = (this->local_numbering) ? k-this->rank_group_la_*this->R_.size() : k;
        temp=this->R_[k-1]*A;
        this->localA_[k-1]=temp*(this->R_[k-1].transpose());
    }
    this->localA_created_=1;
  };  

  
 

public:

  LocalMatrices(Domain dom,const Decomposition&  dec,const SpMat& A, int np, int current_rank=0) :
    domain(dom),DataDD(std::move(dec)),R_(DataDD.nsub()),R_tilde_(DataDD.nsub()), localA_(DataDD.nsub()),localA_created_(0),
    current_rank_(current_rank),rank_group_la_(current_rank),local_numbering(false), 
    sub_assignment_(LA(np,DataDD.nsub_x(),DataDD.nsub_t()))
  {   
      
      if(np/DataDD.nsub_x() > 1)
        this->rank_group_la_ = current_rank_%(np/DataDD.nsub_x());
      
      this->sub_assignment_.createSubDivision();
      this->createRMatrices();
      this->createAlocal(A);
      std::cout<<"local matrices created"<<std::endl; 
      std::cout<<"size of R_: "<<R_.size()<<std::endl;

  }; 


  // constructor for custom matrix in sub assignment
  LocalMatrices(Domain dom,const Decomposition&  dec,const SpMat& A, int np, int current_rank, Eigen::MatrixXi custom_mat) :
    domain(dom),DataDD(std::move(dec)),R_(DataDD.nsub()),R_tilde_(DataDD.nsub()), localA_(DataDD.nsub()),localA_created_(0),
    current_rank_(current_rank),rank_group_la_(current_rank),local_numbering(false), 
    sub_assignment_(LA(np,DataDD.nsub_x(),DataDD.nsub_t(),custom_mat))
  {   
      
      if(np/DataDD.nsub_x() > 1)
        this->rank_group_la_ = current_rank_%(np/DataDD.nsub_x());
      
      this->sub_assignment_.createSubDivision();
      this->createRMatrices();
      this->createAlocal(A);
      std::cout<<"local matrices created"<<std::endl; 
      std::cout<<"size of R_: "<<R_.size()<<std::endl;

  };  //avoid copies with move, da capire!


  void createRMatrices()
  {
    auto sub_division_vec = this->sub_assignment_.sub_division_vec()[this->current_rank_];
    auto size_assigned = sub_division_vec.size();

    // parallel framework
    if(size_assigned < this->DataDD.nsub() && this->sub_assignment_.custom_matrix()==false){     
        this->local_numbering = true;
        this->R_.resize(size_assigned);
        this->R_tilde_.resize(size_assigned);
        this->localA_.resize(size_assigned);
        for(unsigned int k : sub_division_vec){
            auto k_local = k - this->rank_group_la_*size_assigned; 
            std::pair<SpMat, SpMat> res= this->createRK(k);
            this->R_[k_local-1] = std::move(res.first);
            this->R_tilde_[k_local-1] = std::move(res.second);
            }  
    }
    //sequential framework
    else{
        for(unsigned int k : sub_division_vec){
            std::pair<SpMat, SpMat> res= this->createRK(k);
            this->R_[k-1]= std::move(res.first);
            this->R_tilde_[k-1] = std::move(res.second);
            }
    }
  }; 


  std::pair<SpMat, SpMat> getRk(unsigned int k) const
  {
    k = (this->local_numbering) ? k-this->rank_group_la_*this->R_.size() : k;
    return std::make_pair(this->R_[k-1], this->R_tilde_[k-1]);
  };



  SpMat getAk(unsigned int k) const
  {
    if (k>this->DataDD.nsub()){
        std::cerr<<"k not valid"<<std::endl;
        return {1,1};
    }
    if (this->localA_created_==0) {
        std::cerr << "local A not created" << std::endl;
        return {1, 1};
    }
    else{
        k = (this->local_numbering) ? k-this->rank_group_la_*this->R_.size() : k;
        return this->localA_[k-1];
    }
  };

  // getters
  auto localA_created() const {return localA_created_;};
  auto rank() const {return current_rank_;};
  auto sub_assignment() const {return sub_assignment_;};



};

#endif
