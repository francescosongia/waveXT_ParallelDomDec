#ifndef SUB_ASSIGNMENT_HPP_
#define SUB_ASSIGNMENT_HPP_

#include "decomposition.hpp"
#include <iostream>

template<class LA>
class SubAssignment {

public:
  SubAssignment(int nproc, unsigned int nsubx,unsigned int nsubt)
      : np_(nproc), nsub_x_(nsubx), nsub_t_(nsubt), sub_division_(nproc,nsubt), sub_division_vec_(nproc)
      {};  

  // AGGIUNGERE FRAMWORK IN CUI NEL COSTRUTTORE PASSO MATRIX SUBDIVISION CUSTOM. LIMITARE QUESTA SCELTA
  // SOLO SE SI USA SEQLA 
  // se si fa evitare di fare localnumbering in local_matrices perchè non sono piu ordinate (aggiungere qui 
  // una flag che dica se uso divisione custom, in localmatrices/createRmatrices leggo questa flag e se si non
  // faccio localnumbering)

      //per ora assumiamo solo parallelizzazione in tempo, poi si puo fare i casi con strategie diverse
      // comunque si tratta solo di fornire una strategia per dividere i sub tra i vari core, potrei creare
      // costruttore che legge una matrice grande come domain diviso e ogni elemnto è un numero che identifica core

      // con l'aggiunta di sub_division_Vec perde di importanza la matrice sub_division (che potrei pensare solo come 
      // argomento per una versione di un costrcuttore). meglio salvarsi cosi un vector per ogni processore


protected:
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

  void createSubDivision()
  {
    LA func_wrapper(this->np_,this->nsub_x_,this->nsub_t_);
    return func_wrapper.createSubDivision();
  };

};

// SEQUENTIAL LINEAR ALGEBRA 
class SeqLA : public SubAssignment<SeqLA>
{
  public:
    SeqLA(int nproc, unsigned int nsubx,unsigned int nsubt):
     SubAssignment<SeqLA>(nproc,nsubx,nsubt)
     {
        this->sub_division_.resize(nproc,(nsubx/nproc)*nsubt);
     };
    void createSubDivision()
    { 
      // PARALLEL (assign subs to different cores when computing the sum in precondAction)
      // assign each time stride to a single process.
      if (this->np_ > 1 && this->nsub_x_% this->np_ ==0 ){  
        int partition{0};
        partition = this->nsub_x_/ this->np_;
        for(int i=0; i< this->np_; ++i){
          auto temp = Eigen::VectorXi::LinSpaced(this->nsub_t_*partition, this->nsub_t_*partition*i + 1, this->nsub_t_*partition*(i+1)) ;
          this->sub_division_vec_[i] = temp;
          this->sub_division_(i,Eigen::seq(0,this->nsub_t_*partition-1)) = temp;
        }
      }
      // SEQUENTIAL (all subs assigned to the same process)
      else if(this->np_ == 1){
        this->sub_division_.setZero();
        auto temp = Eigen::VectorXi::LinSpaced(this->nsub_t_*this->nsub_x_, 1, this->nsub_t_*this->nsub_x_) ;
        this->sub_division_vec_[0]=temp;
        //this->sub_division_vec_.push_back(temp);
        
      }
      else{
        std::cerr<<"error in subs division among processes"<<std::endl;
      }

    };
};


// PARALLEL LINEAR ALGEBRA
class ParLA : public SubAssignment<ParLA>
{
  public:
    ParLA(int nproc, unsigned int nsubx,unsigned int nsubt): SubAssignment<ParLA>(nproc,nsubx,nsubt) 
    {};

    void createSubDivision()
    {
      // n.° process > nsubx. Assign to each time stride at least two processses. In thi way the linear 
      // algebra computation are parallelized between them. 

      int partition{0};
      partition = this->np_ / this->nsub_x_ ;  //how many process are assigned to a single time stride
      int i_group_la{0};

      if (this->np_% this->nsub_x_ == 0){  
        for(int i=0; i< this->np_; ++i){
          i_group_la = i % partition;
          auto temp = Eigen::VectorXi::LinSpaced(this->nsub_t_, this->nsub_t_*i_group_la + 1, this->nsub_t_*(i_group_la+1)) ;
          this->sub_division_vec_[i] = temp;
          this->sub_division_(i,Eigen::seq(0,this->nsub_t_-1)) = temp;
        }
      }
      else{
        std::cerr<<"error in subs division among processes"<<std::endl;
      }
    };


};




#endif
