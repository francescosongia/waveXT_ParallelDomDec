#ifndef SUB_ASSIGNMENT_HPP_
#define SUB_ASSIGNMENT_HPP_

#include "decomposition.hpp"
//#include "policyLA.hpp"
#include <iostream>

template<class LA>
class SubAssignment {

public:
/*
  SubAssignment()
      : np_(0), nsub_x_(0), nsub_t_(0), sub_division_(0,0), sub_division_vec_(0) {};
*/

  SubAssignment(int nproc, unsigned int nsubx,unsigned int nsubt)
      : np_(nproc), nsub_x_(nsubx), nsub_t_(nsubt), sub_division_(nproc,nsubt), sub_division_vec_(nproc)
      {};   //IF GRANDE SE NON SONO DIVISIBILI DI BLOCCARSI PRIMA

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


class SeqLA : public SubAssignment<SeqLA>
{
  public:
    SeqLA(int nproc, unsigned int nsubx,unsigned int nsubt):
     SubAssignment<SeqLA>(nproc,nsubx,nsubt)
     {
        this->sub_division_.resize(nproc,(nsubx/nproc)*nsubt);
     };
    void createSubDivision()
    { //4subx 2 np
      int partition{0};
      partition = this->nsub_x_/ this->np_;

      if (this->nsub_x_% this->np_ ==0){  //this->np_ == this->nsub_x_
        for(int i=0; i< this->np_; ++i){
          auto temp = Eigen::VectorXi::LinSpaced(this->nsub_t_*partition, this->nsub_t_*partition*i + 1, this->nsub_t_*partition*(i+1)) ;
          this->sub_division_vec_[i] = temp;
          this->sub_division_(i,Eigen::seq(0,this->nsub_t_*partition-1)) = temp;
        }
      }
      else if(this->np_ == 0){
        this->sub_division_.setZero();
        auto temp = Eigen::VectorXi::LinSpaced(this->nsub_t_*this->nsub_x_, 1, this->nsub_t_*this->nsub_x_) ;
        this->sub_division_vec_.push_back(temp);
        
      }
      else{
        std::cerr<<"error in subs division among processes_ sub_ass"<<std::endl;
      }


    };
};




// 2proc  2subx   -> SeqLA

class ParLA : public SubAssignment<ParLA>
{
  public:
    ParLA(int nproc, unsigned int nsubx,unsigned int nsubt): SubAssignment<ParLA>(nproc,nsubx,nsubt) 
    {};

    void createSubDivision()
    {
      // 4 process e 2 subx. quindi due proc sono i principali e hanno due aiutanti
      int partition{0};
      partition = this->np_ / this->nsub_x_ ;  //quanti p sono assegnati a striscia temp
      int i_group_la{0};

      if (this->np_% this->nsub_x_ == 0){  
        for(int i=0; i< this->np_; ++i){
          i_group_la = i % partition;
          auto temp = Eigen::VectorXi::LinSpaced(this->nsub_t_, this->nsub_t_*i_group_la + 1, this->nsub_t_*(i_group_la+1)) ;
          this->sub_division_vec_[i] = temp;
          this->sub_division_(i,Eigen::seq(0,this->nsub_t_-1)) = temp;
        }
        std::cout<<"matrix subdivision"<<std::endl;
        std::cout<<sub_division_<<std::endl;
      }
      else{
        std::cerr<<"error in subs division among processes"<<std::endl;
      }
    };


};




#endif
