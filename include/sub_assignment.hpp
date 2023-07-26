#ifndef SUB_ASSIGNMENT_HPP_
#define SUB_ASSIGNMENT_HPP_

#include "decomposition.hpp"
#include <iostream>

template<class LA>
class SubAssignment {

public:
  SubAssignment(int nproc, unsigned int nsubx,unsigned int nsubt)
      : np_(nproc), nsub_x_(nsubx), nsub_t_(nsubt), sub_division_(nproc,nsubt), sub_division_vec_(nproc),custom_matrix_(false)
      {};  

  
  //custom solo per seqla   
  SubAssignment(int nproc, unsigned int nsubx,unsigned int nsubt, const Eigen::MatrixXi& sub_division)
      : np_(nproc), nsub_x_(nsubx), nsub_t_(nsubt), sub_division_(sub_division), sub_division_vec_(nproc),custom_matrix_(true)
      {
        std::vector<std::vector<int>> temp(nproc);
        int current_proc{0};
        for(Eigen::Index i = 0; i < nsub_x_; ++i){
          for(Eigen::Index j = 0; j < nsub_t_; ++j){
            current_proc = sub_division(i,j);
            if(current_proc>=nproc)
              std::cerr<<"element of custom matrix greater than size-1"<<std::endl;
            temp[current_proc].push_back(i*nsubt+(j+1));
          }
        }
        for(int i=0;i<nproc;++i){
          Eigen::VectorXi eigenVector = Eigen::Map<Eigen::VectorXi>(temp[i].data(), temp[i].size());
          sub_division_vec_[i] = eigenVector;
        }
      };  
    

 
protected:
  int np_;
  int nsub_x_;
  int nsub_t_;

  Eigen::MatrixXi sub_division_;
  std::vector<Eigen::VectorXi> sub_division_vec_;

  bool custom_matrix_;

public:
  auto np() const { return np_; };
  auto nsub_x() const { return nsub_x_; };
  auto nsub_t() const { return nsub_t_; };
  auto sub_division() const { return sub_division_; };
  auto sub_division_vec() const { return sub_division_vec_; };
  auto custom_matrix() const { return custom_matrix_; };

  virtual void createSubDivision() = 0;

};

// SEQUENTIAL LINEAR ALGEBRA.
// it does not mean that all the solver is sequential, it could be parallelized by assigning the subdomains to different processes
class SeqLA : public SubAssignment<SeqLA>
{
  public:
    SeqLA(int nproc, unsigned int nsubx,unsigned int nsubt):
     SubAssignment<SeqLA>(nproc,nsubx,nsubt)
     {
        this->sub_division_.resize(nproc,(nsubx/nproc)*nsubt);
     };

    SeqLA(int nproc, unsigned int nsubx,unsigned int nsubt,const Eigen::MatrixXi& sub_division):
     SubAssignment<SeqLA>(nproc,nsubx,nsubt,sub_division)
     {};


    void createSubDivision() override
    { 
      if(this->custom_matrix_ == false){
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
        else
          std::cerr<<"error in subs assignment among processes"<<std::endl;
      }
      else
        std::cout<<"sub assignment is done with custom matrix"<<std::endl;

    };
};


// PARALLEL LINEAR ALGEBRA
class ParLA : public SubAssignment<ParLA>
{
  public:
    ParLA(int nproc, unsigned int nsubx,unsigned int nsubt): SubAssignment<ParLA>(nproc,nsubx,nsubt) 
    {};
    
    ParLA(int nproc, unsigned int nsubx,unsigned int nsubt,const Eigen::MatrixXi& sub_division): 
      SubAssignment<ParLA>(nproc,nsubx,nsubt,sub_division) 
    {
      std::cerr<<"custom sub assignment is not possible with ParLA"<<std::endl;
    };
    

    void createSubDivision() override
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

class SplitTime : public SubAssignment<SplitTime>
{
  public:
    SplitTime(int nproc, unsigned int nsubx,unsigned int nsubt): SubAssignment<SplitTime>(nproc,nsubx,nsubt) 
    {
      sub_division_.resize(nproc, nsubx*nsubt/nproc);
    };
    
    SplitTime(int nproc, unsigned int nsubx,unsigned int nsubt,const Eigen::MatrixXi& sub_division): 
      SubAssignment<SplitTime>(nproc,nsubx,nsubt,sub_division) 
    {
      std::cerr<<"custom sub assignment is not possible with SplitTime"<<std::endl;
    };
    

    void createSubDivision() override
    {
      // 4 proc, 2 subx, 
      
      int partition{0};
      partition = this->np_ / this->nsub_x_ ;  //how many process are assigned to a single time stride
      int time_partition{0};
      time_partition = this->nsub_t_ / partition ; // ASSERT CHE SIANO DIVISIBILI

      if ( (this->nsub_x_*this->nsub_t_)%this->np_ == 0){  
        for(int i=0; i< this->np_; ++i){
          auto temp = Eigen::VectorXi::LinSpaced(time_partition, time_partition*i+1, time_partition*(i+1)+1 ) ;
          this->sub_division_vec_[i] = temp;
          this->sub_division_(i,Eigen::seq(0,time_partition-1)) = temp;
        }
      }
      else{
        std::cerr<<"error in subs division among processes"<<std::endl;
      }
    };
 
};




#endif
