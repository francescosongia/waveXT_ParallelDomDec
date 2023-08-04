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

  
  //custom matrix to assign subdomains
  SubAssignment(int nproc, unsigned int nsubx,unsigned int nsubt, const Eigen::MatrixXi& sub_division)
      : np_(nproc), nsub_x_(nsubx), nsub_t_(nsubt), sub_division_(sub_division), sub_division_vec_(nproc),custom_matrix_(true)
      {
        std::vector<std::vector<int>> temp(nproc);
        int current_proc{0};
        for(Eigen::Index i = 0; i < nsub_x_; ++i)
        {
          for(Eigen::Index j = 0; j < nsub_t_; ++j)
          {
            current_proc = sub_division(i,j);
            if(current_proc>=nproc)
              std::cerr<<"element of custom matrix greater than size-1"<<std::endl;
            temp[current_proc].push_back(i*nsubt+(j+1));
          }
        }
        for(int i=0;i<nproc;++i)
        {
          Eigen::VectorXi eigenVector = Eigen::Map<Eigen::VectorXi>(temp[i].data(), temp[i].size());
          sub_division_vec_[i] = eigenVector;
        }
      };  
    

 
protected:
  int np_; //number of processes
  int nsub_x_;
  int nsub_t_;

  // structures that contain the subdomains numbers assigned to each of the processes
  Eigen::MatrixXi sub_division_;
  std::vector<Eigen::VectorXi> sub_division_vec_;

  // flag to underline if a custom matrix for subs assighment is provided
  bool custom_matrix_;

  
public:
  auto np() const { return np_; };
  auto nsub_x() const { return nsub_x_; };
  auto nsub_t() const { return nsub_t_; };
  auto sub_division() const { return sub_division_; };
  auto sub_division_vec() const { return sub_division_vec_; };
  auto custom_matrix() const { return custom_matrix_; };

  virtual void createSubDivision() = 0;

  
  unsigned int idxSub_to_LocalNumbering(unsigned int k,int current_rank) const  
  // returns the index in the local matrices storage structures correpsonding to subdomain k.
  // for the policies presented here this function is the same so it is declared here. 
  // for future policies consider to overload it if a different behaviour is needed
  {
    auto sub_division_vec = this->sub_division_vec_[current_rank];
    auto local_k = k-sub_division_vec(0)+1;
    return local_k;
  }; 
  

};

/*
   SUB ASSIGNMENT POLICIES
- AloneOnStride
    On each temporal stride only one core is working. It is suited for the fully sequential version or for the parallel one 
    in which the linear algebra is maintened sequential and each temporal stride is assigned to a different core

- CooperationOnStride         
    On each temporal stride more cores are working. It is suited for the parallel version where the linear algebra is parallelized
    between the cores working on the same stride. 
    This policy is used also for RasPipelined in a SplitTime fashion: more cores work on the same stride but they cooperated dividing
    the corresponding time subdomains between them. 

- CooperationCooperationSplitTime        
    On each temporal stride more cores are working and they further divide in the time direction the subdomains assigned. It can be seen
    as a complete subdivision among space and time. It is suited for RAS method.

*/




// SEQUENTIAL LINEAR ALGEBRA.
// it does not mean that all the solver is sequential, it could be parallelized by assigning the subdomains to different processes
class AloneOnStride : public SubAssignment<AloneOnStride>
{
  public:
    AloneOnStride(int nproc, unsigned int nsubx,unsigned int nsubt):
     SubAssignment<AloneOnStride>(nproc,nsubx,nsubt)
     {
        this->sub_division_.resize(nproc,(nsubx/nproc)*nsubt);
     };

    AloneOnStride(int nproc, unsigned int nsubx,unsigned int nsubt,const Eigen::MatrixXi& sub_division):
     SubAssignment<AloneOnStride>(nproc,nsubx,nsubt,sub_division)
     {};

    void createSubDivision() override
    { 
      if(this->custom_matrix_ == false){
        // PARALLEL (assign subs to different cores when computing the sum in precondAction)
        // assign each time stride to a single process.
        if (this->np_ > 1 && this->nsub_x_% this->np_ ==0 )
        {  
          int partition{0};
          partition = this->nsub_x_/ this->np_;
          for(int i=0; i< this->np_; ++i)
          {
            auto temp = Eigen::VectorXi::LinSpaced(this->nsub_t_*partition, this->nsub_t_*partition*i + 1, this->nsub_t_*partition*(i+1)) ;
            this->sub_division_vec_[i] = temp;
            this->sub_division_(i,Eigen::seq(0,this->nsub_t_*partition-1)) = temp;
          }
        }
        // SEQUENTIAL (all subs assigned to the same process)
        else if(this->np_ == 1)
        {
          this->sub_division_.setZero();
          auto temp = Eigen::VectorXi::LinSpaced(this->nsub_t_*this->nsub_x_, 1, this->nsub_t_*this->nsub_x_) ;
          this->sub_division_vec_[0]=temp;          
        }
        else
          std::cerr<<"error in subs assignment among processes"<<std::endl;
      }
      else
        std::cout<<"sub assignment is done with custom matrix"<<std::endl;

    };
};


// PARALLEL LINEAR ALGEBRA
class CooperationOnStride : public SubAssignment<CooperationOnStride>
{
  public:
    CooperationOnStride(int nproc, unsigned int nsubx,unsigned int nsubt): SubAssignment<CooperationOnStride>(nproc,nsubx,nsubt) 
    {};
    
    CooperationOnStride(int nproc, unsigned int nsubx,unsigned int nsubt,const Eigen::MatrixXi& sub_division): 
      SubAssignment<CooperationOnStride>(nproc,nsubx,nsubt,sub_division) 
    {
      std::cerr<<"custom sub assignment is not possible with CooperationOnStride"<<std::endl;
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





class CooperationSplitTime : public SubAssignment<CooperationSplitTime>
{
  public:
    CooperationSplitTime(int nproc, unsigned int nsubx,unsigned int nsubt): SubAssignment<CooperationSplitTime>(nproc,nsubx,nsubt) 
    {
      sub_division_.resize(nproc, nsubx*nsubt/nproc);
    };
    
    CooperationSplitTime(int nproc, unsigned int nsubx,unsigned int nsubt,const Eigen::MatrixXi& sub_division): 
      SubAssignment<CooperationSplitTime>(nproc,nsubx,nsubt,sub_division) 
    {
      std::cerr<<"custom sub assignment is not possible with CooperationSplitTime"<<std::endl;
    };


    void createSubDivision() override
    {
      
      int partition{0};
      partition = this->np_ / this->nsub_x_ ;  //how many process are assigned to a single time stride
      int time_partition{0};
      time_partition = this->nsub_t_ / partition ; 


      Eigen::VectorXi list=Eigen::VectorXi::LinSpaced(this->np_,0,this->np_-1);  
      auto matrix_cores=list.reshaped(partition, this->np_ / partition);  
      auto vec_cores = matrix_cores.transpose().reshaped();
      if ( (this->nsub_x_*this->nsub_t_)%this->np_ == 0){  

        for(int i=0; i< this->np_; ++i){
          auto local_i =vec_cores(i);
          auto temp = Eigen::VectorXi::LinSpaced(time_partition, time_partition*i+1, time_partition*(i+1)+1 ) ;
          this->sub_division_vec_[local_i] = temp;
          this->sub_division_(local_i,Eigen::seq(0,time_partition-1)) = temp;
        }
      }
      else{
        std::cerr<<"error in subs division among processes"<<std::endl;
      }
    };
 
};




#endif
