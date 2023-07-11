#ifndef POLICY_LA_HPP_
#define POLICY_LA_HPP_

#include<iostream>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double>
        SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class SeqLA
{
  public:

    void createSubDivision(int np_, int nsub_x_,int nsub_t_, Eigen::MatrixXi sub_division_, std::vector<Eigen::VectorXi> sub_division_vec_)
    {
      if (np_ == nsub_x_){  //metterci divisibile per nsubx
        for(int i=0; i< np_; ++i){
          auto temp = Eigen::VectorXi::LinSpaced(nsub_t_, nsub_t_*i + 1, nsub_t_*(i+1)) ;
          sub_division_vec_[i] = temp;
          sub_division_(i,Eigen::seq(0,nsub_t_-1)) = temp;
        }
      }
      else if(np_ == 0){
        sub_division_.setZero();
        auto temp = Eigen::VectorXi::LinSpaced(nsub_t_*nsub_x_, 1, nsub_t_*nsub_x_) ;
        sub_division_vec_.push_back(temp);
        
      }
      else{
        std::cerr<<"error in subs division among processes"<<std::endl;
      }


        };
};


class ParLA {
  public:
    void createSubDivision(int np_, int nsub_x_,int nsub_t_, Eigen::MatrixXi sub_division_, std::vector<Eigen::VectorXi> sub_division_vec_){
      std::cout<<"da fare"<<std::endl;
    };
};


#endif
