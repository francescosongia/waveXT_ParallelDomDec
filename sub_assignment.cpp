/*
#include "sub_assignment.hpp"
#include <iostream>

template<class LA>
void SubAssignment<LA>::createSubDivision() {
  // np vs subx

  
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

  // poi aggiungere altri casi, tipo 
  // - se ho meno processori di nsubx 
  // - se nsubx non è divisibile per np, allora assegno i mancanti in qualche modo (tutti all'ultimo, oppure iterando e assegnando un po a tutti)
  // - se np è maggiore di nsubx, messaggio che dice che sarebbe meglio usare parallelizzazione con aiutanti (intra)
 
}
*/ 
