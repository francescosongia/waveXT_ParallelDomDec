#include "sub_assignment.hpp"
#include <iostream>


void SubAssignment::createSubDivision() {
  // np vs subx
  if (np_ == nsub_x_){
    for(int i=0; i< np_; ++i){
      auto temp = Eigen::VectorXi::LinSpaced(nsub_t_, nsub_t_*i + 1, nsub_t_*(i+1)) ;
      sub_division_(i,Eigen::seq(0,nsub_t_-1)) = temp;
    }
  }
  else if(np_ == 0){
    sub_division_.setZero();
  }
  else{
    std::cerr<<"error in subs division among processes"<<std::endl;
  }

  // poi aggiungere altri casi, tipo 
  // - se ho meno processori di nsubx 
  // - se nsubx non è divisibile per np, allora assegno i mancanti in qualche modo (tutti all'ultimo, oppure iterando e assegnando un po a tutti)
  // - se np è maggiore di nsubx, messaggio che dice che sarebbe meglio usare parallelizzazione con aiutanti (intra)
 
}