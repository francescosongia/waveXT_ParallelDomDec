#include "solver_results.hpp"

#include <vector>
#include <iostream>
#include <fstream>

  void SolverResults::formatGNU(int uw_flag, const std::string& coord_file, unsigned int n_elem, unsigned int nln) const{
    std::ifstream coord(coord_file);
    Eigen::MatrixXd u_resh;
    if(uw_flag==0)
      u_resh = this->getU().reshaped(nln, n_elem);
    else if(uw_flag==1)
      u_resh = this->getW().reshaped(nln, n_elem);
    else
      std::cerr<<"error in uw flag, provide 0 or 1"<<std::endl;
    
    Eigen::MatrixXd u_mean = u_resh.colwise().sum();
    u_mean /= static_cast<double>(nln);

    std::ofstream output;
    if(uw_flag==0)
      output.open("u_gnuplot.txt");
    else
      output.open("w_gnuplot.txt");

    if(coord.is_open() && output.is_open()){
    
      std::string line,line_u;
      int count = 0;
      while (std::getline(coord,line)){
        line_u = line + ","+std::to_string(u_mean(0,count));
        count+=1;
        output<<line_u<<std::endl;
      }
      coord.close();
      output.close();
    }
    else{
      std::cout<<"problems with opening files"<<std::endl;
    }


}