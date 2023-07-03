#include "solver_results.hpp"

#include <vector>
#include <iostream>
#include <fstream>

  void SolverResults::formatGNU(const std::string& coord_file, unsigned int n_elem, unsigned int nln) const{
    std::ifstream coord(coord_file);
    
    auto u_resh = this->getU().reshaped(nln, n_elem);
    auto w_resh = this->getW().reshaped(nln, n_elem);
    Eigen::MatrixXd u_mean = u_resh.colwise().sum();
    Eigen::MatrixXd w_mean = w_resh.colwise().sum();
    u_mean /= static_cast<double>(nln);
    w_mean /= static_cast<double>(nln);

    std::ofstream u_output{"u_gnuplot.txt"};
    std::ofstream w_output{"w_gnuplot.txt"};

    if(coord.is_open() && u_output.is_open()&& w_output.is_open()){
      std::string line,line_u,line_w;
      int count = 0;
      while (std::getline(coord,line)){
        line_u = line + ","+std::to_string(u_mean(0,count));
        line_w = line + ","+std::to_string(w_mean(0,count)); 
        count+=1;
        u_output<<line_u<<std::endl;
        w_output<<line_w<<std::endl;
      }
      coord.close();
      u_output.close();
      w_output.close();
    }
    else{
      std::cout<<"problems with opening files"<<std::endl;
    }


}