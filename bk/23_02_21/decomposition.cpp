#include "decomposition.hpp"
#include <cmath>
#include <vector>
#include <iostream>

void Decomposition::createDec(double n = -1, double m = -1) {
  unsigned int nx = domain.nx();
  unsigned int nt = domain.nt();
  if (n * nsub_x_ <= nx ||
      m * nsub_t_ <= nt) { // controllare caso in cui m e n settati in
                          // modo che formino sub di solo overlap
    double molt = 1.2;
    n = floor(nx * molt / nsub_x_);
    m = floor(nt * molt / nsub_t_);
  }
  sub_sizes_[0] = n;
  sub_sizes_[1] = m;
  // ora creo liste overlap e elem inizio

  // average ot,ox, rest
  double ot_mean , ox_mean;
  int ot_rest, ox_rest;
  double val=m - (nt - m) / (nsub_t_ - 1);
  ot_mean = (val>=0)? floor(val) : ceil(val);
  ot_rest = (m - ot_mean) * (nsub_t_ - 1) - (nt - m);
  val=n - (nx - n)/ (nsub_x_ - 1);
  ox_mean = (val>=0)? floor(val) : ceil(val);
  ox_rest = (n - ox_mean) * (nsub_x_ - 1) - (nx - n);

  // ox
  Eigen::MatrixXi a1 = Eigen::MatrixXi::Constant(ox_rest, nsub_t_, ox_mean + 1);
  Eigen::MatrixXi b1 =
      Eigen::MatrixXi::Constant(nsub_x_ - ox_rest - 1, nsub_t_, ox_mean);
  Eigen::MatrixXi c1 = Eigen::MatrixXi::Constant(1, nsub_t_, 0);
  Eigen::MatrixXi to_resh(a1.rows() + b1.rows() + c1.rows(), a1.cols());
  to_resh << a1, b1, c1;
  to_resh.transposeInPlace();
  Eigen::MatrixXi to_resh1(a1.rows() + b1.rows() + c1.rows(), a1.cols());
  to_resh1 << c1, a1, b1;
  to_resh1.transposeInPlace();
  overlap_forw_.row(0) = to_resh.reshaped(1, nsub_x_ * nsub_t_);
  overlap_back_.row(0) = to_resh1.reshaped(1, nsub_x_ * nsub_t_);

  // ot
  std::vector<double> a(ot_rest, ot_mean + 1);
  std::vector<double> b(nsub_t_ - ot_rest - 1, ot_mean);
  std::vector<double> to_rep, to_rep1;
  to_rep.insert(to_rep.end(), a.begin(), a.end());
  to_rep.insert(to_rep.end(), b.begin(), b.end());
  to_rep.push_back(0);
  to_rep1.push_back(0);
  to_rep1.insert(to_rep1.end(), a.begin(), a.end());
  to_rep1.insert(to_rep1.end(), b.begin(), b.end());
  std::vector<unsigned int> ot_forw;
  std::vector<unsigned int> ot_back;
  for (unsigned int i = 0; i < nsub_x_; ++i) {
    ot_forw.insert(ot_forw.end(), to_rep.begin(), to_rep.end());
    ot_back.insert(ot_back.end(), to_rep1.begin(), to_rep1.end());
  }
  for (size_t i = 0; i < ot_forw.size(); ++i) {
    overlap_back_(1, i) = ot_back[i];
    overlap_forw_(1, i) = ot_forw[i];
  }

  // start_elem
  Eigen::MatrixXi ot_forw_mat;
  ot_forw_mat = overlap_forw_.row(1).reshaped(nsub_t_, nsub_x_).transpose();
  Eigen::MatrixXi ox_forw_mat;
  ox_forw_mat = overlap_forw_.row(0).reshaped(nsub_t_, nsub_x_).transpose();
  Eigen::MatrixXi start_elem_eig = Eigen::MatrixXi::Constant(nsub_x_, nsub_t_, 0);
  for (Eigen::Index i = 0; i < nsub_x_; ++i) {
      if (i==0){
          start_elem_eig(i, 0)=1;
      }
      else {
          Eigen::Index k=nsub_t_-1;
          start_elem_eig(i, 0) = start_elem_eig(i - 1, 0) +(n-ox_forw_mat(i-1, k)) * domain.nt();
      }

      for (Eigen::Index j = 1; j < nsub_t_; ++j) {
          start_elem_eig(i, j) =
              start_elem_eig(i, j-1 ) + m - ot_forw_mat(i, j - 1);
      }
  }

  //start_elem_eig = start_elem_eig.transpose().reshaped(1, nsub_t_ * nsub_x_);
  auto temp= start_elem_eig.transpose().reshaped();//1, nsub_t_ * nsub_x_);
  for (size_t i = 0; i < nsub_t_ * nsub_x_; ++i) {
    start_elem_[i] = temp(i, 0);
  }
}

std::tuple<unsigned int, unsigned int, unsigned int,unsigned int,unsigned int> Decomposition::get_info_subK(unsigned int k) {
    //startelem, ox ot forw,back
    if (k>this->nsub()){
        std::cerr<<"k is not a valid subdomain number"<<std::endl;
        return std::make_tuple(0,0,0,0,0);
    }
    k=k-1;
    return std::make_tuple(start_elem_[k], overlap_forw_(0,k), overlap_forw_(1,k),overlap_back_(0,k),overlap_back_(1,k));
}


/*
  // ot
  std::vector<double> a(ot_rest, ot_mean + 1);
  std::vector<double> b(nsub_t - ot_rest - 1, ot_mean);
  std::vector<double> to_rep, to_rep1;
  to_rep.insert(to_rep.end(), a.begin(), a.end());
  to_rep.insert(to_rep.end(), b.begin(), b.end());
  to_rep.push_back(0);
  to_rep1.push_back(0);
  to_rep1.insert(to_rep1.end(), a.begin(), a.end());
  to_rep1.insert(to_rep1.end(), b.begin(), b.end());
  for (unsigned int i = 0; i < nsub_x; ++i) {
    ot_forw.insert(ot_forw.end(), to_rep.begin(), to_rep.end());
    ot_back.insert(ot_back.end(), to_rep1.begin(), to_rep1.end());
  }

  // ox
  Eigen::MatrixXi a1 = Eigen::MatrixXi::Constant(ox_rest, nsub_t, ox_mean + 1);
  Eigen::MatrixXi b1 =
      Eigen::MatrixXi::Constant(nsub_x - ox_rest - 1, nsub_t, ox_mean);
  Eigen::MatrixXi c1 = Eigen::MatrixXi::Constant(1, nsub_t, 0);
  Eigen::MatrixXi to_resh(a1.rows() + b1.rows() + c1.rows(), a1.cols());
  to_resh << a1, b1, c1;
  to_resh.transposeInPlace();
  Eigen::MatrixXi to_resh1(a1.rows() + b1.rows() + c1.rows(), a1.cols());
  to_resh1 << c1, a1, b1;
  to_resh1.transposeInPlace();
  Eigen::MatrixXi ox_forw_eig = to_resh.reshaped(1, nsub_x * nsub_t);
  Eigen::MatrixXi ox_back_eig = to_resh1.reshaped(1, nsub_x * nsub_t);
  std::vector<unsigned int> temp(ox_forw_eig.data(),
                                 ox_forw_eig.data() +
                                     ox_forw_eig.rows() * ox_forw_eig.cols());
  std::vector<unsigned int> temp1(ox_back_eig.data(),
                                  ox_back_eig.data() +
                                      ox_back_eig.rows() * ox_back_eig.cols());
  ox_forw = temp;
  ox_back = temp1;

  // start_elem
  Eigen::MatrixXi ot_forw_eig(1, t_forw.size());
  for (size_t i = 0; i < ot_forw.size(); ++i) {
    ot_forw_eig(i) = ot_forw[i];
  }
  Eigen::MatrixXi ot_forw_mat =
      ot_forw_eig.reshaped(nsub_t, nsub_x).transpose();
  Eigen::MatrixXi ox_forw_mat =
      ox_forw_eig.reshaped(nsub_t, nsub_x).transpose();

  Eigen::MatrixXi start_elem_eig = Eigen::MatrixXi::Constant(nsub_x, nsub_t, 0);
*/
// row58

