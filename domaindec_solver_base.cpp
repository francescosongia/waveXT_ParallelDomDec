#include "domaindec_solver_base.hpp"
#include <iostream>
#include <fstream>

std::pair<SpMat, SpMat> DomainDecSolverBase::createRK(unsigned int k) {
    unsigned int start_elem,ox_forw,ot_forw,ox_back,ot_back;
    std::tie(start_elem,ox_forw,ot_forw,ox_back,ot_back) = DataDD.get_info_subK(k);
    unsigned int m,n,nt,nx,nln;
    double theta;
    m = DataDD.sub_sizes()[1];
    n = DataDD.sub_sizes()[0];
    nt = domain.nt();
    nx = domain.nx();
    nln = domain.nln();
    theta = DataDD.theta();

    Eigen::ArrayXi indexcol = Eigen::ArrayXi::Zero(nln*m*n*2);
    Eigen::ArrayXd values = Eigen::ArrayXd::Zero(nln*m*n*2);

    for (size_t q=0;q<=n-1;++q) {
        auto temp = Eigen::VectorXi::LinSpaced(nln * m, q * nt * nln + (start_elem - 1) * nln,
                                               (q * nt + m) * nln + (start_elem - 1) * nln-1);
        indexcol(Eigen::seq(q * nln * m, nln * m * (q + 1)-1)) << temp;
    }
    indexcol(Eigen::seq(nln*m*n,nln*m*n*2 - 1))=indexcol(Eigen::seq(0,nln*m*n-1))+(nt*nx*nln); //extend for w

    auto temp1 = Eigen::ArrayXd::Constant(ot_back*nln,(1-theta)/2);
    auto temp2 = Eigen::ArrayXd::Constant((m-ot_forw-ot_back)*nln,0.5);
    auto temp3 = Eigen::ArrayXd::Constant(ot_forw*nln,theta/2);
    for(size_t q=0;q<ox_back;++q){
        values(Eigen::seq(q*nln*m,nln*m*(q+1)-1))<<temp1,temp2,temp3;
    }
    for(size_t q=ox_back;q<n-ox_forw;++q){
        values(Eigen::seq(q*nln*m,nln*m*(q+1)-1))<<temp1*2,temp2*2,temp3*2;
    }
    for(size_t q=n-ox_forw;q<n;++q){
        values(Eigen::seq(q*nln*m,nln*m*(q+1)-1))<<temp1,temp2,temp3;
    }
    values(Eigen::seq(nln*m*n,nln*m*n*2 - 1))=values(Eigen::seq(0,nln*m*n-1)); //extend for w

    std::vector<T> tripletList;
    tripletList.reserve(nln*m*n*2);
    std::vector<T> tripletList_tilde;
    tripletList_tilde.reserve(nln*m*n*2);
    for(size_t i=0;i<nln*m*n*2;++i){
        tripletList.emplace_back(i,indexcol[i],1.);
        tripletList_tilde.emplace_back(i,indexcol[i],values[i]);
    }
    SpMat Rk(nln*m*n*2,nln*nt*nx*2),Rk_tilde(nln*m*n*2,nln*nt*nx*2);
    Rk.setFromTriplets(tripletList.begin(), tripletList.end());
    Rk_tilde.setFromTriplets(tripletList_tilde.begin(), tripletList_tilde.end());  //already compressed

    return std::make_pair(Rk, Rk_tilde);


}

void DomainDecSolverBase::createRMatrices() {
    for(unsigned int k=1;k<DataDD.nsub()+1;++k){
        std::pair<SpMat, SpMat> res= createRK(k);
        R_[k-1]=res.first;
        R_tilde_[k-1]=res.second;
    }
}

std::pair<SpMat, SpMat> DomainDecSolverBase::getRk(unsigned int k) const {
    return std::make_pair(R_[k-1], R_tilde_[k-1]);
}

SpMat DomainDecSolverBase::getAk(unsigned int k) const{
    if (k>DataDD.nsub()){
        std::cerr<<"k not valid"<<std::endl;
        return {1,1};
    }
    if (localA_created==0) {
        std::cerr << "local A not created" << std::endl;
        return {1, 1};
    }
    else
        return localA_[k-1];
}

void DomainDecSolverBase::createAlocal(const SpMat& A) {
    unsigned int m,n,nt,nx,nln;
    nt = domain.nt();
    nx = domain.nx();
    nln = domain.nln();
    m = DataDD.sub_sizes()[1];
    n = DataDD.sub_sizes()[0];
    SpMat temp(nln*m*n*2,nln*nt*nx*2);
    for(unsigned int k=1;k<DataDD.nsub()+1;++k){
        //Rk(nln*m*n*2,nln*nt*nx*2),
        //Ak=Rk*A*transpose(Rk);
        temp=R_[k-1]*A;
        localA_[k-1]=temp*(R_[k-1].transpose());
    }
    localA_created=1;
}
