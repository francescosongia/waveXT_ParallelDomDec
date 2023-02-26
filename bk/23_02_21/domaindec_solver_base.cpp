#include "domaindec_solver_base.hpp"
#include <iostream>

std::pair<SpMat, SpMat> DomainDecSolverBase::createRK(unsigned int k) {
    unsigned int start_elem,ox_forw,ot_forw,ox_back,ot_back;
    std::tie(start_elem,ox_forw,ot_forw,ox_back,ot_back) = DataDD.get_info_subK(k);
    unsigned int m,n,nt,nx,nln;
    double theta(DataDD.theta());
    m = DataDD.sub_sizes()[1];
    n = DataDD.sub_sizes()[0];
    nt = domain.nt();
    nx = domain.nx();
    nln = domain.nln();
    theta = DataDD.theta();

    Eigen::ArrayXi indexcol = Eigen::ArrayXi::Zero(nln*m*n);
    Eigen::ArrayXd values = Eigen::ArrayXd::Zero(nln*m*n);

    for (size_t q=0;q<n-1;++q) {
        auto temp = Eigen::VectorXi::LinSpaced(nln * m, q * nt * nln + 1 + (start_elem - 1) * nln,
                                               (q * nt + m) * nln + (start_elem - 1) * nln);
        indexcol(Eigen::seq(q * nln * m, nln * m * (q + 1)-1)) << temp;
    }

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
    std::vector<T> tripletList;
    tripletList.reserve(nln*m*n);
    std::vector<T> tripletList_tilde;
    tripletList_tilde.reserve(nln*m*n);
    for(size_t i=0;i<nln*m*n;++i){
        tripletList.emplace_back(i,indexcol[i],1);
        tripletList_tilde.emplace_back(i,indexcol[i],values[i]);
    }
    SpMat Rk(nln*m*n,nln*nt*nx),Rk_tilde(nln*m*n,nln*nt*nx);
    Rk.setFromTriplets(tripletList.begin(), tripletList.end());
    Rk_tilde.setFromTriplets(tripletList_tilde.begin(), tripletList_tilde.end());  //already compressed
    return std::make_pair(Rk, Rk_tilde);


}

void DomainDecSolverBase::createRMatrices() {
    for(unsigned int k=1;k<DataDD.nsub()+1;++k){
        std::pair<SpMat, SpMat> res= createRK(k);
        //considerare anche w
        R_[k-1]=res.first;
        R_tilde_[k-1]=res.second;
    }
}

std::pair<SpMat, SpMat> DomainDecSolverBase::getRk(unsigned int k) {
    return std::make_pair(R_[k-1], R_tilde_[k-1]);
}

void DomainDecSolverBase::createAlocal(SpMat A) {
    for(unsigned int k=1;k<DataDD.nsub()+1;++k){
        //Ak=Rk*A*transpose(Rk);
        localA_[k-1]=R_[k-1]*A*R_[k-1].transpose();
    }

}
