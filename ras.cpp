#include <iostream>
#include "ras.hpp"
#include <chrono>

Eigen::VectorXd Ras::precondAction(const SpMat& x) {
    Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    //Eigen::VectorXd xk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    for(unsigned int k=1;k<=DataDD.nsub();++k){
        //uk=localA_[k-1]\xk;
/*
        Eigen::SimplicialCholesky<SpMat> chol(localA_[k-1]);  // performs a Cholesky factorization of A
        uk = chol.solve(R_[k-1]*x);

        Eigen::BiCGSTAB<SpMat > bic;
        bic.compute(localA_[k-1]);
        uk = bic.solve(R_[k-1]*x);
*/
        Eigen::SparseLU<SpMat > lu;
        lu.compute(localA_[k-1]);
        uk = lu.solve(R_[k-1]*x);

        z=z+(R_tilde_[k-1].transpose())*uk;
    }
    return z;
}

Eigen::VectorXd Ras::solve(const SpMat& A, const SpMat& b, SolverTraits traits) {
    auto start = std::chrono::steady_clock::now();
    double tol=traits.tol();
    unsigned int max_it=traits.max_it();
    double res=tol+1;
    std::vector<double> relres2P_vec;
    unsigned int niter=0;
    double Pb2=precondAction(b).norm();

    Eigen::VectorXd uw0=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    Eigen::VectorXd uw1(domain.nln()*domain.nt()*domain.nx()*2);

    Eigen::VectorXd z= precondAction(b); //b-A*uw0
    while(res>tol and niter<max_it){
        uw1=uw0+z;  //forse si puo evitare l'uso di entrambi uw0 e uw1
        z= precondAction(b-A*uw1);
        res=(z/Pb2).norm();
        relres2P_vec.push_back(res);
        niter++;
        uw0=uw1;
    }


    auto end = std::chrono::steady_clock::now();
    std::cout<<"niter: "<<niter<<std::endl;
    std::cout<<"solves: "<<niter*DataDD.nsub()<<std::endl;
    std::cout <<"time in seconds: "<< std::chrono::duration_cast<std::chrono::seconds>(end - start).count()<<std::endl;

    return uw1;
}