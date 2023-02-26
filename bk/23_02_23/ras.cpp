#include <iostream>
#include "ras.hpp"
Eigen::VectorXd Ras::precondAction(const SpMat& x) {
    Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    //Eigen::VectorXd xk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    for(unsigned int k=1;k<=DataDD.nsub();++k){
        /*
        if(k==1) {
            Eigen::VectorXd xk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
            xk = R_[k - 1] * x;
            std::cout << xk(Eigen::seq(0, 10)) << std::endl;
            std::cout<<std::endl;
        }
         */

        //uk=localA_[k-1]\xk;

        Eigen::SimplicialCholesky<SpMat> chol(localA_[k-1]);  // performs a Cholesky factorization of A
        uk = chol.solve(R_[k-1]*x);
/*
        Eigen::BiCGSTAB<SpMat > bic;
        bic.compute(localA_[k-1]);
        uk = bic.solve(R_[k-1]*x);
        */

        z=z+(R_tilde_[k-1].transpose())*uk;
    }
    return z;
}

Eigen::VectorXd Ras::solve(const SpMat& A, const SpMat& b, unsigned int max_it, double tol) {
    double res=tol+1;
    std::vector<double> relres2P_vec;
    unsigned int niter=0;
    double Pb2=precondAction(b).norm();
    //std::cout<<"pb2: "<<Pb2<<std::endl;

    Eigen::VectorXd uw0=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    Eigen::VectorXd uw1(domain.nln()*domain.nt()*domain.nx()*2);

    Eigen::VectorXd z= precondAction(b); //b-A*uw0
    while(res>tol and niter<max_it){
        //std::cout<<"res: "<<res<<std::endl;
        uw1=uw0+z;
        z= precondAction(b-A*uw1);
        res=(z/Pb2).norm();
        //std::cout<<"res2: "<<res<<std::endl;
        relres2P_vec.push_back(res);
        niter++;
        uw0=uw1;
    }
    std::cout<<"niter: "<<niter<<std::endl;

    return uw1;


}