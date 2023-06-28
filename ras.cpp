#include <iostream>
#include "ras.hpp"
#include <chrono>
#include <mpi.h>


// SEQUENTIAL
/*
Eigen::VectorXd Ras::precondAction(const SpMat& x) {
    Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    //Eigen::VectorXd xk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    for(unsigned int k=1;k<=DataDD.nsub();++k){

        Eigen::SparseLU<SpMat > lu;
        lu.compute(local_mat.getAk(k));

        auto temp = local_mat.getRk(k);

        uk = lu.solve(temp.first*x);

        z=z+(temp.second.transpose())*uk;
    }
    return z;
}

*/


// PARALLEL
// - prima prova: spezzo il for loop e assegno ai due rank la prima metà e poi l'altra
 Eigen::VectorXd Ras::precondAction(const SpMat& x) {

     
     Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
     Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);

    // questo for devo prendere i k giusti che vede quel core, mi appoggio su sub_assingment
    auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
    for(unsigned int k : sub_division_vec){    
        Eigen::SparseLU<SpMat > lu;
        lu.compute(local_mat.getAk(k));
        auto temp = local_mat.getRk(k);
        uk = lu.solve(temp.first*x);
        z=z+(temp.second.transpose())*uk;
     }

     MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     
    //  if(rank==0){
    //    MPI_Reduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //  }
    //  else{
    //    MPI_Reduce(z.data(), nullptr, domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //  }
     
     return z;
 }


SolverResults Ras::solve(const SpMat& A, const SpMat& b, SolverTraits traits) {
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
    // int rank{0},size{0};
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // if(rank==0){
    std::cout<<"niter: "<<niter<<std::endl;
    unsigned int solves = niter*DataDD.nsub();
    std::cout<<"solves: "<<solves<<std::endl;
    double time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout <<"time in seconds: "<< time<<std::endl;
    //}
    SolverResults res_obj(uw1,solves,time, traits, DataDD);

    return res_obj;
}
