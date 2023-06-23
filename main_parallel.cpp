#include "domain.hpp"
#include "decomposition.hpp"
#include "local_matrices.hpp"
#include "domaindec_solver_base.hpp"
#include "domaindec_solver_factory.hpp"
#include "ras.hpp"
#include "solver_traits.h"
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "exchange_txt.h"
#include <mpi.h>
//troppi include

int main(int argc, char **argv) {
    unsigned int nx,nt,nln,nsub_x,nsub_t;
    double X,T;
    int n,m;
    nx=20;
    nt=20;
    X=1;
    T=1;
    nln=6;

    nsub_x=2; //3
    nsub_t=4;
    n=12;  //8
    m=6;
    // then with GetPot

    //NEXT
    // fare lo stesso per pipe
    // ognuno fa solo le sue local
    // aggiungo vettore ordinamento locale rk
    // gestire meglio np, rank. ParallelTraits
    // generica interfaccia sequantial vs parallel
    // intraparallelization
    // postproccesing


    Domain dom(nx, nt, X, T, nln);
    Decomposition DataDD(dom, nsub_x, nsub_t,n,m);
    std::cout<<"Decomposition created"<<std::endl;
    std::string filenameA=R"(/home/scientific-vm/Desktop/projectPACS/A.txt)";
    std::string filenameb=R"(/home/scientific-vm/Desktop/projectPACS/b.txt)";
    //std::string filenameA=R"(C:\Users\franc\Desktop\pacsPROJECT_test\A.txt)";
    //std::string filenameb=R"(C:\Users\franc\Desktop\pacsPROJECT_test\b.txt)";
    MPI_Init(NULL,NULL);
    int rank{0},np{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    SpMat A=readMat_fromtxt(filenameA,nt*nx*nln*2,nt*nx*nln*2);
    SpMat b=readMat_fromtxt(filenameb,nt*nx*nln*2,1);
    std::cout<<"get problem matrices, rank: "<<rank<<std::endl;

    double tol{1e-10};
    unsigned int max_it{50};
    SolverTraits traits(max_it,tol);
    std::string method="RAS";
    std::cout<<"method used: "<<method<<", rank: "<<rank<<std::endl;

    LocalMatrices local_mat(dom, DataDD, A, np, rank);
    DomainDecSolverFactory solver(dom,DataDD,local_mat);
    Eigen::VectorXd res=solver(method,A,b,traits);
    if (rank==0){
    std::cout<<res(0)<<std::endl;
    //std::string f=R"(C:\Users\franc\Desktop\pacsPROJECT_test\u.txt)";
    std::string f=R"(/home/scientific-vm/Desktop/projectPACS/u.txt)";
    saveVec_totxt(f,res);
    }
    MPI_Finalize();


/*
    //vedere valori di sparsa
    int i=0;
    for (int k=0; k<A1.outerSize(); ++k)
        for (SpMat::InnerIterator it(A1,k); it; ++it)
        {

            //it.value();
            //it.row();   // row index
            //it.col();   // col index (here it is equal to k)
            //it.index(); // inner index, here it is equal to it.row()
            if (i<50) {
                std::cout << "i: " << it.row()+1 << "  j: " << it.col()+1 << std::endl;
                std::cout << it.value() << std::endl<<std::endl;
                ++i;
            }
        }

*/

    //DomainDecSolverBase base(dom,DataDD,A);

    //SpMat R1=base.getRk(1).first;
/*
    Eigen::VectorXd one=Eigen::VectorXd::Ones(20*20*6*2);
    Eigen::VectorXd one_1=R1*one;
    Eigen::VectorXd one_1_all=R1.transpose()*one_1;
    std::cout<<one_1_all(35)<<std::endl;
    //std::string f1=R"(C:\Users\franc\Desktop\pacsPROJECT_test\oneall.txt)";
    //saveVec_totxt(f1,one_1_all);

*/








    return 0;
}
