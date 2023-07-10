#include "domain.hpp"
#include "decomposition.hpp"
#include "domaindec_solver_base.hpp"
#include "domaindec_solver_factory.hpp"
#include "GetPot"
#include "ras.hpp"
#include "solver_traits.h"
#include "solver_results.hpp"
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "exchange_txt.h"

int main(int argc, char **argv) {

     // tramite Getpot leggo tutti i parametri utili per la risoluzione
    GetPot command_line(argc, argv);
    const std::string filename = command_line.follow("data", 2, "-f", "--file");
    GetPot datafile(filename.c_str());

    unsigned int nx = datafile("parameters/problem/nx", 20);
    unsigned int nt = datafile("parameters/problem/nt", 20);
    unsigned int nln = datafile("parameters/problem/nln", 6);
    double X = datafile("parameters/problem/x", 1.);
    double T = datafile("parameters/problem/t", 1.);

    unsigned int nsub_x = datafile("parameters/decomposition/nsubx", 2);
    unsigned int nsub_t = datafile("parameters/decomposition/nsubt", 3);
    int n = datafile("parameters/decomposition/size_subx", 0);
    int m = datafile("parameters/decomposition/size_subt", 0);

    std::string test_matrices=datafile("file_matrices/test", "test1");
    std::string filenameA = "//home//scientific-vm//Desktop//projectPACS//problem_matrices//"+test_matrices+"//A.txt";
    std::string filenameb = "//home//scientific-vm//Desktop//projectPACS//problem_matrices//"+test_matrices+"//b.txt";
    std::string filename_coord = "//home//scientific-vm//Desktop//projectPACS//problem_matrices//"+test_matrices+"//coord.txt";
    
    /*
    //20 100, x1t5
    nx=20;
    nt=100;
    X=1;
    T=5;
    nln=6;

    nsub_x=2; //3
    nsub_t=20; //4
    // n=12;  //8
    // m=6;
    std::string filenameA=R"(/home/scientific-vm/Desktop/projectPACS/A_1_5.txt)";
    std::string filenameb=R"(/home/scientific-vm/Desktop/projectPACS/b_1_5.txt)";
    //std::string filenameA=R"(C:\Users\franc\Desktop\pacsPROJECT_test\A.txt)";
    //std::string filenameb=R"(C:\Users\franc\Desktop\pacsPROJECT_test\b.txt)";
    

    
    nx=20;
    nt=20;
    X=1;
    T=1;
    nln=6;

    nsub_x=2;
    nsub_t=10; //4;
    // n=12;
    // m=6;
    std::string filenameA=R"(/home/scientific-vm/Desktop/projectPACS/A.txt)";
    std::string filenameb=R"(/home/scientific-vm/Desktop/projectPACS/b.txt)";
    //std::string filenameA=R"(C:\Users\franc\Desktop\pacsPROJECT_test\A.txt)";
    //std::string filenameb=R"(C:\Users\franc\Desktop\pacsPROJECT_test\b.txt)";
    */


    Domain dom(nx, nt, X, T, nln);    
    Decomposition DataDD(dom, nsub_x, nsub_t);
    std::cout<<"Decomposition created"<<std::endl;

    SpMat A=readMat_fromtxt(filenameA,nt*nx*nln*2,nt*nx*nln*2);
    SpMat b=readMat_fromtxt(filenameb,nt*nx*nln*2,1);
    std::cout<<"get problem matrices"<<std::endl;

    double tol{1e-10};
    unsigned int max_it{50};
    SolverTraits traits(max_it,tol);
    std::string method="RAS";
    std::cout<<"method used: "<<method<<std::endl;

    int np = 0;
    int rank = 0;
    LocalMatrices local_mat(dom, DataDD, A, np, rank);
    DomainDecSolverFactory solver(dom,DataDD, local_mat);
    SolverResults res_obj=solver(method,A,b,traits);
    auto res = res_obj.getUW();
    //Eigen::VectorXd res=solver(method,A,b,traits);
    std::cout<<res(0)<<std::endl;
    std::string f=R"(/home/scientific-vm/Desktop/projectPACS/u.txt)";
    saveVec_totxt(f,res);

    res_obj.formatGNU(0,filename_coord,nx*nt,nln);
    res_obj.formatGNU(1,filename_coord,nx*nt,nln);


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
