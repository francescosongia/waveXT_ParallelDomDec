#include "domaindec_solver_factory.hpp"
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "exchange_txt.h"
#include "GetPot"
#include <cassert>

int main(int argc, char* argv[]) {

     // tramite Getpot leggo tutti i parametri utili per la risoluzione
    GetPot command_line(argc, argv);
    const std::string test_name = command_line.follow("test1", 1, "-t", "--test");
    const std::string filename = ".//tests//"+test_name+"//data";
    GetPot datafile(filename.c_str());

    std::string folder_root = ".//";

    unsigned int nx = datafile("parameters/problem/nx", 20);
    unsigned int nt = datafile("parameters/problem/nt", 20);
    unsigned int nln = datafile("parameters/problem/nln", 6);
    double X = datafile("parameters/problem/x", 1.);
    double T = datafile("parameters/problem/t", 1.);

    unsigned int nsub_x = datafile("parameters/decomposition/nsubx", 2);
    unsigned int nsub_t = datafile("parameters/decomposition/nsubt", 3);
    int n = datafile("parameters/decomposition/size_subx", 0);
    int m = datafile("parameters/decomposition/size_subt", 0);

    std::string method= datafile("parameters/traits/method", "RAS");
    std::string la= datafile("parameters/traits/linear_algebra", "SeqLA");
    unsigned int max_it = datafile("parameters/traits/max_iter", 100);
    double tol = datafile("parameters/traits/tol", 1e-10);
    double tol_pipe_sx = datafile("parameters/traits/tol_pipe_sx", 1e-10);
    double it_wait = datafile("parameters/traits/it_wait_pipe", 3);

    std::string filenameA = folder_root+"tests//"+test_name+"//A.txt";
    std::string filenameb = folder_root+"tests//"+test_name+"//b.txt";
    std::string filename_coord = folder_root+"tests//"+test_name+"//coord.txt";
    
    assert(la=="SeqLA"                      && "Sequential policy required.");
    assert(!(nsub_t == 1 || nsub_x == 1)    && "nsubx and nsubt must be >= 1.");

    Domain dom(nx, nt, X, T, nln);    

    std::cout<<"Method used: "<<method<<std::endl<<std::endl;
    std::cout<<"STEP 1/3: Creating decomposition"<<std::endl;
    Decomposition DataDD(dom, nsub_x, nsub_t,n,m);
    std::cout<<"Size of space sub: "<<DataDD.sub_sizes()[0]<<"  and time sub: "<<DataDD.sub_sizes()[1]<<std::endl;

    SpMat A=readMat_fromtxt(filenameA,nt*nx*nln*2,nt*nx*nln*2);
    SpMat b=readMat_fromtxt(filenameb,nt*nx*nln*2,1);

    SolverTraits traits(max_it,tol,tol_pipe_sx,it_wait);


    int np = 1; 
    int rank = 0;

    LocalMatrices<SeqLA> local_mat(dom, DataDD, A, np, rank);
    SolverResults res_obj;
    if (method == "RAS" && la =="SeqLA"){
        DomainDecSolverFactory<Sequential,SeqLA> solver(dom,DataDD, local_mat,traits);
        res_obj=solver(method,A,b);
    }
    else{ // PIPE
        DomainDecSolverFactory<PipeSequential,SeqLA> solver(dom,DataDD, local_mat,traits);
        res_obj=solver(method,A,b);
    }

    
    auto res = res_obj.getUW();
    std::cout<<res(0)<<std::endl;
    std::string f= "results//u.txt";
    saveVec_totxt(f,res);

    res_obj.formatGNU(0,filename_coord,nx*nt,nln);
    res_obj.formatGNU(1,filename_coord,nx*nt,nln);

    return 0;
}
