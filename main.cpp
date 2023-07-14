#include "domain.hpp"
#include "decomposition.hpp"
#include "domaindec_solver_base.hpp"
#include "domaindec_solver_factory.hpp"
#include "GetPot"
#include "ras.hpp"
#include "ras_pipelined.hpp"
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

    std::string folder_root = "//home//scientific-vm//Desktop//branch_pacs//projectPACS//";

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

    std::string test_matrices=datafile("file_matrices/test", "test1");
    std::string filenameA = folder_root+"problem_matrices//"+test_matrices+"//A.txt";
    std::string filenameb = folder_root+"problem_matrices//"+test_matrices+"//b.txt";
    std::string filename_coord = folder_root+"problem_matrices//"+test_matrices+"//coord.txt";
    
    if(la!= "SeqLA"){
        std::cerr<<"sequential policy required"<<std::endl;
        return 0;
    }
    if(nsub_t == 1 || nsub_x == 1){
        std::cerr<<"nsubx and nsubt must be >= 1"<<std::endl;
        return 0;
    }

    Domain dom(nx, nt, X, T, nln);    
    Decomposition DataDD(dom, nsub_x, nsub_t);
    std::cout<<"Decomposition created"<<std::endl;

    SpMat A=readMat_fromtxt(filenameA,nt*nx*nln*2,nt*nx*nln*2);
    SpMat b=readMat_fromtxt(filenameb,nt*nx*nln*2,1);
    std::cout<<"get problem matrices"<<std::endl;

    double tol{1e-10};
    unsigned int max_it{50};
    SolverTraits traits(max_it,tol);

    std::cout<<"method used: "<<method<<std::endl;

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
    std::string f= folder_root+"u.txt";
    saveVec_totxt(f,res);

    res_obj.formatGNU(0,filename_coord,nx*nt,nln);
    res_obj.formatGNU(1,filename_coord,nx*nt,nln);

    return 0;
}
