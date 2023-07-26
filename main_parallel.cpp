#include "domaindec_solver_factory.hpp"
#include <iostream>
#include <vector>
#include "Eigen/Dense"  // eigen version 3.4.0
#include "exchange_txt.h"
#include "GetPot"
#include <mpi.h>
#include <random>
#include <cassert>
#include <set>


int main(int argc, char* argv[]) {

    GetPot command_line(argc, argv);
    const std::string test_name = command_line.follow("test1", 1, "-t", "--test");
    const std::string filename = ".//tests//"+test_name+"//data";
    GetPot datafile(filename.c_str());

    std::string folder_root =  ".//";
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
    unsigned int it_wait = datafile("parameters/traits/it_wait_pipe", 3);

    //std::string test_matrices=datafile("file_matrices/test", "test1");
    std::string filenameA = folder_root+"tests//"+test_name+"//A.txt";
    std::string filenameb = folder_root+"tests//"+test_name+"//b.txt";
    std::string filename_coord = folder_root+"tests//"+test_name+"//coord.txt";


    MPI_Init(&argc, &argv);
    
    int rank{0},size{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    //--------------------------------------------------------------------
    //NEXT
    
    // commentare
    // tabellina speed up
    // make help 

    // chiedere:
    // far vedere le parti di parallelizz
    // if constexpr
    // chiedere strutture e genericità generali
    // librerire come iegen non in include (linkarle)
    // controllare rowmajor ordering of spmat (parto da commento in localmatrices). SE LO CAMBIO NON FUNZIONA NULLA, NON VALE LA PENA FORSE PERDERCI TEMPO                           
    //--------------------------------------------------------------------


    /*
    std::set<std::string> method_implemented = {"RAS", "PIPE" };
    std::set<std::string> la_policy_implemented = {"SeqLA", "ParLA" };
    if (method_implemented.find(method) == method_implemented.end()) 
        std::cerr<<"Method non available. Choose between RAS and PIPE"<<std::endl;
    if (la_policy_implemented.find(la) == la_policy_implemented.end()) 
        std::cerr<<"Linear Algebra policy non available. Choose between SeqLA and ParLA"<<std::endl;
    
    assert(!(size%2 !=0)                               && "Even number of cores required.");
    assert(!(nsub_x == 1 && la =="ParLA")              && "ParLA not possibile with subx = 1.");
    assert(!(nsub_x == 1 && la =="SeqLA")              && "Increase the number of subx, SeqLA requires size = nsubx with size >= 2. ");
    assert(!(size == nsub_x && la =="ParLA")           && "With size = nsubx, SeqLA is needed. If you want to use ParLA decrease the number of subx.");
    assert(!(la == "SeqLA" && size!=nsub_x && size>=2) && "SeqLA requires size = nsubx.");
    assert(!(la == "ParLA" && size%nsub_x !=0)         && "ParLA requires size proportional to nsubx.");
    assert(!(nsub_t == 1 || nsub_x == 1)               && "nsubx and nsubt must be >= 1.");
    */
    
    Domain dom(nx, nt, X, T, nln);
    if(rank==0) {
        std::cout<<"Method used: "<<method<<std::endl;
        std::cout<<"LinearAlgebra policy used: "<<la<<std::endl<<std::endl;
        std::cout<<"STEP 1/3: Creating decomposition"<<std::endl;
        };
    Decomposition DataDD(dom, nsub_x, nsub_t,n,m);
    if(rank==0){std::cout<<"Size of space sub: "<<DataDD.sub_sizes()[0]<<"  and time sub: "<<DataDD.sub_sizes()[1]<<std::endl;};
    
    

    SpMat A=readMat_fromtxt(filenameA,nt*nx*nln*2,nt*nx*nln*2);
    SpMat b=readMat_fromtxt(filenameb,nt*nx*nln*2,1);

    SolverTraits traits(max_it,tol,tol_pipe_sx,it_wait);

    SolverResults res_obj;
    LocalMatrices<SeqLA> local_mat(dom, DataDD, A, size, rank);
    
    /*
    if (method == "RAS" && la == "ParLA"){
        LocalMatrices<ParLA> local_mat(dom, DataDD, A, size, rank);
        DomainDecSolverFactory<Parallel_ParLA,ParLA> solver(dom,DataDD,local_mat,traits);
        res_obj=solver(method,A,b);
    }
    else if (method == "PIPE" && la == "ParLA") {
        LocalMatrices<ParLA> local_mat(dom, DataDD, A, size, rank);
        DomainDecSolverFactory<PipeParallel_ParLA,ParLA> solver(dom,DataDD,local_mat,traits);
        res_obj=solver(method,A,b);
    }
    else if (method == "RAS" && la == "SeqLA") {
        LocalMatrices<SeqLA> local_mat(dom, DataDD, A, size, rank);
        DomainDecSolverFactory<Parallel_SeqLA,SeqLA> solver(dom,DataDD,local_mat,traits);
        res_obj=solver(method,A,b);
    }
    else if (method == "PIPE" and la == "SeqLA") {
        LocalMatrices<SeqLA> local_mat(dom, DataDD, A, size, rank);
        DomainDecSolverFactory<PipeParallel_SeqLA,SeqLA> solver(dom,DataDD,local_mat,traits);
        res_obj=solver(method,A,b);
    }
    else{
        std::cerr<<"invalid method"<<std::endl;
        return 0;
    }

    auto res = res_obj.getUW();
    if (rank==0){
        std::cout<<res(0)<<std::endl;
        std::string f= "results/u.txt"; 
        saveVec_totxt(f,res);

        res_obj.formatGNU(0,filename_coord,nx*nt,nln);
        res_obj.formatGNU(1,filename_coord,nx*nt,nln);
    }
    */
    MPI_Finalize();

    return 0;
}
