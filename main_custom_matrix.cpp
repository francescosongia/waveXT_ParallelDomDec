#include "domaindec_solver_factory.hpp"
#include <iostream>
#include <vector>
#include "Eigen/Dense"  // eigen version 3.4.0
#include "exchange_txt.h"
#include "GetPot"
#include <mpi.h>
#include <random>
#include <fstream>
#include <cassert>
#include <set>


int main(int argc, char* argv[]) {
    
    // read parameters and polices from getpot
    GetPot command_line(argc, argv);
    const std::string test_name = "custom";
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
    std::string la= datafile("parameters/traits/ParPolicy", "AloneOnStride");
    unsigned int max_it = datafile("parameters/traits/max_iter", 100);
    double tol = datafile("parameters/traits/tol", 1e-10);
    double tol_pipe_sx = datafile("parameters/traits/tol_pipe_sx", 1e-10);
    unsigned int it_wait = datafile("parameters/traits/it_wait_pipe", 3);

    std::string filenameA = folder_root+"tests//"+test_name+"//A.txt";
    std::string filenameb = folder_root+"tests//"+test_name+"//b.txt";
    std::string filename_coord = folder_root+"tests//"+test_name+"//coord.txt";

    int custom_matrix_random = datafile("custom_matrix/random", 1);


    MPI_Init(&argc, &argv);
    
    int rank{0},size{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // check parameters and policies
    std::set<std::string> method_implemented = {"RAS", "PIPE" };
    if (method_implemented.find(method) == method_implemented.end()) 
        std::cerr<<"Method non available. Choose between RAS and PIPE"<<std::endl;

    assert(!(la != "AloneOnStride")   
           && "Custom matrix not possible with CooperationOnStride or CooperationSplitTime.");
    assert(!(size%2 !=0)          
           && "Even number of cores required.");
    assert(!(nsub_x == 1 && la =="CooperationOnStride")  
           && "CooperationOnStride not possibile with subx = 1.");
    assert(!(nsub_x == 1 && la =="AloneOnStride")     
           && "Increase the number of subx, AloneOnStride requires size = nsubx with size >= 2. ");
    assert(!(size == nsub_x && la =="CooperationOnStride") 
           && "With size = nsubx, AloneOnStride is needed. If you want to use CooperationOnStride decrease the number of subx.");
    assert(!(la == "AloneOnStride" && size!=nsub_x && size>=2) 
           && "AloneOnStride requires size = nsubx.");
    assert(!(la == "CooperationOnStride" && size%nsub_x !=0)   
           && "CooperationOnStride requires size proportional to nsubx.");
    assert(!(nsub_t == 1 || nsub_x == 1)   
           && "nsubx and nsubt must be >= 1.");

    // --------------------------------------------------------------------------
    // CUSTOM MATRIX
    Eigen::MatrixXi custom_matrix_sub_assignment(nsub_x, nsub_t);
    // random
    if(custom_matrix_random > 0){
        std::mt19937 gen(1234);  
        std::uniform_int_distribution<> dis(0, size-1);
        
        for (int i = 0; i < custom_matrix_sub_assignment.rows(); ++i) {
            for (int j = 0; j < custom_matrix_sub_assignment.cols(); ++j) 
                custom_matrix_sub_assignment(i, j) = dis(gen);
        }
    }
    //from file
    else{
        std::ifstream file_custom_mat;
        file_custom_mat.open(folder_root+"tests//"+test_name+"//custom_matrix.txt");
        if (!file_custom_mat.is_open()) {
            std::cerr << "Error when reading the custom matrix file." << std::endl;
            return 1;
        }

        for (int i = 0; i < nsub_x; ++i) 
            for (int j = 0; j < nsub_t; ++j) 
                if (!(file_custom_mat >> custom_matrix_sub_assignment(i, j))) {
                    std::cerr << "Error when reading the custom matrix file, check dimensions." << std::endl;
                    file_custom_mat.close();
                    return 1;
                }
        file_custom_mat.close();
    }

    if(rank==0){
        std::cout<<"custom matrix"<<std::endl;
        std::cout<<custom_matrix_sub_assignment<<std::endl;
    }
    // ---------------------------------------------------------------------------

    Domain dom(nx, nt, X, T, nln);
    if(rank==0) {
        std::cout<<"Method used: "<<method<<std::endl;
        std::cout<<"LinearAlgebra policy used: "<<la<<std::endl<<std::endl;
        std::cout<<"STEP 1/3: Creating decomposition"<<std::endl;
        };
    Decomposition DataDD(dom, nsub_x, nsub_t,n,m);
    if(rank==0){std::cout<<"Size of space sub: "<<DataDD.sub_sizes()[0]<<"  and time sub: "<<DataDD.sub_sizes()[1]<<std::endl;};
    
    //read global matrices from file
    SpMat A=readMat_fromtxt(filenameA,nt*nx*nln*2,nt*nx*nln*2);
    SpMat b=readMat_fromtxt(filenameb,nt*nx*nln*2,1);

    SolverTraits traits(max_it,tol,tol_pipe_sx,it_wait);

    SolverResults res_obj;
    
    if (method == "RAS" && la == "AloneOnStride") {
        LocalMatrices<AloneOnStride> local_mat(dom, DataDD, A, size, rank,custom_matrix_sub_assignment);
        DomainDecSolverFactory<Parallel_AloneOnStride,AloneOnStride> solver(dom,DataDD,local_mat,traits);
        res_obj=solver(method,A,b);
    }
    else if (method == "PIPE" and la == "AloneOnStride") {
        LocalMatrices<AloneOnStride> local_mat(dom, DataDD, A, size, rank,custom_matrix_sub_assignment);
        DomainDecSolverFactory<PipeParallel_AloneOnStride,AloneOnStride> solver(dom,DataDD,local_mat,traits);
        res_obj=solver(method,A,b);
    }
    else{
        std::cerr<<"invalid method"<<std::endl;
        return 0;
    }
    // postprocessing
    auto res = res_obj.getUW();
    if (rank==0){
        std::string f= "results/u.txt"; 
        saveVec_totxt(f,res);

        res_obj.formatGNU(0,filename_coord,nx*nt,nln);
        res_obj.formatGNU(1,filename_coord,nx*nt,nln);
    }
    
    MPI_Finalize();

    return 0;
}
