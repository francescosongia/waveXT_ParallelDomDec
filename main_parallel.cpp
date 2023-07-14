#include "local_matrices.hpp"
#include "domaindec_solver_factory.hpp"
#include "solver_traits.h"
#include "solver_results.hpp"
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "exchange_txt.h"
#include "GetPot"
#include <mpi.h>
//troppi include

int main(int argc, char **argv) {

    GetPot command_line(argc, argv);
    const std::string filename = command_line.follow("data", 2, "-f", "--file");
    GetPot datafile(filename.c_str());

    std::string folder_root = "//home//scientific-vm//Desktop//branch_pacs//projectPACS//";
                                //"//home//scientific-vm//Desktop//projectPACS//";

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

    MPI_Init(&argc, &argv);
    
    int rank{0},size{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //NEXT
    
    // sistemare makefile
    // sistemare include
    // opzione verbose
    // commentare e ordinare file (rimuovere cpp che non servono)

    // assert
    // prova con problma piu grosso, altri test cases e dividere i data file di getpot
    // aggiungere const                             
    // gestire altri modi di parallelizzare (matrice dei sub assegnati, caso in cui non sono divisibili)
    // postproccesing (senza codice, confrontare però la varie policy in termini di tempo e solves)
    // aggiungere nel getpot anche iter e toll
    
    
    // qui sto facendo if con errori, sarebbe da proporre versione 
    // accettabile (modificare nsubx o policy LA) e andare avanti
    // FARLI CON ASSERT !
    if(size%2 !=0){
        std::cerr<<"even number of cores required"<<std::endl;
        return 0;
    }
    if(nsub_x == 1 && la =="ParLA"){
        std::cerr<<"ParLA not possibile with subx=1"<<std::endl;
        return 0;
    }
    if(nsub_x == 1 && la =="SeqLA"){
        std::cerr<<"increase the number of subx, SeqLA requires size=nsubx, size>=2 "<<std::endl;
        return 0;
    }
    if(size == nsub_x && la =="ParLA"){
        std::cerr<<"with size = nsubx, SeqLA is needed. If you want to use ParLA decrease the number of subx."<<std::endl;
        return 0;
    }
    if(la=="SeqLA" && size!=nsub_x && size>=2){
        std::cerr<<"SeqLA requires size=nsubx."<<std::endl;
        return 0;
    }
    if(la=="ParLA" && size%nsub_x !=0){
        std::cerr<<"ParLA requires size proportinal to nsubx."<<std::endl;
        return 0;
    }

    if(nsub_t == 1 || nsub_x == 1){
        std::cerr<<"nsubx and nsubt must be >= 1"<<std::endl;
        return 0;
    }



    Domain dom(nx, nt, X, T, nln);
    Decomposition DataDD(dom, nsub_x, nsub_t);
    std::cout<<"Decomposition created, rank: "<<rank<<std::endl;

    SpMat A=readMat_fromtxt(filenameA,nt*nx*nln*2,nt*nx*nln*2);
    SpMat b=readMat_fromtxt(filenameb,nt*nx*nln*2,1);
    std::cout<<"get problem matrices, rank: "<<rank<<std::endl;

    double tol{1e-10};
    unsigned int max_it{50};
    SolverTraits traits(max_it,tol);


    std::cout<<"method used: "<<method<<", rank: "<<rank<<std::endl;
    std::cout<<"LA policy used: "<<la<<", rank: "<<rank<<std::endl;    
        
    SolverResults res_obj;
    
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
    else{// if (method == "PIPE" and la == "SeqLA") {
        LocalMatrices<SeqLA> local_mat(dom, DataDD, A, size, rank);
        DomainDecSolverFactory<PipeParallel_SeqLA,SeqLA> solver(dom,DataDD,local_mat,traits);
        res_obj=solver(method,A,b);
    }

    auto res = res_obj.getUW();
    if (rank==0){
        std::cout<<res(0)<<std::endl;
        std::string f= folder_root+"u.txt";
        saveVec_totxt(f,res);

        res_obj.formatGNU(0,filename_coord,nx*nt,nln);
        res_obj.formatGNU(1,filename_coord,nx*nt,nln);
    }
    
    MPI_Finalize();

    return 0;
}



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

    nsub_x=2; //3
    nsub_t=10; //4
    // n=12;  //8
    // m=6;
    std::string filenameA=R"(/home/scientific-vm/Desktop/projectPACS/A.txt)";
    std::string filenameb=R"(/home/scientific-vm/Desktop/projectPACS/b.txt)";
    //std::string filenameA=R"(C:\Users\franc\Desktop\pacsPROJECT_test\A.txt)";
    //std::string filenameb=R"(C:\Users\franc\Desktop\pacsPROJECT_test\b.txt)";
    */