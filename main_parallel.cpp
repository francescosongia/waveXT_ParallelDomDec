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
         // tramite Getpot leggo tutti i parametri utili per la risoluzione
    GetPot command_line(argc, argv);
    const std::string filename = command_line.follow("data", 2, "-f", "--file");
    GetPot datafile(filename.c_str());

    std::string folder_root = "//home//scientific-vm//Desktop//branch_pacs//projectPACS//";//"//home//scientific-vm//Desktop//projectPACS//";

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

    std::string test_matrices=datafile("file_matrices/test", "test1");
    std::string filenameA = folder_root+"problem_matrices//"+test_matrices+"//A.txt";
    std::string filenameb = folder_root+"problem_matrices//"+test_matrices+"//b.txt";
    std::string filename_coord = folder_root+"problem_matrices//"+test_matrices+"//coord.txt";


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

    /*
    con questa versione ho creato localmatrices prima dividendo questo step dallo step del solver. va bene anche cosi. 
    prima pensavo di farlo per non far fare a core 2 il solver ma in realtà deve farlo per forza. 
    teniamo comunque questa versione, forse piu ordinata nella creazione delle local mat e gia subito posso decidere 
    quali creare a seconda del rank in cui sono
    */

    //NEXT
    // fare lo stesso per pipe                      OK MA ASSUNZIONE (RAGIONEVOLE) NSUBX divisibile per NP
    // ognuno fa solo le sue local                  OK
    // aggiungo vettore ordinamento locale rk       OK MA SOLO NEL CASO SUPER BASE CON PARALL SPAZIO E TUTTO DIVISIBILE
    // gestire meglio np, rank. ParallelTraits
    // pensare se è necessario fare allreduce in precondAction  OK, MAIL BAIONI
    // generica interfaccia sequantial vs parallel  OK IDEA CON ESEMPIO COMPARE POLICY
    // prova con problma piu grosso                 OK PARALLEL ANCORA PIU LENTO
    // mettere mpi.h in include                     OK
    // utlizzare solver_results come return in solve OK
    // aggiungere const                             OK  
    // gestire altri modi di parallelizzare (matrice dei sub assegnati, caso in cui non sono divisibili)
    // intraparallelization
    // postproccesing                               FATTO GNUPLOT
    // correggere solve in pipe, differenziarlo
    // mettere in cpp
    // gesitre tutti i casi, divisione aribitaria sub vs np. 


    Domain dom(nx, nt, X, T, nln);
    Decomposition DataDD(dom, nsub_x, nsub_t);
    std::cout<<"Decomposition created"<<std::endl;
    
    MPI_Init(&argc, &argv);
    //MPI_Init(NULL,NULL);
    
    int rank{0},np{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    SpMat A=readMat_fromtxt(filenameA,nt*nx*nln*2,nt*nx*nln*2);
    SpMat b=readMat_fromtxt(filenameb,nt*nx*nln*2,1);
    std::cout<<"get problem matrices, rank: "<<rank<<std::endl;

    double tol{1e-10};
    unsigned int max_it{50};
    SolverTraits traits(max_it,tol);
    //std::string method="RAS";
    std::cout<<"method used: "<<method<<", rank: "<<rank<<std::endl;

    LocalMatrices local_mat(dom, DataDD, A, np, rank);
    SolverResults res_obj;
    
    if (method == "RAS"){
        DomainDecSolverFactory<Parallel> solver(dom,DataDD,local_mat,traits);
        res_obj=solver(method,A,b);
    }
    else {
        DomainDecSolverFactory<PipeParallel> solver_pipe(dom,DataDD,local_mat,traits);
        res_obj=solver_pipe(method,A,b);
    }

    auto res = res_obj.getUW();
    if (rank==0){
    std::cout<<res(0)<<std::endl;
    //std::string f=R"(C:\Users\franc\Desktop\pacsPROJECT_test\u.txt)";
    //std::string f=R"(/home/scientific-vm/Desktop/pacs_primepolicy_nonfunziona_0607/u.txt)";
    std::string f= folder_root+"u.txt";
    saveVec_totxt(f,res);

    res_obj.formatGNU(0,filename_coord,nx*nt,nln);
    res_obj.formatGNU(1,filename_coord,nx*nt,nln);
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
