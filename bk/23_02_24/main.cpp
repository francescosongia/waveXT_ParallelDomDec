#include "domain.hpp"
#include "decomposition.hpp"
#include "domaindec_solver_base.hpp"
#include "domaindec_solver_factory.hpp"
#include "ras.hpp"
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "exchange_txt.h"

int main() {
    std::cout<<"Helol"<<std::endl;

    Domain dom(20, 20, 1, 1, 6); //unsigned int nx, unsigned int nt, double x, double t, unsigned int nln
    std::cout << dom.nx() << std::endl;

    Decomposition DataDD(dom, 3, 4,8,6);

    std::cout << DataDD.overlap_back() << std::endl << std::endl;
    std::cout << DataDD.overlap_forw() << std::endl << std::endl;

    for (auto i: DataDD.start_elem()){
        std::cout << i<<" ";
    }
    std::cout<<std::endl;

    //DomainDecSolverBase solv(dom,DataDD);
    std::string filenameA=R"(C:\Users\franc\Desktop\pacsPROJECT_test\A.txt)";
    std::string filenameb=R"(C:\Users\franc\Desktop\pacsPROJECT_test\b.txt)";
    SpMat A=readMat_fromtxt(filenameA,20*20*6*2,20*20*6*2);
    SpMat b=readMat_fromtxt(filenameb,20*20*6*2,1);

    //DomainDecSolverBase base(dom,DataDD,A);

    //SpMat R1=base.getRk(1).first;
/*
    int i=0;
    for (int k=0; k<R1.outerSize(); ++k)
        for (SpMat::InnerIterator it(R1,k); it; ++it)
        {
            //it.value();
            //it.row();   // row index
            //it.col();   // col index (here it is equal to k)
            //it.index(); // inner index, here it is equal to it.row()

           // if (i<50) {
                std::cout << "i: " << it.row()+1 << "  j: " << it.col()+1 << std::endl;
                std::cout << it.value() << std::endl<<std::endl;
                ++i;
           // }

        }
       */
/*
    Eigen::VectorXd one=Eigen::VectorXd::Ones(20*20*6*2);
    Eigen::VectorXd one_1=R1*one;
    Eigen::VectorXd one_1_all=R1.transpose()*one_1;
    std::cout<<one_1_all(35)<<std::endl;
    //std::string f1=R"(C:\Users\franc\Desktop\pacsPROJECT_test\oneall.txt)";
    //saveVec_totxt(f1,one_1_all);

*/

    DomainDecSolverFactory solver(dom,DataDD,A);
    double tol{1e-10};
    unsigned int max_it{50};
    Eigen::VectorXd res=solver("RAS",b,max_it,tol);
    std::cout<<res(0)<<std::endl;
    std::string f=R"(C:\Users\franc\Desktop\pacsPROJECT_test\u.txt)";
    saveVec_totxt(f,res);



    /*
    Eigen::VectorXd u1 = Eigen::VectorXd::Constant(20*20*6*2,1);  //nln*nt*nx
    SpMat R1=solv.getRk(1).first;

    //C:\Users\franc\Desktop\pacsPROJECT_test
    std::string filename=R"(C:\Users\franc\Desktop\pacsPROJECT_test\A.txt)";
    SpMat A=solv.readA_fromtxt(filename);
    //std::cout<<A.row(0)<<std::endl;

    solv.createAlocal(A);
    SpMat A1=solv.getAk(1);
    std::cout<<A1.row(0)<<std::endl;
     */
/*
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









    return 0;
}
