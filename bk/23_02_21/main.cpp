#include "domain.hpp"
#include "decomposition.hpp"
#include "domaindec_solver_base.hpp"
#include <iostream>
#include <vector>
#include "Eigen/Dense"

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

    DomainDecSolverBase solv(dom,DataDD);
    Eigen::VectorXd u1 = Eigen::VectorXd::Constant(20*20*6,1);  //nln*nt*nx
    SpMat R1=solv.getRk(1).first;
    /*
    int i=1;
    for (int k=0; k<R1.outerSize(); ++k)
        for (SpMat::InnerIterator it(R1,k); it; ++it)
        {

            //it.value();
            //it.row();   // row index
            //it.col();   // col index (here it is equal to k)
            //it.index(); // inner index, here it is equal to it.row()

            //std::cout<<i<<std::endl;
            i++;
            std::cout<<it.value();
        }
    std::cout<<std::endl;
    std::cout<<i<<std::endl;
*/

    Eigen::VectorXd temp= R1 * u1;
    std::cout<<R1.rows()<<std::endl;
    std::cout<<R1.cols()<<std::endl;
    std::cout<<u1.rows()<<std::endl;
    std::cout<<u1.cols()<<std::endl;

    Eigen::VectorXd res= R1.transpose() * temp;
    double s=0;
    for(Eigen::Index i=0;i<res.size();++i){
        s+=res(i);
    }
    std::cout<<s<<std::endl;
    std::cout<<res.size()<<std::endl;







    return 0;
}
