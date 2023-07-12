#ifndef RAS_HPP_
#define RAS_HPP_

#include <utility>

#include "domaindec_solver_base.hpp"
#include <iostream>


template<class P, class LA>
class Ras : public DomainDecSolverBase<P,LA>{

public:

  Ras(Domain dom, const Decomposition& dec,const LocalMatrices<LA> local_matrices,const SolverTraits& traits) : 
    DomainDecSolverBase<P,LA>(dom,dec,local_matrices,traits)
    {};

  SolverResults solve(const SpMat& A, const SpMat& b) override
  {
    P func_wrapper(this->domain,this->DataDD,this->local_mat,this->traits_);
    return func_wrapper.solve(A,b);
  };


protected:
  
  Eigen::VectorXd precondAction(const SpMat& x) override
  {
    P func_wrapper(this->domain,this->DataDD,this->local_mat,this->traits_);
    return func_wrapper.precondAction(x);
    //PolicyFunctionWrapper<P> func_wrapper;
    //return func_wrapper.precondAction(x);
  };

};

 

class Parallel_SeqLA : public Ras<Parallel_SeqLA,SeqLA>
{
  public:
    Parallel_SeqLA(Domain dom, const Decomposition& dec,const LocalMatrices<SeqLA> local_matrices,const SolverTraits& traits) :
     Ras<Parallel_SeqLA,SeqLA>(dom,dec,local_matrices,traits) {};

    Eigen::VectorXd 
    precondAction(const SpMat& x) 
    {
      Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
      Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);

      // questo for devo prendere i k giusti che vede quel core, mi appoggio su sub_assingment
      auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
      for(unsigned int k : sub_division_vec){    
          Eigen::SparseLU<SpMat > lu;
          lu.compute(local_mat.getAk(k));
          auto temp = local_mat.getRk(k);
          uk = lu.solve(temp.first*x);
          z=z+(temp.second.transpose())*uk;
          }

      MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return z;
    }


    SolverResults solve(const SpMat& A, const SpMat& b)
    {
      auto start = std::chrono::steady_clock::now();
      double tol=this->traits_.tol();
      unsigned int max_it=this->traits_.max_it();
      double res=tol+1;
      std::vector<double> relres2P_vec;
      unsigned int niter=0;
      double Pb2=precondAction(b).norm();

      Eigen::VectorXd uw0=Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
      Eigen::VectorXd uw1(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);

      Eigen::VectorXd z= precondAction(b); //b-A*uw0
      while(res>tol and niter<max_it){
          uw1=uw0+z;  //forse si puo evitare l'uso di entrambi uw0 e uw1
          z= precondAction(b-A*uw1);
          res=(z/Pb2).norm();
          relres2P_vec.push_back(res);
          niter++;
          uw0=uw1;
      }

      auto end = std::chrono::steady_clock::now();
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      unsigned int solves{0};
      double time{0.0};
      if(rank==0){
      std::cout<<"niter: "<<niter<<std::endl;
      solves = niter*this->DataDD.nsub();
      std::cout<<"solves: "<<solves<<std::endl;
      time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
      std::cout <<"time in seconds: "<< time<<std::endl;
      }
      SolverResults res_obj(uw1,solves,time, this->traits_, DataDD);

      return res_obj;
    };
};


class Sequential : public Ras<Sequential,SeqLA>
{
public:
  Sequential(Domain dom, const Decomposition& dec,const LocalMatrices<SeqLA> local_matrices,const SolverTraits& traits) :
   Ras<Sequential,SeqLA>(dom,dec,local_matrices,traits) {};
  
  Eigen::VectorXd precondAction(const SpMat& x) 
  {
    Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    //Eigen::VectorXd xk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    for(unsigned int k=1;k<=DataDD.nsub();++k){

        Eigen::SparseLU<SpMat > lu;
        lu.compute(local_mat.getAk(k));

        auto temp = local_mat.getRk(k);

        uk = lu.solve(temp.first*x);

        z=z+(temp.second.transpose())*uk;
    }
    return z;
  }

  SolverResults solve(const SpMat& A, const SpMat& b)
  {
    auto start = std::chrono::steady_clock::now();
    double tol=this->traits_.tol();
    unsigned int max_it=this->traits_.max_it();
    double res=tol+1;
    std::vector<double> relres2P_vec;
    unsigned int niter=0;
    double Pb2=precondAction(b).norm();

    Eigen::VectorXd uw0=Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
    Eigen::VectorXd uw1(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);

    Eigen::VectorXd z= precondAction(b); //b-A*uw0
    while(res>tol and niter<max_it){
        uw1=uw0+z;  //forse si puo evitare l'uso di entrambi uw0 e uw1
        z= precondAction(b-A*uw1);
        res=(z/Pb2).norm();
        relres2P_vec.push_back(res);
        niter++;
        uw0=uw1;
    }

    auto end = std::chrono::steady_clock::now();
    std::cout<<"niter: "<<niter<<std::endl;
    unsigned int solves = niter*this->DataDD.nsub();
    std::cout<<"solves: "<<solves<<std::endl;
    double time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout <<"time in seconds: "<< time<<std::endl;
    SolverResults res_obj(uw1,solves,time, this->traits_, this->DataDD);

    return res_obj;
  };

};


////////

class Parallel_ParLA : public Ras<Parallel_ParLA,ParLA>
{
  public:
    Parallel_ParLA(Domain dom, const Decomposition& dec,const LocalMatrices<ParLA> local_matrices,const SolverTraits& traits) :
     Ras<Parallel_ParLA,ParLA>(dom,dec,local_matrices,traits) {};

    Eigen::VectorXd 
    precondAction(const SpMat& x) 
    {
      Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
      Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);

      // questo for devo prendere i k giusti che vede quel core, mi appoggio su sub_assingment
      auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
      for(unsigned int k : sub_division_vec){    
          Eigen::SparseLU<SpMat > lu;
          lu.compute(local_mat.getAk(k));
          auto temp = local_mat.getRk(k);
          uk = lu.solve(temp.first*x);
          z=z+(temp.second.transpose())*uk;
          }

      MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return z;
    }


    SolverResults solve(const SpMat& A, const SpMat& b)
    {
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      // assumimao size divisibile per due

      auto start = std::chrono::steady_clock::now();
      double tol=this->traits_.tol();
      unsigned int max_it=this->traits_.max_it();
      double res=tol+1;
      std::vector<double> relres2P_vec;
      unsigned int niter=0;
      double Pb2=precondAction(b).norm();

      Eigen::VectorXd uw0=Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
      Eigen::VectorXd uw1(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);

      //scatter A, ottengo a_intraparallel_1_2

      Eigen::VectorXd z= precondAction(b); //b-A*uw0
      while(res>tol and niter<max_it){
          uw1=uw0+z;  //forse si puo evitare l'uso di entrambi uw0 e uw1
          z= precondAction(b-A*uw1);
          res=(z/Pb2).norm();
          relres2P_vec.push_back(res);
          niter++;
          uw0=uw1;
      }

      auto end = std::chrono::steady_clock::now();
      
      unsigned int solves{0};
      double time{0.0};
      if(rank==0){
      std::cout<<"niter: "<<niter<<std::endl;
      solves = niter*this->DataDD.nsub();
      std::cout<<"solves: "<<solves<<std::endl;
      time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
      std::cout <<"time in seconds: "<< time<<std::endl;
      }
      SolverResults res_obj(uw1,solves,time, this->traits_, DataDD);

      return res_obj;
    };
};



#endif
