#ifndef RAS_HPP_
#define RAS_HPP_

#include "domaindec_solver_base.hpp"


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
  };

};


// --------------------------------------------------------------------------------
// PARALLEL
//### Parallel_SeqLA: sequential linear algebra but subs assigned to different processes
//      precondAction
//      solve

//### Parallel_ParLA: parallel linear algebra and subs assigned to different processes
//      precondAction
//      solve
// --------------------------------------------------------------------------------
// SEQUENTIAL 
//### Sequential: sequential linear algebra and subs assigned only to one process
//      precondAction
//      solve
// --------------------------------------------------------------------------------
 

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
  private:
    int partition_;
    
    // vectors used to store local dimensions for parallel linear algebra computations
    std::vector<int> dim_local_res_vec1_;
    std::vector<int> dim_cum_vec1_;
    std::vector<int> dim_local_res_vec2_;
    std::vector<int> dim_cum_vec2_;

  public:
    Parallel_ParLA(Domain dom, const Decomposition& dec,const LocalMatrices<ParLA> local_matrices,const SolverTraits& traits) :
     Ras<Parallel_ParLA,ParLA>(dom,dec,local_matrices,traits), 
     partition_( local_matrices.sub_assignment().np() /DataDD.nsub_x()), dim_local_res_vec1_(local_matrices.sub_assignment().np(),0),
     dim_cum_vec1_(local_matrices.sub_assignment().np(),0), dim_local_res_vec2_(local_matrices.sub_assignment().np(),0), 
     dim_cum_vec2_(local_matrices.sub_assignment().np(),0)
     {
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      int dim_res{static_cast<int>(this->domain.nln()*this->domain.nt()*this->domain.nx()*2)};
      int dim_k{static_cast<int>(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2)};

      int start_for = (rank<partition_) ? rank : rank % partition_;

      for(int i=start_for; i<size; i+=this->DataDD.nsub_x()){
        // ogni process della partition sa le dimensione sue e del suo aiutante
        dim_local_res_vec1_[i] = dim_k/partition_ + (i< (dim_k % partition_));
        dim_local_res_vec2_[i] = dim_res/partition_ + (i< (dim_res % partition_));
      }   
      for(int i=1; i<size;++i){
        dim_cum_vec1_[i] = dim_local_res_vec1_[i-1] + dim_cum_vec1_[i-1];
        dim_cum_vec2_[i] = dim_local_res_vec2_[i-1] + dim_cum_vec2_[i-1];
      }
     };

    Eigen::VectorXd 
    precondAction(const SpMat& x) 
    {
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      int dim_res{static_cast<int>(this->domain.nln()*this->domain.nt()*this->domain.nx()*2)};
      int dim_k{static_cast<int>(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2)};


      Eigen::VectorXd z=Eigen::VectorXd::Zero(dim_res);
      Eigen::VectorXd zero=Eigen::VectorXd::Zero(dim_res);
      Eigen::VectorXd uk(dim_k);

      /*
      Rk*x   x: dim_res -> dimk
      R'*uk  dimk-> dimres
      */

      Eigen::VectorXd prod1(dim_k);  //Rk*x
      Eigen::VectorXd local_prod1(dim_local_res_vec1_[rank]);
      Eigen::VectorXd prod2(dim_res);  //Rtilde'*uk
      Eigen::VectorXd local_prod2(dim_local_res_vec2_[rank]);

      // questo for devo prendere i k giusti che vede quel core, mi appoggio su sub_assingment
      auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
      for(unsigned int k : sub_division_vec){    
          Eigen::SparseLU<SpMat > lu;
          lu.compute(local_mat.getAk(k));
          auto temp = local_mat.getRk(k);


          //prod1 temp.first*x
          local_prod1 =  temp.first.middleRows(dim_cum_vec1_[rank], dim_local_res_vec1_[rank])* x;
      
          MPI_Gatherv (local_prod1.data(), dim_local_res_vec1_[rank], MPI_DOUBLE, prod1.data(), dim_local_res_vec1_.data(), 
                      dim_cum_vec1_.data() , MPI_DOUBLE, rank % partition_ , MPI_COMM_WORLD);
          
          if(rank < partition_)
            for(int dest=rank+partition_; dest<size; dest+=partition_)
              MPI_Send (prod1.data(), dim_k, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);                                                  
          else
             MPI_Recv (prod1.data(), dim_k, MPI_DOUBLE, rank%partition_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          uk = lu.solve(prod1);

          //prod2  (temp.second.transpose())*uk
          local_prod2 =  temp.second.transpose().middleRows(dim_cum_vec2_[rank], dim_local_res_vec2_[rank])* uk;
      
          MPI_Gatherv (local_prod2.data(), dim_local_res_vec2_[rank], MPI_DOUBLE, prod2.data(), dim_local_res_vec2_.data(), 
                      dim_cum_vec2_.data() , MPI_DOUBLE, rank % partition_ , MPI_COMM_WORLD);
          
          if(rank < partition_)
            for(int dest=rank+partition_; dest<size; dest+=partition_)
              MPI_Send (prod2.data(), dim_res, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);                                                  
          else
             MPI_Recv (prod2.data(), dim_res, MPI_DOUBLE, rank%partition_, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          z=z+prod2;
          
          }

      if(rank<this->partition_)
        MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      else
        MPI_Allreduce(zero.data(), z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // zero è brutto, in alternativa fare all reduce + waitall per far conoscere a 0 e 1, e poi fare bcast per far conoscere 2 3

      return z;
    }

    SolverResults solve(const SpMat& A, const SpMat& b)
    {
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      auto start = std::chrono::steady_clock::now();
      double tol=this->traits_.tol();
      unsigned int max_it=this->traits_.max_it();
      double res=tol+1;
      std::vector<double> relres2P_vec;
      unsigned int niter=0;
      double Pb2=precondAction(b).norm();

      int dim_res{static_cast<int>(this->domain.nln()*this->domain.nt()*this->domain.nx()*2)};

      
      std::vector<int> dim_local_res_vec(size,0);
      std::vector<int> dim_cum_vec(size,0);

      for(int i=0;i<size;++i){
        dim_local_res_vec[i] = dim_res/size + (i< (dim_res % size));
        if(i!=0)
          dim_cum_vec[i] = dim_local_res_vec[i-1] + dim_cum_vec[i-1];
      }
      
      Eigen::VectorXd uw0=Eigen::VectorXd::Zero(dim_res);
      Eigen::VectorXd uw1(dim_res);

      Eigen::VectorXd prod(dim_res);
      Eigen::VectorXd local_prod(dim_local_res_vec[rank]);

      Eigen::VectorXd z= precondAction(b); //b-A*uw0
      while(res>tol and niter<max_it){
          uw1=uw0+z;  //forse si puo evitare l'uso di entrambi uw0 e uw1

          // parallelo A*uw1
          //popoliamo local_prod          
          local_prod =  A.middleRows(dim_cum_vec[rank], dim_local_res_vec[rank])* uw1;

          //gatherv di prod
          MPI_Allgatherv (local_prod.data(), dim_local_res_vec[rank], MPI_DOUBLE, prod.data(), dim_local_res_vec.data(), 
                      dim_cum_vec.data(), MPI_DOUBLE, MPI_COMM_WORLD);

          z= precondAction(b-prod);          
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
