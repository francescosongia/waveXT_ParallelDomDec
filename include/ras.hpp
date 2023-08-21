#ifndef RAS_HPP_
#define RAS_HPP_

#include "domaindec_solver_base.hpp"


template<class P, class LA>
class Ras : public DomainDecSolverBase<P,LA>{

public:
  Ras(Domain dom, const Decomposition& dec,const LocalMatrices<LA>& local_matrices,const SolverTraits& traits) : 
    DomainDecSolverBase<P,LA>(dom,dec,local_matrices,traits)
    {};

  SolverResults solve(const SpMat& A, const SpMat& b) override
  {
    // It creates an object P that represent a Policy and then call the solve method implemented in that Policy
    P func_wrapper(this->domain,this->DataDD,this->local_mat,this->traits_);
    return func_wrapper.solve(A,b);
  };
};

/*
--------------------------------------------------------------------------------
PARALLEL
### Parallel_AloneOnStride: sequential linear algebra but subs assigned to different processes
     precondAction
     solve

### Parallel_CooperationOnStride: parallel linear algebra and subs assigned to different processes
     precondAction
     solve

### Parallel_CooperationSplitTime: sequential linear algebra and subs assigned to different processes
                                   in both spatial and time direction
     precondAction
     solve
--------------------------------------------------------------------------------
SEQUENTIAL 
### Sequential: sequential linear algebra and subs assigned only to one process
     precondAction
     solve
--------------------------------------------------------------------------------
*/
 

class Parallel_AloneOnStride : public Ras<Parallel_AloneOnStride,AloneOnStride>
{
  public:
    Parallel_AloneOnStride(Domain dom, const Decomposition& dec,const LocalMatrices<AloneOnStride>& local_matrices,const SolverTraits& traits) :
     Ras<Parallel_AloneOnStride,AloneOnStride>(dom,dec,local_matrices,traits) {};

    Eigen::VectorXd 
    precondAction(const SpMat& x) 
    {
      // solve the local problems in the subdomains
      Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
      Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);

      // Get the subdomains assigned to the current rank
      auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
      for(unsigned int k : sub_division_vec){        
          auto temp = local_mat.getRk(k); // temp contains R_k and Rtilde_k
          uk = this->get_LU_k(k).solve(temp.first*x);
          z=z+(temp.second.transpose())*uk;
          }

      // Each rank compute the solution over the subdomains assigned and then collect and sum all the results with Allreduce
      MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return z;
    }


    SolverResults solve(const SpMat& A, const SpMat& b)
    {
      // uw_n+1 = uw_n + P^-1 * (b-A*uw_n)
      // where P^-1 is the sum over all subdomains k of (Rtilde_k' * A_k^-1 * R_k)
      // for convergence check the L2 normalized residual

      auto start = std::chrono::steady_clock::now();
      double tol = this->traits_.tol();
      unsigned int max_it = this->traits_.max_it();
      double res = tol+1;
      unsigned int niter = 0;
      double Pb2 = precondAction(b).norm();

      Eigen::VectorXd uw0 = Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
      Eigen::VectorXd uw1(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);

      Eigen::VectorXd z = precondAction(b);
      while(res>tol and niter<max_it){
          uw1 = uw0 + z;  
          z = precondAction(b - A*uw1);
          res = (z/Pb2).norm();
          niter++;
          uw0 = uw1;
      }

      auto end = std::chrono::steady_clock::now();
      int rank{0},size{0};

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      unsigned int solves{0};
      double time{0.0};
      if(rank == 0){
        std::cout<<"niter: "<<niter<<std::endl;
        solves = niter*this->DataDD.nsub();
        std::cout<<"solves: "<<solves<<std::endl;
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout <<"time in milliseconds: "<< time<<std::endl;
      }
      // return object with all informations about the procedure
      SolverResults res_obj(uw1,solves,time, this->traits_, DataDD);

      return res_obj;
    };
};


class Sequential : public Ras<Sequential,AloneOnStride>
{
public:
  Sequential(Domain dom, const Decomposition& dec,const LocalMatrices<AloneOnStride>& local_matrices,const SolverTraits& traits) :
   Ras<Sequential,AloneOnStride>(dom,dec,local_matrices,traits) {};
  
  Eigen::VectorXd precondAction(const SpMat& x) 
  {
    // solve the local problems in the subdomains

    Eigen::VectorXd z = Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
    for(unsigned int k = 1; k <= DataDD.nsub(); ++k){
        auto temp = local_mat.getRk(k);
        uk = this->get_LU_k(k).solve(temp.first*x);
        z = z + (temp.second.transpose())*uk;
    }
    return z;
  }

  SolverResults solve(const SpMat& A, const SpMat& b)
  {
    // uw_n+1 = uw_n + P^-1 * (b-A*uw_n)
    // where P^-1 is the sum over all subdomains k of (Rtilde_k' * A_k^-1 * R_k)
    // for convergence check the L2 normalized residual

    auto start = std::chrono::steady_clock::now();
    double tol = this->traits_.tol();
    unsigned int max_it = this->traits_.max_it();
    double res = tol+1;
    unsigned int niter = 0;
    double Pb2 = precondAction(b).norm();

    Eigen::VectorXd uw0 = Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
    Eigen::VectorXd uw1(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);

    Eigen::VectorXd z = precondAction(b);
    while(res>tol and niter<max_it){
        uw1 = uw0 + z;  
        z = precondAction(b-A*uw1);
        res = (z/Pb2).norm();
        niter++;
        uw0 = uw1;
    }

    auto end = std::chrono::steady_clock::now();
    std::cout<<"niter: "<<niter<<std::endl;
    unsigned int solves = niter*this->DataDD.nsub();
    std::cout<<"solves: "<<solves<<std::endl;
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout <<"time in milliseconds: "<< time<<std::endl;
    SolverResults res_obj(uw1,solves,time, this->traits_, this->DataDD);

    return res_obj;
  };

};



class Parallel_CooperationOnStride : public Ras<Parallel_CooperationOnStride,CooperationOnStride>
{
  private:
    int partition_;
    
    // vectors used to store local dimensions for parallel linear algebra computations
    std::vector<int> dim_local_res_vec1_;
    std::vector<int> dim_cum_vec1_;
    std::vector<int> dim_local_res_vec2_;
    std::vector<int> dim_cum_vec2_;

  public:
    Parallel_CooperationOnStride(Domain dom, const Decomposition& dec,const LocalMatrices<CooperationOnStride>& local_matrices,const SolverTraits& traits) :
     Ras<Parallel_CooperationOnStride,CooperationOnStride>(dom,dec,local_matrices,traits), 
     partition_( local_matrices.sub_assignment().np() /DataDD.nsub_x()), dim_local_res_vec1_(local_matrices.sub_assignment().np(),0),
     dim_cum_vec1_(local_matrices.sub_assignment().np(),0), dim_local_res_vec2_(local_matrices.sub_assignment().np(),0), 
     dim_cum_vec2_(local_matrices.sub_assignment().np(),0)
     {
      // compute vectors for local dimensions
      // more than one processes are assigned to the same time stride.
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      int dim_res{static_cast<int>(this->domain.nln()*this->domain.nt()*this->domain.nx()*2)};
      int dim_k{static_cast<int>(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2)};

      int start_for = (rank<partition_) ? rank : rank % partition_;

      for(int i=start_for; i<size; i+=this->DataDD.nsub_x()){
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
      // solve the local problems in the subdomains

      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      int dim_res{static_cast<int>(this->domain.nln()*this->domain.nt()*this->domain.nx()*2)};
      int dim_k{static_cast<int>(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2)};

      Eigen::VectorXd z=Eigen::VectorXd::Zero(dim_res);
      Eigen::VectorXd uk(dim_k);

      /*
      prod1: restriction over subk:            Rk * x   
      prod2: prolungation in the whole domain: Rtilde_k' * uk  
      */

      Eigen::VectorXd prod1(dim_k);  
      Eigen::VectorXd local_prod1(dim_local_res_vec1_[rank]);
      Eigen::VectorXd prod2(dim_res); 
      Eigen::VectorXd local_prod2(dim_local_res_vec2_[rank]);

      // Get the subdomains assigned to the current rank
      auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
      for(unsigned int k : sub_division_vec){    
          auto temp = local_mat.getRk(k);

          // prod1: temp.first*x
          local_prod1 =  temp.first.middleRows(dim_cum_vec1_[rank], dim_local_res_vec1_[rank])* x;

          // gathering the result in the main core of the time stride
          MPI_Gatherv (local_prod1.data(), dim_local_res_vec1_[rank], MPI_DOUBLE, prod1.data(), dim_local_res_vec1_.data(), 
                      dim_cum_vec1_.data() , MPI_DOUBLE, rank % partition_ , MPI_COMM_WORLD);
          
          // send the result also to the other processes of the time stride
          if(rank < partition_)
            for(int dest=rank+partition_; dest<size; dest+=partition_)
              MPI_Send (prod1.data(), dim_k, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);                                                  
          else
             MPI_Recv (prod1.data(), dim_k, MPI_DOUBLE, rank%partition_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          uk = this->get_LU_k(k).solve(prod1);
          
          // ------------------------------------------------------------------
          //prod2:  (temp.second.transpose())*uk
          local_prod2 =  temp.second.transpose().middleRows(dim_cum_vec2_[rank], dim_local_res_vec2_[rank])* uk;
          
          // gathering the result in the main core of the time stride
          MPI_Gatherv (local_prod2.data(), dim_local_res_vec2_[rank], MPI_DOUBLE, prod2.data(), dim_local_res_vec2_.data(), 
                      dim_cum_vec2_.data() , MPI_DOUBLE, rank % partition_ , MPI_COMM_WORLD);

          // send the result also to the other processes of the time stride
          if(rank < partition_)
            for(int dest=rank+partition_; dest<size; dest+=partition_)
              MPI_Send (prod2.data(), dim_res, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);                                                  
          else
             MPI_Recv (prod2.data(), dim_res, MPI_DOUBLE, rank%partition_, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          z = z + prod2;
          
          }

      // The main processes of each time stride collect and sum the results with Allreduce

      MPI_Comm master_comm; // communicator with one member for each temporal stride
      MPI_Comm workers_comm; // communicator with all helpers cores

      int color,color2;
      color = (rank < DataDD.nsub_x()) ? 0 : 1;
      color2 = (rank >= DataDD.nsub_x() || rank == 0) ? 0 : 1;

      MPI_Comm_split(MPI_COMM_WORLD, color, rank, &master_comm);
      MPI_Comm_split(MPI_COMM_WORLD, color2, rank, &workers_comm);
      // master ranks sums the result between them
      if(color==0)
        MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, master_comm);
      // then rank0 communicate the final result to all others ranks
      if (color2 == 0)
        MPI_Bcast(z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, 0, workers_comm);

      MPI_Comm_free(&master_comm);
      MPI_Comm_free(&workers_comm);

      return z;
    }

    SolverResults solve(const SpMat& A, const SpMat& b)
    {
      // uw_n+1 = uw_n + P^-1 * (b-A*uw_n)
      // where P^-1 is the sum over all subdomains k of (Rtilde_k' * A_k^-1 * R_k)
      // for convergence check the L2 normalized residual

      // in this function the computation of the residual is parallelized among all the processes

      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      auto start = std::chrono::steady_clock::now();
      double tol = this->traits_.tol();
      unsigned int max_it = this->traits_.max_it();
      double res = tol+1;
      unsigned int niter = 0;
      double Pb2 = precondAction(b).norm();

      int dim_res{static_cast<int>(this->domain.nln()*this->domain.nt()*this->domain.nx()*2)};

      
      std::vector<int> dim_local_res_vec(size,0);
      std::vector<int> dim_cum_vec(size,0);
      // compute the local dimension assigned to each process
      for(int i = 0; i<size; ++i){
        dim_local_res_vec[i] = dim_res/size + (i < (dim_res % size));
        if(i!=0)
          dim_cum_vec[i] = dim_local_res_vec[i-1] + dim_cum_vec[i-1];
      }
      

      Eigen::VectorXd uw0=Eigen::VectorXd::Zero(dim_res);
      Eigen::VectorXd uw1(dim_res);

      Eigen::VectorXd prod(dim_res);
      Eigen::VectorXd local_prod(dim_local_res_vec[rank]);

      Eigen::VectorXd z = precondAction(b);
      while(res>tol and niter<max_it){
          uw1=uw0+z; 

          // parallelize A*uw1
          //compute local_prod          
          local_prod =  A.middleRows(dim_cum_vec[rank], dim_local_res_vec[rank])* uw1;

          //gatherv di prod
          MPI_Allgatherv (local_prod.data(), dim_local_res_vec[rank], MPI_DOUBLE, prod.data(), dim_local_res_vec.data(), 
                      dim_cum_vec.data(), MPI_DOUBLE, MPI_COMM_WORLD);

          z = precondAction(b-prod);          
          res = (z/Pb2).norm();
          niter++;
          uw0 = uw1;
      }

      auto end = std::chrono::steady_clock::now();
      
      unsigned int solves{0};
      double time{0.0};
      if(rank==0){
        std::cout<<"niter: "<<niter<<std::endl;
        solves = niter*this->DataDD.nsub();
        std::cout<<"solves: "<<solves<<std::endl;
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout <<"time in milliseconds: "<< time<<std::endl;
      }
      SolverResults res_obj(uw1,solves,time, this->traits_, DataDD);

      return res_obj;
    };
};



class Parallel_CooperationSplitTime : public Ras<Parallel_CooperationSplitTime,CooperationSplitTime>
{
  public:
    Parallel_CooperationSplitTime(Domain dom, const Decomposition& dec,const LocalMatrices<CooperationSplitTime>& local_matrices,const SolverTraits& traits) :
     Ras<Parallel_CooperationSplitTime,CooperationSplitTime>(dom,dec,local_matrices,traits) {};

    Eigen::VectorXd 
    precondAction(const SpMat& x) 
    {
      // solve the local problems in the subdomains
      Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
      Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
      int rank{0};

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      // Get the subdomains assigned to the current rank
      auto sub_division_vec = local_mat.sub_assignment().sub_division_vec()[local_mat.rank()];
  
      for(unsigned int k : sub_division_vec){     
          auto temp = local_mat.getRk(k); // temp contains R_k and Rtilde_k
          uk = this->get_LU_k(k).solve(temp.first*x);
          z=z+(temp.second.transpose())*uk;
          }

      // Each rank compute the solution over the subdomains assigned and then collect and sum all the results with Allreduce
      MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return z;
    }


    SolverResults solve(const SpMat& A, const SpMat& b)
    {
      // uw_n+1 = uw_n + P^-1 * (b-A*uw_n)
      // where P^-1 is the sum over all subdomains k of (Rtilde_k' * A_k^-1 * R_k)
      // for convergence check the L2 normalized residual
      int rank{0},size{0};

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      auto start = std::chrono::steady_clock::now();
      double tol = this->traits_.tol();
      unsigned int max_it = this->traits_.max_it();
      double res = tol+1;
      unsigned int niter = 0;
      double Pb2 = precondAction(b).norm();

      Eigen::VectorXd uw0 = Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
      Eigen::VectorXd uw1(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);

      Eigen::VectorXd z = precondAction(b);
      while(res>tol and niter<max_it){
          uw1 = uw0 + z; 
          z = precondAction(b - A*uw1);
          res = (z/Pb2).norm();
          niter++;
          uw0 = uw1;
      }

      auto end = std::chrono::steady_clock::now();
      
      unsigned int solves{0};
      double time{0.0};
      if(rank == 0){
        std::cout<<"niter: "<<niter<<std::endl;
        solves = niter*this->DataDD.nsub();
        std::cout<<"solves: "<<solves<<std::endl;
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout <<"time in milliseconds: "<< time<<std::endl;
      }
      // return object with all informations about the procedure
      SolverResults res_obj(uw1,solves,time, this->traits_, DataDD);

      return res_obj;
    };
};




#endif
