#ifndef RAS_PIPELINED_HPP_
#define RAS_PIPELINED_HPP_

#include <utility>

#include "domaindec_solver_base.hpp"

template<class P,class LA>
class RasPipelined : public DomainDecSolverBase<P,LA> {
public:
  RasPipelined(Domain dom, const Decomposition& dec, LocalMatrices<LA> local_matrices,const SolverTraits& traits) : 
        DomainDecSolverBase<P,LA>(dom,dec,local_matrices,traits), matrix_domain_(dec.nsub_t(),dec.nsub_x())
        {   
            // matrix_domain_ is a mtrix representing the domain, it will support the computations
            Eigen::VectorXi list=Eigen::VectorXi::LinSpaced(dec.nsub(),1,dec.nsub());  
            matrix_domain_=list.transpose().reshaped(dec.nsub_t(),dec.nsub_x());  
        };


  SolverResults solve(const SpMat& A, const SpMat& b) override
    {
         // uw_n+1 = uw_n + P^-1 * (b-A*uw_n)
        // where P^-1 is the sum (over all subdomains k inside the zone) of (Rtilde_k' * A_k^-1 * R_k)
        // for convergence check the L_Inf residual

        // initialize parameters for zone evolution 
        this->traits_.setZone(2);
        this->traits_.setSubtSx(1);
        this->traits_.setItWaited(0);
        this->traits_.setSolves(0);

        P func_wrapper(this->domain,this->DataDD,this->local_mat,this->traits_);
        SolverResults res_obj = func_wrapper.solve(A,b);

        return res_obj;
    };
  
protected:
    Eigen::MatrixXi matrix_domain_;


    Eigen::VectorXd precondAction(const SpMat& x) override
    {   
        // update the zone and then solve the local problems in the subdomains
        P func_wrapper(this->domain,this->DataDD,this->local_mat,this->traits_);
        auto res=func_wrapper.precondAction(x); 
        
        // update parameters for zone evolution 
        this->traits_.setZone(func_wrapper.traits().zone());
        this->traits_.setSubtSx(func_wrapper.traits().subt_sx());
        this->traits_.setItWaited(func_wrapper.traits().it_waited());
        this->traits_.setSolves(func_wrapper.traits().solves());

        return res;
    };


    unsigned int check_sx(const Eigen::VectorXd& v)
    {
        // check if the left side of the zone has reached convergence
        // and it returns the left element of the zone (of the first time stride)
        P func_wrapper(this->domain,this->DataDD,this->local_mat,this->traits_);
        return func_wrapper.check_sx(v); 
    };

};

// --------------------------------------------------------------------------------
// PARALLEL
//### PipeParallel_SeqLA: sequential linear algebra but subs assigned to different processes
//      precondAction
//      check_sx
//      solve

//### PipeParallel_ParLA: parallel linear algebra and subs assigned to different processes
//      precondAction
//      check_sx
//      solve
// --------------------------------------------------------------------------------
// SEQUENTIAL 
//### PipeSequential: sequential linear algebra and subs assigned only to one process
//      precondAction
//      check_sx
//      solve
// --------------------------------------------------------------------------------
 

class PipeParallel_SeqLA : public RasPipelined<PipeParallel_SeqLA,SeqLA>
{
  public:
    PipeParallel_SeqLA(Domain dom, const Decomposition& dec,const LocalMatrices<SeqLA> local_matrices, const SolverTraits& traits) : 
    RasPipelined<PipeParallel_SeqLA,SeqLA>(dom,dec,local_matrices,traits) 
    { };
    
    SolverResults solve(const SpMat& A, const SpMat& b) override
    {   
        int rank{0};
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto start = std::chrono::steady_clock::now();
        double tol=this->traits_.tol();
        unsigned int max_it=this->traits_.max_it();
        double res=tol+1;
        std::vector<double> resinf_vec;   
        unsigned int niter=0;

        Eigen::VectorXd uw=Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
        Eigen::VectorXd v(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);

        while(res>tol and niter<max_it){
            v=b-A*uw;
            res=v.lpNorm<Eigen::Infinity>();
            uw=uw+precondAction(b-A*uw);
            resinf_vec.push_back(res);
            niter++;

        }
        unsigned int solves = this->traits_.solves();
        MPI_Allreduce(MPI_IN_PLACE, &solves, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

        auto end = std::chrono::steady_clock::now();
        double time=std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        if(rank==0){
            std::cout<<"niter: "<<niter<<std::endl;
            std::cout<<"solves: "<<solves<<std::endl;
            std::cout <<"time in seconds: "<< time<<std::endl;
        }

        SolverResults res_obj(uw,solves,time, this->traits_, this->DataDD);

        return res_obj;
    };


    Eigen::VectorXd precondAction(const SpMat& x)
    {
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      // collect the parameters for the zone evolution
      unsigned int zone_ = this->traits_.zone(); 
      unsigned int subt_sx_ = this->traits_.subt_sx(); 
      unsigned int it_waited_ = this->traits_.it_waited(); 
      unsigned int solves_ = this->traits_.solves(); 

      if(this->DataDD.nsub_x() % size != 0){
          std::cerr<<"sub_x not proportional to number of processes chosen"<<std::endl;
      }
      int partition = this->DataDD.nsub_x() / size;  

      Eigen::VectorXd z = Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);

      // --------- ZONE UPDATE -----------------------------------------------------
      unsigned int it_wait = this->traits_.it_wait();
      unsigned int sx1= check_sx(x);
      unsigned int f=0;
      if (sx1 >subt_sx_ and sx1<=this->DataDD.nsub_t()){
          subt_sx_=sx1;
          zone_--;
          f=1;
      }
      unsigned int isend = (zone_+subt_sx_ == this->DataDD.nsub_t()+1) ? 1 : 0;
      if (isend==0 and zone_==0)
          zone_++;
      //update dx
      if(isend==0){
          if(it_waited_>=it_wait or (f==1 and it_waited_<it_wait)){
              zone_ = zone_ +1 ;
              it_waited_=0;
          }
          else if(f==0 and it_waited_<it_wait){
              it_waited_++;
          }
      } 
      // nsubt > = subt_sx+2 hypotesis   //INSERIRE QUESTO CHECK?

      unsigned int dx=subt_sx_+zone_;
      // -------------------------------------------------------------------------

      auto zonematrix=matrix_domain_(Eigen::seq(subt_sx_-1,dx-2),Eigen::seq(rank*partition, (rank+1)*partition-1));
      Eigen::VectorXi sub_in_zone=zonematrix.reshaped();
      solves_+=sub_in_zone.size();

      Eigen::VectorXd uk(this->domain.nln()*this->DataDD.sub_sizes()[0]*this->DataDD.sub_sizes()[1]*2);

      for(unsigned int k:sub_in_zone){
          Eigen::SparseLU<SpMat > lu;
          lu.compute(this->local_mat.getAk(k));
          auto temp = this->local_mat.getRk(k);
          uk = lu.solve(temp.first*x);
          z=z+(temp.second.transpose())*uk;
      }
      // Each rank compute the solution over the subdomains assigned and then collect and sum all the results with Allreduce
      MPI_Allreduce(MPI_IN_PLACE, z.data(), this->domain.nln()*this->domain.nt()*this->domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);   
      
      // update the zone evolution parameters
      this->traits_.setZone(zone_);
      this->traits_.setSolves(solves_);
      this->traits_.setItWaited(it_waited_);
      this->traits_.setSubtSx(subt_sx_);

      return z;

    };


    unsigned int check_sx(const Eigen::VectorXd& v)
    {
      // it returns the left element of the zone (of the first time stride)
      int np = this->local_mat.sub_assignment().np();
      int partition = this->DataDD.nsub_x() / np;  
      int rank = this->local_mat.rank();
      unsigned int subt_sx_ = this->traits_.subt_sx(); 
      double tol_sx = this->traits_.tol_pipe_sx(); 

      unsigned int sx1 = subt_sx_;
      if (subt_sx_>this->DataDD.nsub_t()){
          std::cout<<subt_sx_<<std::endl;
          std::cerr<<"err in the definiton of left edge of subdomain window"<<std::endl;
          return 0;
      }
      int fail = 0;
      Eigen::VectorXd res(this->domain.nln()*this->DataDD.sub_sizes()[0]*this->DataDD.sub_sizes()[1]*2);

      for(size_t i=subt_sx_+this->DataDD.nsub_t()*partition*rank; i<=subt_sx_+this->DataDD.nsub_t()*partition*(rank +1)-1; i+=this->DataDD.nsub_t()){
          res=this->local_mat.getRk(i).first*v;
          auto err= res.lpNorm<Eigen::Infinity>();
          if(err > tol_sx){
              fail = 1;
              break;
          }
      }
      // Each rank checks the subdomains assigned and then intersect the results with Allreduce
      MPI_Allreduce(MPI_IN_PLACE, &fail, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
      if(fail==0)
          sx1++;
    
      return sx1;
    };

};


class PipeParallel_ParLA : public RasPipelined<PipeParallel_ParLA,ParLA>
{
  private:
    int partition_;

    // vectors used to store local dimensions for parallel linear algebra computations
    std::vector<int> dim_local_res_vec1_;
    std::vector<int> dim_cum_vec1_;
    std::vector<int> dim_local_res_vec2_;
    std::vector<int> dim_cum_vec2_;

  public:
    PipeParallel_ParLA(Domain dom, const Decomposition& dec,const LocalMatrices<ParLA> local_matrices, const SolverTraits& traits):
                         
    RasPipelined<PipeParallel_ParLA,ParLA>(dom,dec,local_matrices,traits),
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


    SolverResults solve(const SpMat& A, const SpMat& b) override
    {
        // in this function the computation of the residual is parallelized among all the processes

        int rank{0},size{0};
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        auto start = std::chrono::steady_clock::now();
        double tol=this->traits_.tol();
        unsigned int max_it=this->traits_.max_it();
        double res=tol+1;
        std::vector<double> resinf_vec;   
        unsigned int niter=0;

        int dim_res{static_cast<int>(this->domain.nln()*this->domain.nt()*this->domain.nx()*2)};

        Eigen::VectorXd uw=Eigen::VectorXd::Zero(dim_res);
        Eigen::VectorXd v(dim_res);

        // compute the local dimension assigned to each process
        std::vector<int> dim_local_res_vec(size,0);
        std::vector<int> dim_cum_vec(size,0);

        for(int i=0;i<size;++i){
            dim_local_res_vec[i] = dim_res/size + (i< (dim_res % size));
            if(i!=0)
            dim_cum_vec[i] = dim_local_res_vec[i-1] + dim_cum_vec[i-1];
        }

        Eigen::VectorXd prod(dim_res);
        Eigen::VectorXd local_prod(dim_local_res_vec[rank]);

        while(res>tol and niter<max_it){
            // parallelize A*uw1
            //compute local_prod           
            local_prod =  A.middleRows(dim_cum_vec[rank], dim_local_res_vec[rank])* uw;

            //gatherv di prod
            MPI_Allgatherv (local_prod.data(), dim_local_res_vec[rank], MPI_DOUBLE, prod.data(), dim_local_res_vec.data(), 
                      dim_cum_vec.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            v=b-prod;
            res=v.lpNorm<Eigen::Infinity>();
            uw=uw+precondAction(b-prod);
            resinf_vec.push_back(res);
            niter++;

        }
        unsigned int solves = this->traits_.solves();
        MPI_Allreduce(MPI_IN_PLACE, &solves, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

        auto end = std::chrono::steady_clock::now();
        double time=std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        if(rank==0){
            std::cout<<"niter: "<<niter<<std::endl;
            std::cout<<"solves: "<<solves<<std::endl;
            std::cout <<"time in seconds: "<< time<<std::endl;
        }

        SolverResults res_obj(uw,solves,time, this->traits_, this->DataDD);

        return res_obj;
    };

    Eigen::VectorXd precondAction(const SpMat& x)
    {
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      
      unsigned int zone_ = this->traits_.zone(); 
      unsigned int subt_sx_ = this->traits_.subt_sx(); 
      unsigned int it_waited_ = this->traits_.it_waited(); 
      unsigned int solves_ = this->traits_.solves(); 
      
      int dim_res{static_cast<int>(this->domain.nln()*this->domain.nt()*this->domain.nx()*2)};
      int dim_k{static_cast<int>(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2)};

      Eigen::VectorXd z=Eigen::VectorXd::Zero(dim_res);
      Eigen::VectorXd zero=Eigen::VectorXd::Zero(dim_res);
      unsigned int it_wait=this->traits_.it_wait();

      // --------- ZONE UPDATE -----------------------------------------------------
      unsigned int sx1= check_sx(x);
      unsigned int f=0;
      if (sx1 >subt_sx_ and sx1<=this->DataDD.nsub_t()){
          subt_sx_=sx1;
          zone_--;
          f=1;
      }
      unsigned int isend= (zone_+subt_sx_ == this->DataDD.nsub_t()+1) ? 1 : 0;
      if (isend==0 and zone_==0)
          zone_++;
      //update dx
      if(isend==0){
          if(it_waited_>=it_wait or (f==1 and it_waited_<it_wait)){
              zone_++;
              it_waited_=0;
          }
          else if(f==0 and it_waited_<it_wait)
              it_waited_++;
      } 
      

      unsigned int dx=subt_sx_+zone_;
      // ---------------------------------------------------------------------------

      auto zonematrix=matrix_domain_(Eigen::seq(subt_sx_-1,dx-2),Eigen::seq(rank%partition_,rank%partition_));//Eigen::seq(i_group_la*partition_, (i_group_la+1)*partition_-partition_));
      Eigen::VectorXi sub_in_zone=zonematrix.reshaped();
      solves_+=sub_in_zone.size();

      Eigen::VectorXd uk(dim_k);

      /*
      prod1: restriction over subk:            Rk * x   
      prod2: prolungation in the whole domain: Rtilde_k' * uk  
      */
 
      Eigen::VectorXd prod1(dim_k);  
      Eigen::VectorXd local_prod1(dim_local_res_vec1_[rank]);
      Eigen::VectorXd prod2(dim_res); 
      Eigen::VectorXd local_prod2(dim_local_res_vec2_[rank]);

      for(unsigned int k:sub_in_zone){
          Eigen::SparseLU<SpMat > lu;
          lu.compute(this->local_mat.getAk(k));
          auto temp = this->local_mat.getRk(k);

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

          uk = lu.solve(prod1);

          // ------------------------------------------------------------------
          //prod2  (temp.second.transpose())*uk
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

          z = z+prod2;
      }

      // The main processes of each time stride collect and sum the results with Allreduce
      if(rank<this->partition_)
        MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      else
        MPI_Allreduce(zero.data(), z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


      this->traits_.setZone(zone_);
      this->traits_.setSolves(solves_);
      this->traits_.setItWaited(it_waited_);
      this->traits_.setSubtSx(subt_sx_);

      return z;

    };


    unsigned int check_sx(const Eigen::VectorXd& v)
    {
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      int dim_k{static_cast<int>(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2)};
      unsigned int subt_sx_ = this->traits_.subt_sx(); 
      double tol_sx = this->traits_.tol_pipe_sx(); 

      unsigned int sx1=subt_sx_;
      if (subt_sx_>this->DataDD.nsub_t()){
          std::cout<<subt_sx_<<std::endl;
          std::cerr<<"err in the definiton of left edge of subdomain window"<<std::endl;
          return 0;
      }
      int fail=0;
      Eigen::VectorXd res(dim_k);
      Eigen::VectorXd local_prod3(dim_local_res_vec1_[rank]);

      auto zonematrix=matrix_domain_(Eigen::seq(subt_sx_-1,subt_sx_-1),Eigen::seq(rank%partition_,rank%partition_));
      Eigen::VectorXi subsx_in_zone=zonematrix.reshaped();  

      for(unsigned int i: subsx_in_zone){
          // parallelize the computation of the local residual
          local_prod3 =  this->local_mat.getRk(i).first.middleRows(dim_cum_vec1_[rank], dim_local_res_vec1_[rank])* v;
      
          MPI_Gatherv (local_prod3.data(), dim_local_res_vec1_[rank], MPI_DOUBLE, res.data(), dim_local_res_vec1_.data(), 
                      dim_cum_vec1_.data() , MPI_DOUBLE, rank % partition_ , MPI_COMM_WORLD);
          
          if(rank < partition_)
            for(int dest=rank+partition_; dest<size; dest+=partition_)
              MPI_Send (res.data(), dim_k, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);                                                  
          else
             MPI_Recv (res.data(), dim_k, MPI_DOUBLE, rank%partition_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


          auto err= res.lpNorm<Eigen::Infinity>();
          if(err>tol_sx){
              fail=1;
              break;
          }
      }
    
      // The main processes of each time stride check the subdomains assigned and then intersect the results with Allreduce
      int base{1};
      if(rank<this->partition_)
        MPI_Allreduce(MPI_IN_PLACE, &fail, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
      else
        MPI_Allreduce(&base, &fail, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

      if(fail==0)
          sx1++;
      return sx1;
    };

};


class PipeSequential : public RasPipelined<PipeSequential,SeqLA>
{
  public:
    PipeSequential(Domain dom, const Decomposition& dec,const LocalMatrices<SeqLA> local_matrices,const SolverTraits& traits):
     RasPipelined<PipeSequential,SeqLA>(dom,dec,local_matrices,traits)
      { };


    SolverResults solve(const SpMat& A, const SpMat& b) override
    {
        
        auto start = std::chrono::steady_clock::now();
        double tol=this->traits_.tol();
        unsigned int max_it=this->traits_.max_it();
        double res=tol+1;
        std::vector<double> resinf_vec; 
        unsigned int niter=0;

        Eigen::VectorXd uw=Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
        Eigen::VectorXd v(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);

        while(res>tol and niter<max_it){
            v=b-A*uw;
            res=v.lpNorm<Eigen::Infinity>();
            uw=uw+precondAction(b-A*uw);
            resinf_vec.push_back(res);
            niter++;

        }
        unsigned int solves = this->traits_.solves();

        auto end = std::chrono::steady_clock::now();
        double time=std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        std::cout<<"niter: "<<niter<<std::endl;
        std::cout<<"solves: "<<solves<<std::endl;
        std::cout <<"time in seconds: "<< time<<std::endl;
        

        SolverResults res_obj(uw,solves,time, this->traits_, this->DataDD);

        return res_obj;
    };

    Eigen::VectorXd precondAction(const SpMat& x) 
    {
        Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
        unsigned int it_wait=this->traits_.it_wait();

        unsigned int zone_ = this->traits_.zone(); 
        unsigned int subt_sx_ = this->traits_.subt_sx(); 
        unsigned int it_waited_ = this->traits_.it_waited(); 
        unsigned int solves_ = this->traits_.solves(); 

        // --------- ZONE UPDATE -----------------------------------------------------
        unsigned int sx1= check_sx(x);
        unsigned int f=0;
        if (sx1 >subt_sx_ and sx1<=DataDD.nsub_t()){
            subt_sx_=sx1;
            zone_--;
            f=1;
        }
        unsigned int isend= (zone_+subt_sx_ == DataDD.nsub_t()+1) ? 1 : 0;
        if (isend==0 and zone_==0)
            zone_++;

        //update dx
        if(isend==0){
            if(it_waited_>=it_wait or (f==1 and it_waited_<it_wait)){
                zone_++;
                it_waited_=0;
            }
            else if(f==0 and it_waited_<it_wait)
                it_waited_++;
        } 
        // nsubt > = subt_sx+2 hypotesis

        unsigned int dx=subt_sx_+zone_;

        // -----------------------------------------------------------------------
        auto zonematrix=matrix_domain_(Eigen::seq(subt_sx_-1,dx-2),Eigen::seq(0,matrix_domain_.cols()-1));  
        Eigen::VectorXi sub_in_zone=zonematrix.reshaped();
        solves_+=sub_in_zone.size();

        Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);

        for(unsigned int k:sub_in_zone){
            Eigen::SparseLU<SpMat > lu;
            lu.compute(local_mat.getAk(k));
            auto temp = local_mat.getRk(k);
            uk = lu.solve(temp.first*x);
            z=z+(temp.second.transpose())*uk;

        }

        this->traits_.setZone(zone_);
        this->traits_.setSolves(solves_);
        this->traits_.setItWaited(it_waited_);
        this->traits_.setSubtSx(subt_sx_);

        return z;
    }

    unsigned int check_sx(const Eigen::VectorXd& v) 
    {
        unsigned int subt_sx_ = this->traits_.subt_sx(); 
        double tol_sx = this->traits_.tol_pipe_sx(); 
        unsigned int sx1=subt_sx_;
        if (subt_sx_>DataDD.nsub_t()){
            std::cout<<subt_sx_<<std::endl;
            std::cerr<<"err in the definiton of left edge of subdomain window"<<std::endl;
            return 0;
        }
        unsigned int fail=0;
        Eigen::VectorXd res(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
        for(size_t i=subt_sx_;i<=DataDD.nsub_t()*(DataDD.nsub_x()-1)+1;i+=DataDD.nsub_t()){
            res=local_mat.getRk(i).first*v;
            auto err= res.lpNorm<Eigen::Infinity>();
            if(err>tol_sx){
                fail=1;
                break;
            }
        }
        if(fail==0)
            sx1++;
        return sx1;
    }


};


#endif



