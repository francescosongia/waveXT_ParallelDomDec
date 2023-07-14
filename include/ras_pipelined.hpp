#ifndef RAS_PIPELINED_HPP_
#define RAS_PIPELINED_HPP_

#include <utility>

#include "domaindec_solver_base.hpp"

template<class P,class LA>
class RasPipelined : public DomainDecSolverBase<P,LA> {
public:
  RasPipelined(Domain dom, const Decomposition& dec, LocalMatrices<LA> local_matrices,const SolverTraits& traits) : 
        //DomainDecSolverBase<P,LA>(dom,dec,local_matrices,traits), matrix_domain_(dec.nsub_t(),dec.nsub_x())
        DomainDecSolverBase<P,LA>(dom,dec,local_matrices,traits), matrix_domain_(dec.nsub_t(),dec.nsub_x())
        {
            Eigen::VectorXi list=Eigen::VectorXi::LinSpaced(dec.nsub(),1,dec.nsub());  
            matrix_domain_=list.transpose().reshaped(dec.nsub_t(),dec.nsub_x());  //serve traspose?
        };


  SolverResults solve(const SpMat& A, const SpMat& b) override
    {
        auto start = std::chrono::steady_clock::now();
        double tol=this->traits_.tol();
        //double tol_sx=traits.tol_pipe_sx();
        unsigned int max_it=this->traits_.max_it();
        //unsigned int it_wait=traits.it_wait();
        double res=tol+1;
        std::vector<double> resinf_vec;   //non è nemmeno relativo, ne precondz. Controllare.
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
        //MPI_Allreduce(MPI_IN_PLACE, &solves_, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);


        // dividere anche qui rank0 e altri. quindi servirà differenziare solve
        auto end = std::chrono::steady_clock::now();
        std::cout<<"niter: "<<niter<<std::endl;
        auto solves = this->traits_.solves();
        std::cout<<"solves: "<<solves<<std::endl;
        //std::cout<<"solves: "<<solves_<<std::endl;
        double time=std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        std::cout <<"time in seconds: "<< time<<std::endl;

        this->traits_.setZone(2);
        this->traits_.setSubtSx(1);
        this->traits_.setItWaited(0);
        this->traits_.setSolves(0);

        SolverResults res_obj(uw,solves,time, this->traits_, this->DataDD);

        return res_obj;
    };
  
protected:
    Eigen::MatrixXi matrix_domain_;


    Eigen::VectorXd precondAction(const SpMat& x) override
    {   
        P func_wrapper(this->domain,this->DataDD,this->local_mat,this->traits_);

        auto res=func_wrapper.precondAction(x); 

        this->traits_.setZone(func_wrapper.traits().zone());
        this->traits_.setSubtSx(func_wrapper.traits().subt_sx());
        this->traits_.setItWaited(func_wrapper.traits().it_waited());
        this->traits_.setSolves(func_wrapper.traits().solves());

        /*
        get traits in base
        fare i setters qui sopra
        in ogni funzione leggere i 4 param da traits
        la
        */

        return res;
    };


    unsigned int check_sx(const Eigen::VectorXd& v,double tol_sx)
    {
        P func_wrapper(this->domain,this->DataDD,this->local_mat,this->traits_);
        return func_wrapper.check_sx(v,tol_sx); 
    };

};


class PipeParallel_SeqLA : public RasPipelined<PipeParallel_SeqLA,SeqLA>
{
  public:
    PipeParallel_SeqLA(Domain dom, const Decomposition& dec,const LocalMatrices<SeqLA> local_matrices, const SolverTraits& traits) : 
    RasPipelined<PipeParallel_SeqLA,SeqLA>(dom,dec,local_matrices,traits) 
    { };

    Eigen::VectorXd precondAction(const SpMat& x)
    {
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      unsigned int zone_ = this->traits_.zone(); 
      unsigned int subt_sx_ = this->traits_.subt_sx(); 
      unsigned int it_waited_ = this->traits_.it_waited(); 
      unsigned int solves_ = this->traits_.solves(); 

      if(this->DataDD.nsub_x() % size != 0){
          std::cerr<<"sub_x not proportional to number of processes chosen"<<std::endl;
      }
      int partition = this->DataDD.nsub_x() / size;  

      Eigen::VectorXd z=Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
      double tol_sx=this->traits_.tol_pipe_sx();
      unsigned int it_wait=this->traits_.it_wait();

      unsigned int sx1= check_sx(x,tol_sx);
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
      // nsubt > = subt_sx+2 hypotesis

      unsigned int dx=subt_sx_+zone_;
    
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

      MPI_Allreduce(MPI_IN_PLACE, z.data(), this->domain.nln()*this->domain.nt()*this->domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);   
      

      this->traits_.setZone(zone_);
      this->traits_.setSolves(solves_);
      this->traits_.setItWaited(it_waited_);
      this->traits_.setSubtSx(subt_sx_);

      return z;

    };


    unsigned int check_sx(const Eigen::VectorXd& v,double tol_sx)
    {
      int np = this->local_mat.sub_assignment().np();
      int partition = this->DataDD.nsub_x() / np;  
      int rank = this->local_mat.rank();
      unsigned int subt_sx_ = this->traits_.subt_sx(); 
      //questi tre valori perche devo calcolarli sempre? non sarebbe meglio metterli come membri (anche privati)

      unsigned int sx1=subt_sx_;
      if (subt_sx_>this->DataDD.nsub_t()){
          std::cout<<subt_sx_<<std::endl;
          std::cerr<<"err in the definiton of left edge of subdomain window"<<std::endl;
          return 0;
      }
      int fail=0;
      Eigen::VectorXd res(this->domain.nln()*this->DataDD.sub_sizes()[0]*this->DataDD.sub_sizes()[1]*2);

      // posso far fare questo controllo un po ad ognuno dei rank, da defnire però l'intersezione tra i e quello che il rank puo vedere

      //for(size_t i=subt_sx_;i<=DataDD.nsub_t()*(DataDD.nsub_x()-1)+1;i+=DataDD.nsub_t()){
      for(size_t i=subt_sx_+this->DataDD.nsub_t()*partition*rank; i<=subt_sx_+this->DataDD.nsub_t()*partition*(rank +1)-1; i+=this->DataDD.nsub_t()){
          res=this->local_mat.getRk(i).first*v;
          auto err= res.lpNorm<Eigen::Infinity>();
          if(err>tol_sx){
              fail=1;
              break;
          }
      }
      //pensarlo come allrduce
      MPI_Allreduce(MPI_IN_PLACE, &fail, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
      if(fail==0)
          sx1++;
      return sx1;
    };

};


class PipeParallel_ParLA : public RasPipelined<PipeParallel_ParLA,ParLA>
{
  public:
    PipeParallel_ParLA(Domain dom, const Decomposition& dec,const LocalMatrices<ParLA> local_matrices, const SolverTraits& traits):
                         
    RasPipelined<PipeParallel_ParLA,ParLA>(dom,dec,local_matrices,traits) 
    {    };

    Eigen::VectorXd precondAction(const SpMat& x)
    {
      int rank{0},size{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      
      unsigned int zone_ = this->traits_.zone(); 
      unsigned int subt_sx_ = this->traits_.subt_sx(); 
      unsigned int it_waited_ = this->traits_.it_waited(); 
      unsigned int solves_ = this->traits_.solves(); 

      if(this->DataDD.nsub_x() % size != 0){
          std::cerr<<"sub_x not proportional to number of processes chosen"<<std::endl;
      }
      int partition = this->DataDD.nsub_x() / size;  

      Eigen::VectorXd z=Eigen::VectorXd::Zero(this->domain.nln()*this->domain.nt()*this->domain.nx()*2);
      double tol_sx=this->traits_.tol_pipe_sx();
      unsigned int it_wait=this->traits_.it_wait();

      unsigned int sx1= check_sx(x,tol_sx);
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
      // nsubt > = subt_sx+2 hypotesis

      unsigned int dx=subt_sx_+zone_;

      
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

      MPI_Allreduce(MPI_IN_PLACE, z.data(), this->domain.nln()*this->domain.nt()*this->domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);   

      this->traits_.setZone(zone_);
      this->traits_.setSolves(solves_);
      this->traits_.setItWaited(it_waited_);
      this->traits_.setSubtSx(subt_sx_);

      return z;

    };


    unsigned int check_sx(const Eigen::VectorXd& v,double tol_sx)
    {
      int np = this->local_mat.sub_assignment().np();
      int partition = this->DataDD.nsub_x() / np;  
      int rank = this->local_mat.rank();
      unsigned int subt_sx_ = this->traits_.subt_sx(); 
      //questi tre valori perche devo calcolarli sempre? non sarebbe meglio metterli come membri (anche privati)

      unsigned int sx1=subt_sx_;
      if (subt_sx_>this->DataDD.nsub_t()){
          std::cout<<subt_sx_<<std::endl;
          std::cerr<<"err in the definiton of left edge of subdomain window"<<std::endl;
          return 0;
      }
      int fail=0;
      Eigen::VectorXd res(this->domain.nln()*this->DataDD.sub_sizes()[0]*this->DataDD.sub_sizes()[1]*2);

      // posso far fare questo controllo un po ad ognuno dei rank, da defnire però l'intersezione tra i e quello che il rank puo vedere

      //for(size_t i=subt_sx_;i<=DataDD.nsub_t()*(DataDD.nsub_x()-1)+1;i+=DataDD.nsub_t()){
      for(size_t i=subt_sx_+this->DataDD.nsub_t()*partition*rank; i<=subt_sx_+this->DataDD.nsub_t()*partition*(rank +1)-1; i+=this->DataDD.nsub_t()){
          res=this->local_mat.getRk(i).first*v;
          auto err= res.lpNorm<Eigen::Infinity>();
          if(err>tol_sx){
              fail=1;
              break;
          }
      }
      //pensarlo come allrduce
      MPI_Allreduce(MPI_IN_PLACE, &fail, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
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

    Eigen::VectorXd precondAction(const SpMat& x) 
    {
        Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
        double tol_sx=this->traits_.tol_pipe_sx();
        unsigned int it_wait=this->traits_.it_wait();

        unsigned int zone_ = this->traits_.zone(); 
        unsigned int subt_sx_ = this->traits_.subt_sx(); 
        unsigned int it_waited_ = this->traits_.it_waited(); 
        unsigned int solves_ = this->traits_.solves(); 

        unsigned int sx1= check_sx(x,tol_sx);
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

    unsigned int check_sx(const Eigen::VectorXd& v, double tol_sx) 
    {
        unsigned int subt_sx_ = this->traits_.subt_sx(); 
        unsigned int sx1=subt_sx_;
        if (subt_sx_>DataDD.nsub_t()){
            std::cout<<subt_sx_<<std::endl;
            std::cerr<<"err in the definiton of left edge of subdomain window"<<std::endl;
            return 0;
        }
        unsigned int fail=0;
        Eigen::VectorXd res(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);
        // posso far fare questo controllo un po ad ognuno dei rank, da defnire però l'intersezione tra i e quello che il rank puo vedere
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



