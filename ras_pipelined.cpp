#include <iostream>
#include <chrono>
#include "ras_pipelined.hpp"

Eigen::VectorXd RasPipelined::precondAction(const SpMat& x,SolverTraits traits) {
    Eigen::VectorXd z=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    double tol_sx=traits.tol_pipe_sx();
    unsigned int it_wait=traits.it_wait();

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

    //set subs in the window
    Eigen::VectorXi list=Eigen::VectorXi::LinSpaced(DataDD.nsub(),1,DataDD.nsub());
    auto temp=list.transpose().reshaped(DataDD.nsub_t(),DataDD.nsub_x());
    auto zonematrix=temp(Eigen::seq(subt_sx_-1,dx-2),Eigen::seq(0,temp.cols()-1));
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
    return z;
}

unsigned int RasPipelined::check_sx(const Eigen::VectorXd& v, double tol_sx) {
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


Eigen::VectorXd RasPipelined::solve(const SpMat& A, const SpMat& b, SolverTraits traits) {
    auto start = std::chrono::steady_clock::now();
    double tol=traits.tol();
    //double tol_sx=traits.tol_pipe_sx();
    unsigned int max_it=traits.max_it();
    //unsigned int it_wait=traits.it_wait();
    double res=tol+1;
    std::vector<double> resinf_vec;   //non è nemmeno relativo, ne precondz. Controllare.
    unsigned int niter=0;

    Eigen::VectorXd uw=Eigen::VectorXd::Zero(domain.nln()*domain.nt()*domain.nx()*2);
    Eigen::VectorXd v(domain.nln()*domain.nt()*domain.nx()*2);

    while(res>tol and niter<max_it){
        v=b-A*uw;
        //mat_res_pipe=[mat_res_pipe,v]; per salvarsi l'evoluzione nel dominio
        res=v.lpNorm<Eigen::Infinity>();
        uw=uw+precondAction(b-A*uw,traits);
        resinf_vec.push_back(res);
        niter++;
        solves_++;

    }
    auto end = std::chrono::steady_clock::now();
    std::cout<<"niter: "<<niter<<std::endl;
    std::cout<<"solves: "<<solves_<<std::endl;
    std::cout <<"time in seconds: "<< std::chrono::duration_cast<std::chrono::seconds>(end - start).count()<<std::endl;

    solves_=0;
    subt_sx_=1;
    it_waited_=0;
    zone_=2;

    return uw;
}
