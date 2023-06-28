#include <iostream>
#include <chrono>
#include "ras_pipelined.hpp"
#include "mpi.h"



/*
//SEQUENTIAL
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
    // Eigen::VectorXi list=Eigen::VectorXi::LinSpaced(DataDD.nsub(),1,DataDD.nsub());
    // auto temp=list.transpose().reshaped(DataDD.nsub_t(),DataDD.nsub_x());
    // auto zonematrix=temp(Eigen::seq(subt_sx_-1,dx-2),Eigen::seq(0,temp.cols()-1));
    
    auto zonematrix=matrix_domain_(Eigen::seq(subt_sx_-1,dx-2),Eigen::seq(0,matrix_domain_.cols()-1));  
    Eigen::VectorXi sub_in_zone=zonematrix.reshaped();
    solves_+=sub_in_zone.size();

    Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);

    // pensare a come fare intersezione tra subinzone e sub_divison_vec. Magari per pipe conviene definire 
    // una divisione fissa (sopra sotto) in modo da riuscire a risalire facilmente a chi spetta quel sub. 
    // anche in casi in cui la divisone non è perfetta so che ce una regola fissa su come assegnare

    //aggiungere membro nella classe ras pipe pper tenere in mente la zona corrente e considero solo i sub 
    //a cui posso accedere. va rinizializzata tutte le volte che inizio precondAction? NO. 
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


*/


//PARALLEL   
Eigen::VectorXd RasPipelined::precondAction(const SpMat& x,SolverTraits traits) {
    
    int rank = local_mat.rank();
    int np = local_mat.sub_assignment().np();
    if(DataDD.nsub_x() % np != 0){
        std::cerr<<"sub_x not proportional to number of processes chosen"<<std::endl;
    }
    int partition = DataDD.nsub_x() / np;  

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
    // Eigen::VectorXi list=Eigen::VectorXi::LinSpaced(DataDD.nsub(),1,DataDD.nsub());
    // auto temp=list.transpose().reshaped(DataDD.nsub_t(),DataDD.nsub_x());
    // auto zonematrix=temp(Eigen::seq(subt_sx_-1,dx-2),Eigen::seq(0,temp.cols()-1));
    
    
    auto zonematrix=matrix_domain_(Eigen::seq(subt_sx_-1,dx-2),Eigen::seq(rank*partition, (rank+1)*partition-1));
    Eigen::VectorXi sub_in_zone=zonematrix.reshaped();
    solves_+=sub_in_zone.size();

    Eigen::VectorXd uk(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);

    // pensare a come fare intersezione tra subinzone e sub_divison_vec. Magari per pipe conviene definire 
    // una divisione fissa (sopra sotto) in modo da riuscire a risalire facilmente a chi spetta quel sub. 
    // anche in casi in cui la divisone non è perfetta so che ce una regola fissa su come assegnare


    for(unsigned int k:sub_in_zone){
        Eigen::SparseLU<SpMat > lu;
        lu.compute(local_mat.getAk(k));
        auto temp = local_mat.getRk(k);
        uk = lu.solve(temp.first*x);
        z=z+(temp.second.transpose())*uk;
    }

    MPI_Allreduce(MPI_IN_PLACE, z.data(), domain.nln()*domain.nt()*domain.nx()*2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);   
    
    return z;
}

unsigned int RasPipelined::check_sx(const Eigen::VectorXd& v, double tol_sx) {
    
    int np = local_mat.sub_assignment().np();
    int partition = DataDD.nsub_x() / np;  
    int rank = local_mat.rank();
    //questi tre valori perche devo calcolarli sempre? non sarebbe meglio metterli come membri (anche privati)

    unsigned int sx1=subt_sx_;
    if (subt_sx_>DataDD.nsub_t()){
        std::cout<<subt_sx_<<std::endl;
        std::cerr<<"err in the definiton of left edge of subdomain window"<<std::endl;
        return 0;
    }
    int fail=0;
    Eigen::VectorXd res(domain.nln()*DataDD.sub_sizes()[0]*DataDD.sub_sizes()[1]*2);

    // posso far fare questo controllo un po ad ognuno dei rank, da defnire però l'intersezione tra i e quello che il rank puo vedere

    //for(size_t i=subt_sx_;i<=DataDD.nsub_t()*(DataDD.nsub_x()-1)+1;i+=DataDD.nsub_t()){
    for(size_t i=subt_sx_+DataDD.nsub_t()*partition*rank; i<=subt_sx_+DataDD.nsub_t()*partition*(rank +1)-1; i+=DataDD.nsub_t()){
        res=local_mat.getRk(i).first*v;
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
}




SolverResults RasPipelined::solve(const SpMat& A, const SpMat& b, SolverTraits traits) {
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

    }
    //MPI_Allreduce(MPI_IN_PLACE, &solves_, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    std::cout<<"niter: "<<niter<<std::endl;
    std::cout<<"solves: "<<solves_<<std::endl;
    double time=std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout <<"time in seconds: "<< time<<std::endl;

    solves_=0;
    subt_sx_=1;
    it_waited_=0;
    zone_=2;

    SolverResults res_obj(uw,solves_,time, traits, DataDD);

    return res_obj;
}
