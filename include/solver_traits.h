#ifndef SOLVER_TRAITS_H
#define SOLVER_TRAITS_H

class SolverTraits {

public:
    SolverTraits(unsigned int max_it, double tol, double tol_pipe_sx, unsigned int it_wait)
            : max_it_(max_it), tol_(tol), tol_pipe_sx_(tol_pipe_sx), it_wait_(it_wait) {};

    SolverTraits(unsigned int max_it, double tol)
            : max_it_(max_it), tol_(tol), tol_pipe_sx_(1e-10), it_wait_(3),
              solves_(0), subt_sx_(1), zone_(2), it_waited_(0) {};

private:

    unsigned int max_it_;
    double tol_;
    double tol_pipe_sx_;
    unsigned int it_wait_;

    unsigned int solves_;
    unsigned int subt_sx_;
    unsigned int zone_;
    unsigned int it_waited_;

public:
    auto max_it() const { return max_it_; };
    auto tol() const { return tol_; };
    auto tol_pipe_sx() const { return tol_pipe_sx_; };
    auto it_wait() const { return it_wait_; };

    auto solves() const { return solves_; };
    auto subt_sx() const { return subt_sx_; };
    auto zone() const { return zone_; };
    auto it_waited() const { return it_waited_; };

    void setSolves(unsigned int solves) { solves_ = solves; };
    void setSubtSx(unsigned int subt_sx) { subt_sx_ = subt_sx; };
    void setZone(unsigned int zone) { zone_ = zone; };
    void setItWaited(unsigned int it_waited) { it_waited_ = it_waited; };





};
#endif //SOLVER_TRAITS_H
