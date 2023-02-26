#ifndef SOLVER_TRAITS_H
#define SOLVER_TRAITS_H

class SolverTraits {

public:
    SolverTraits(unsigned int max_it, double tol, double tol_pipe_sx, unsigned int it_wait)
            : max_it_(max_it), tol_(tol), tol_pipe_sx_(tol_pipe_sx), it_wait_(it_wait) {};

    SolverTraits(unsigned int max_it, double tol)
            : max_it_(max_it), tol_(tol), tol_pipe_sx_(1e-10), it_wait_(3) {};

private:

    unsigned int max_it_;
    double tol_;
    double tol_pipe_sx_;
    unsigned int it_wait_;

public:
    auto max_it() const { return max_it_; };

    auto tol() const { return tol_; };

    auto tol_pipe_sx() const { return tol_pipe_sx_; };

    auto it_wait() const { return it_wait_; };

};
#endif //SOLVER_TRAITS_H
