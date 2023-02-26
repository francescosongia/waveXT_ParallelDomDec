#ifndef RAS_HPP_
#define RAS_HPP_

#include "domaindec_solver_base.hpp"
#include <std::vector>

typedef std::vector<double> Vec;

class Ras : public DomainDecSolverBase {
public:
  Ras(Decomposition DataDD) : DomainDecSolverBase(DataDD){};

  Vec solve() override;
  // pensare a modo per tonrare tutte le performance, probabile che debba creare
  // una classe result in cui popolo i vari campi

private:
  Vec precondAction();
};

#endif
