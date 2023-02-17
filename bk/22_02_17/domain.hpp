#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

class Domain {

public:
  Domain(unsigned int nx, unsigned int nt, double x, double t, unsigned int nln)
      : nx(nx), nt(nt), X(x), T(t), nln(nln), d(1){};
  // X deve essere un vettore di d dimensione, inizializzo campo X in base a d:
  // se d=1 allora X è double altrimenti sarà vettore

private:
  unsigned int nx;
  unsigned int nt;
  double X;
  double T;
  unsigned int nln;
  unsigned int d;
  // struttura femregion

public:
  auto nx() const { return nx; };
  // completare con altri getters
};

#endif
