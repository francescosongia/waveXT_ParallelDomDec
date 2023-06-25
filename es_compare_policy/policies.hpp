#ifndef POLICIES_HPP_
#define POLICIES_HPP_

#include <iostream>

struct Sequential 
{
  //! I rely on the existing operator
  int
  solve(const int &a, const int &b) const
  {
    std::cout<<"sequential solve"<<std::endl;
    return a * b;
  }

  int
  addition(const int &a, const int &b) const
  {
    std::cout<<"sequential add"<<std::endl;
    return a + b;
  }
};


struct Parallel
{
public:
  //! equality operator that ignore case of characters
  int
  solve(const int &a, const int &b) const
  {
    std::cout<<"parallel solve"<<std::endl;
    return a * b;
  }

  // capire come fare per aggiungere parametri diversi, usare altro template?
  int
  addition(const int &a, const int &b) const
  {
    std::cout<<"parallel add"<<std::endl;
    return a + b;
  }
};


#endif
