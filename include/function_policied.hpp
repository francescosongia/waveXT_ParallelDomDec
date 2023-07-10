#ifndef FUNCTION_POLICIED_HPP_
#define FUNCTION_POLICIED_HPP_

#include <iostream>
#include "domaindec_solver_base.hpp"
#include "policies.hpp"

// template <class P>
// bool
// equal(const std::string &a, const std::string &b)
// {
//   P equal = P{};
//   if(a.size() == b.size())
//     {
//       for(unsigned int i = 0; i < a.size(); ++i)
//         {
//           if(!equal(a[i], b[i]))
//             return false;
//         }
//       return true;
//     }
//   else
//     {
//       return false;
//     }
// }



template <class P> 
class PolicyFunctionWrapper{

public:
  
  Eigen::VectorXd 
  precondAction(const SpMat& x)
  {
    return func_obj.precondAction(x);
  }
  
private:
  P func_obj;
};

#endif
