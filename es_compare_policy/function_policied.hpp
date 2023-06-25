#ifndef FUNCTION_POLICIED_HPP_
#define FUNCTION_POLICIED_HPP_

#include <iostream>

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



template <class Policy> class PolicyFunctionWrapper
{
public:
  int
  solve_fun(const int &a, const int &b) const
  {
    return func_obj.solve(a,b);
  }

  int
  add_fun(const int &a, const int &b) const
  {
    return func_obj.addition(a,b);
  }

private:
  Policy func_obj;
};

#endif
