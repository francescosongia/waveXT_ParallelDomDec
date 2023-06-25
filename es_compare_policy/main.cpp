#include <iostream>
#include <vector>
#include "policies.hpp"
#include "function_policied.hpp"

int main() {
    PolicyFunctionWrapper<Sequential> sequential_func;
    PolicyFunctionWrapper<Parallel> parallel_func;
    int a,b;
    a = 2;
    b=3;
    std::cout<<sequential_func.solve_fun(a,b)<<std::endl;
    std::cout<<parallel_func.solve_fun(a,b)<<std::endl;
    std::cout<<sequential_func.add_fun(a,b)<<std::endl;
    std::cout<<parallel_func.add_fun(a,b)<<std::endl;
    return 0;
}
