#include "bisection.hpp"
#include "extendedAssert.hpp"


double
Bisection::solve()
{
  double a = traits.a();
  double b = traits.b();
  unsigned int n = 0u;
  auto f = traits.f();

  double ya = f(a);
  double yb = f(b);
  double delta = b - a;
  SURE_ASSERT(ya * yb < 0, "Function must change sign at the two end values");
  double yc{ya};
  double c{a};
  while(std::abs(delta) > 2 * traits.tol_incr() && std::abs(f((a + b) / 2.)) > traits.tol_fun() && n < traits.num_it())
    {
      c = (a + b) / 2.;
      yc = f(c);
      if(yc * ya < 0.0)
        {
          yb = yc;
          b = c;
        }
      else
        {
          ya = yc;
          a = c;
        }
      delta = b - a;
      n++;
    }
  return (a + b) / 2.;
}
