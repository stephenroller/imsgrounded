// taken from

/**
 * Approximation to the digamma function, from Radford Neal.
 *
 * Original License: Copyright (c) 1995-2003 by Radford M. Neal
 *
 * Permission is granted for anyone to copy, use, modify, or distribute this program and accompanying
 * programs and documents for any purpose, provided this copyright notice is retained and prominently
 * displayed, along with a note saying that the original programs are available from Radford Neal's web
 * page, and note is made of any changes made to the programs. The programs and documents are distributed
 * without any warranty, express or implied. As the programs were written for research purposes only, they
 * have not been tested to the degree that would be advisable in any important application. All use of these
 * programs is entirely at the user's own risk.
 *
 */
static inline double fastdigamma(double x) {
  double r = 0.0;

  while (x <= 5) {
    r -= 1 / x;
    x += 1;
  }

  double f = 1.0 / (x * x);
  double t = f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0
      + f * (-1.0 / 132.0 + f * (691.0 / 32760.0 + f * (-1.0 / 12.0 + f * 3617.0 / 8160.0)))))));
  return r + log(x) - 0.5 / x + t;
}

// ---- END NEAL'S CODE

static inline double fastlgamma (double alpha)
{
/* returns ln(gamma(alpha)) for alpha>0, accurate to 10 decimal places.  
   Stirling's formula is used for the central polynomial part of the procedure.
   Pike MC & Hill ID (1966) Algorithm 291: Logarithm of the gamma function.
   Communications of the Association for Computing Machinery, 9:684
*/
   double x=alpha, f=0, z;

   if (x<7) {
      f=1;  z=x-1;
      while (++z<7)  f*=z;
      x=z;   f=-log(f);
   }
   z = 1/(x*x);
   return  f + (x-0.5)*log(x) - x + .918938533204673 
    + (((-.000595238095238*z+.000793650793651)*z-.002777777777778)*z
         +.083333333333333)/x;  
}
