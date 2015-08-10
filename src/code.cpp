#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
   This computes an in-place complex-to-complex FFT 
   x and y are the real and imaginary arrays of 2^m points.
   dir =  1 gives forward transform
   dir = -1 gives reverse transform 
*/
short FFT(short int dir,long m,double *x, double *y)
{
   long n,i,i1,j,k,i2,l,l1,l2;
   double c1,c2,tx,ty,t1,t2,u1,u2,z;

   /* Calculate the number of points */
   n = 1;
   for (i=0;i<m;i++) 
      n *= 2;

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
      if (i < j) {
         tx = x[i];
         ty = y[i];
         x[i] = x[j];
         y[i] = y[j];
         x[j] = tx;
         y[j] = ty;
      }
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }

   /* Compute the FFT */
   c1 = -1.0; 
   c2 = 0.0;
   l2 = 1;
   for (l=0;l<m;l++) {
      l1 = l2;
      l2 <<= 1;
      u1 = 1.0; 
      u2 = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
            i1 = i + l1;
            t1 = u1 * x[i1] - u2 * y[i1];
            t2 = u1 * y[i1] + u2 * x[i1];
            x[i1] = x[i] - t1; 
            y[i1] = y[i] - t2;
            x[i] += t1;
            y[i] += t2;
         }
         z =  u1 * c1 - u2 * c2;
         u2 = u1 * c2 + u2 * c1;
         u1 = z;
      }
      c2 = sqrt((1.0 - c1) / 2.0);
      if (dir == 1) 
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
   }

   /* Scaling for forward transform */
   //if (dir == 1) {
   //   for (i=0;i<n;i++) {
   //      x[i] /= n;
   //      y[i] /= n;
   //   }
   //}
   
   return 1;
}


double* dft(double* data, int n_data) {
  //int n_j = n_data / 2;
  int n_j = n_data;
  double* ft = new double[2*n_j];

  double two_pi_over_n = 2.0 * M_PI / (double)n_data;

  for (int j = 0; j < n_j; ++j) {
    double j_two_pi_over_n = (double)j * two_pi_over_n;
    double yj_re = 0.0;
    double yj_im = 0.0;
    for (int k = 0; k < n_data; ++k) {
      double z = j_two_pi_over_n * (double)k;
      yj_re += data[k] * cos(z);
      yj_im += data[k] * sin(-z);
    }
    ft[2*j] = yj_re;
    ft[2*j+1] = yj_im;
  }

  return ft;
}

double random_num() {
  return ((double) rand() / (RAND_MAX));
}


int main(int argc, char** argv) {

  int i;

  long N = 8;

  double* data_re = new double[N];
  double* data_im = new double[N];
  double* ft;

  for (i = 0; i < N; ++i) {
    //data_re[i] = random_num();
    data_re[i] = (double)(i+1);
    data_im[i] = 0.0;
  }
 
  FFT(1, 3, data_re, data_im);

  printf("FFT\n");
  for (i = 0; i < N; ++i) {
    printf("%f %f\n", data_re[i], data_im[i]);
  }

  printf("DFT\n");
  
  for (i = 0; i < N; ++i) {
    data_re[i] = (double)(i+1);
  }
  ft = dft(data_re, 8);
  for (i = 0; i < N; ++i) {
    printf("%f %f\n", ft[2*i], ft[2*i+1]);
  }

  delete [] data_re;
  delete [] data_im;
  delete [] ft;

  return 0;
}
