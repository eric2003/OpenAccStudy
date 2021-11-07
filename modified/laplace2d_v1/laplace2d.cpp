#include <math.h>
#include <string.h>
#include "timer.h"
#define nx 4096
#define _A(array, ix, iy) (array[(ix) + nx * (iy)])
//#define real double
#define real float

int fun( int i, int j )
{
    int result = i + nx * j;
    return result;
}

int main(int argc, char** argv)
{
    int m = 4096;
    int n = 4096;
    int iter_max = 1000;
    
    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 1.0e-5f;
    float error     = 1.0f;
	
	long mn = m * n;
    
	real * A = (real*)malloc(mn*sizeof(real));
	real * Anew = (real*)malloc(mn*sizeof(real));
	real * y0 = (real*)malloc(n*sizeof(real));

    memset(A, 0, mn * sizeof(real));

    for ( int j = 0; j < n; j++ )
    {
        for ( int i = 0; i < m; i++ )
        {
            _A(A, i, j) = 0.0f;
        }
    }
    
    // set boundary conditions
    for (int i = 0; i < m; i++)
    {
		_A(A, i, 0) = 0.0f;
		_A(A, i, n-1) = 0.0f;
    }
    
    for (int j = 0; j < n; j++)
    {
        y0[j] = sinf(pi * j / (n-1));
		_A(A, 0, j) = y0[j];
		_A(A, m-1, j) = y0[j]*expf(-pi);
    }
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
	
    
    StartTimer();
    int iter = 0;
    
    for (int i = 1; i < m; i++)
    {
		_A(Anew, i, 0) = 0.f;
		_A(Anew, i, n-1) = 0.f;
    }	

    for (int j = 1; j < n; j++)
    {
		_A(Anew, 0, j) = y0[j];
		_A(Anew, m-1, j) = y0[j]*expf(-pi);
    }	
    
    while ( error > tol && iter < iter_max )
    {
        error = 0.f;

#pragma acc kernels
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                //Anew[j][i] = 0.25f * ( A[j][i+1] + A[j][i-1]
                //                     + A[j-1][i] + A[j+1][i]);
				real a1 = _A(A, i - 1, j);
				real a2 = _A(A, i + 1, j);
				real a3 = _A(A, i, j + 1);
                real a4 = _A(A, i, j - 1);
				real result = 0.25 * ( a1 + a2 + a3 + a4 );
				real a0 = _A(A, i, j);
				_A(Anew, i, j) = result;
                error = fmaxf( error, fabsf( result - a0 ));
            }
        }
        
#pragma acc kernels
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                //A[j][i] = Anew[j][i];    
				_A(A, i, j) = _A(Anew, i, j);
            }
        }

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    double runtime = GetTimer();
 
    printf(" total: %f s\n", runtime / 1000.f);
	
	free(y0);
	free(A);
	free(Anew);	
}
