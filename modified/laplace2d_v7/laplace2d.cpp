/*
 *  Copyright 2015 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include "timer.h"

//#define DOUBLE
#ifdef DOUBLE
	typedef double Real;
	#define MY_EXP exp
	#define MY_ASIN asin
	#define MY_FMAX fmax
	#define MY_ABS fabs
#else
	typedef float Real;	
	#define MY_EXP expf
	#define MY_ASIN asinf
	#define MY_FMAX fmaxf
	//#define MY_FMAX fmax
	//#define MY_ABS fabsf
	//#define MY_ABS fabs
	#define MY_ABS abs
#endif


int main(int argc, char** argv)
{
    int n = 4096;
    int m = 4096;
    int iter_max = 1000;
    
    const Real pi  = 2.0f * MY_ASIN(1.0f);
    const Real tol = 1.0e-5f;
    Real error     = 1.0f;
    
    Real *restrict A = (Real*)malloc(sizeof(Real)*n*m);
    Real *restrict Anew = (Real*)malloc(sizeof(Real)*n*m);
    Real *restrict y0 = (Real*)malloc(sizeof(Real)*n);

    memset(A, 0, n * m * sizeof(Real));
    
    // set boundary conditions
    for (int i = 0; i < m; i++)
    {
        A[0*m+i]   = 0.f;
        A[(n-1)*m+i] = 0.f;
    }
    
    for (int j = 0; j < n; j++)
    {
        y0[j] = sin(pi * j / (n-1));
        A[j*m+0] = y0[j];
        A[j*m+m-1] = y0[j]*MY_EXP(-pi);
    }
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    StartTimer();
    int iter = 0;
    
#pragma omp parallel for shared(Anew)
    for (int i = 1; i < m; i++)
    {
       Anew[0*m+i]   = 0.0;
       Anew[(n-1)*m+i] = 0.0;
    }
#pragma omp parallel for shared(Anew)    
    for (int j = 1; j < n; j++)
    {
        Anew[j*m+0]   = y0[j];
        Anew[j*m+m-1] = y0[j]*MY_EXP(-pi);
    }
    
    while ( error > tol && iter < iter_max ) {
        error = 0.f;

    #pragma acc kernels 
    {
        #pragma acc loop independent
        for( int j = 1; j < n-1; j++) {
            for( int i = 1; i < m-1; i++ ) {

                Anew[j*m+i] = 0.25f * ( A[j*m+i+1] + A[j*m+i-1]
                                     + A[(j-1)*m+i] + A[(j+1)*m+i]);

                //error = MY_FMAX( error, MY_ABS(Anew[j*m+i]-A[j*m+i]));
				error = fmax( error, MY_ABS(Anew[j*m+i]-A[j*m+i]));
            }
        }
		printf("%5d, %0.6f\n", iter, error);

        #pragma acc loop independent
        for( int j = 1; j < n-1; j++) {
            for( int i = 1; i < m-1; i++ ) {
                A[j*m+i] = Anew[j*m+i];    
            }
        }
    }

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    double runtime = GetTimer();
 
    printf(" total: %f s\n", runtime / 1000.f);
	
	free(A);
	free(Anew);
	free(y0);
}
