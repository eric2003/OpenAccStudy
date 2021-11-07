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
	#define MY_ABS fabsf
#endif

inline int IDX( int i, int j, int M, int N )
{
	return i + j * M;
}

//#define IDX( i, j, M, N ) ( ( i ) + ( j ) * (M ) )

int main(int argc, char** argv)
{
    int N = 4096;
    int M = 4096;
    int iter_max = 1000;
    
    const Real pi  = 2.0f * MY_ASIN(1.0f);
    const Real tol = 1.0e-5f;
    Real error     = 1.0f;
    
    Real *restrict A = (Real*)malloc(sizeof(Real)*N*M);
    Real *restrict Anew = (Real*)malloc(sizeof(Real)*N*M);
    Real *restrict y0 = (Real*)malloc(sizeof(Real)*N);

    memset(A, 0, N * M * sizeof(Real));
    
    // set boundary conditions
    for ( int i = 0; i < M; ++ i )
    {
        //A[0*M+i]   = 0.f;
        //A[(N-1)*M+i] = 0.f;
        A[IDX(i,0,M,N)]   = 0.f;
        A[IDX(i,N-1,M,N)] = 0.f;
    }
    
    for ( int j = 0; j < N; ++ j )
    {
        y0[j] = sin(pi * j / (N-1));
        //A[j*M+0] = y0[j];
        //A[j*M+M-1] = y0[j]*MY_EXP(-pi);
        A[IDX(0,j,M,N)] = y0[j];
        A[IDX(M-1,j,M,N)] = y0[j]*MY_EXP(-pi);
		
    }
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", N, M);
    
    StartTimer();
    int iter = 0;
    
	#pragma omp parallel for shared(Anew)
    for ( int i = 1; i < M; ++ i )
    {
       //Anew[0*M+i]   = 0.0;
       //Anew[(N-1)*M+i] = 0.0;
       Anew[IDX(i,0,M,N)]   = 0.0;
       Anew[IDX(i,N-1,M,N)] = 0.0;
    }
	#pragma omp parallel for shared(Anew)    
    for ( int j = 1; j < N; ++ j )
    {
        //Anew[j*M+0]   = y0[j];
        //Anew[j*M+M-1] = y0[j]*MY_EXP(-pi);
        Anew[IDX(0,j,M,N)]   = y0[j];
        Anew[IDX(M-1,j,M,N)] = y0[j]*MY_EXP(-pi);
    }
    
    while ( error > tol && iter < iter_max )
	{
        error = 0.f;
		int ij = -1;

		#pragma acc kernels 
		{
			#pragma acc loop independent
			for ( int j = 1; j < N-1; ++ j )
			{
				for ( int i = 1; i < M-1; ++ i )
				{
					Anew[IDX(i,j,M,N)] = 0.25f * ( A[IDX(i+1,j,M,N)] + A[IDX(i-1,j,M,N)]
										 + A[IDX(i,j-1,M,N)] + A[IDX(i,j+1,M,N)]);
										 
					
					error = MY_FMAX( error, MY_ABS(Anew[IDX(i,j,M,N)]-A[IDX(i,j,M,N)]));
				}
			}

			#pragma acc loop independent
			for ( int j = 1; j < N-1; ++ j )
			{
				for ( int i = 1; i < M-1; ++ i )
				{
					A[j*M+i] = Anew[j*M+i];    
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
