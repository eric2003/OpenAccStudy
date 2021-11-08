#pragma once

#ifdef _WINDOWS
    #define restrict
#endif

void initialize(double *restrict A, double *restrict Anew, int m, int n);

double calcNext(double *restrict A, double *restrict Anew, int m, int n);
        
void swap(double *restrict A, double *restrict Anew, int m, int n);

void deallocate(double *restrict A, double *restrict Anew);
