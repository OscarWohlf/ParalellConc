/*
============================================================================
Filename    : pi.c
Authors     : Oscar Wohlfahrt and Pablo Sarró Sánchez
SCIPERs 	: 416820 and 416086
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include <omp.h>

double calculate_pi (int num_threads, int samples);

int main (int argc, const char *argv[]) {

    int num_threads, num_samples;
    double pi;

    if (argc != 3) {
		printf("Invalid input! Usage: ./pi <num_threads> <num_samples> \n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
	}

    set_clock();
    pi = calculate_pi (num_threads, num_samples);

    printf("- Using %d threads: pi = %.15g computed in %.4gs.\n", num_threads, pi, elapsed_time());

    return 0;
}


double calculate_pi (int num_threads, int samples) {
    double pi;

    /* Your code goes here */
	omp_set_num_threads(num_threads);
	int N_c = 0;
	#pragma omp parallel
    {
    	rand_gen gen = init_rand(omp_get_thread_num());
    	int thread_n = 0;
		#pragma omp for
    	for (int i = 0; i < samples; i++) {
    		double x = next_rand(gen);
    		double y = next_rand(gen);
    		if ((x*x + y*y) < 1.0) {
    			thread_n += 1;
    		}
    	}

		#pragma omp critical
    	{
    		N_c += thread_n;
    	}
    	free_rand(gen);
    }
	pi = 4.0 * ((double)N_c / (double)samples);
    return pi;
}