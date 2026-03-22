/*
============================================================================
Filename    : sharing_par.c
Author      : Pablo Sarró Sánchez and Oscar Wohlfahrt
SCIPER		: 416086 and XXXXXX
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"

int perform_buckets_computation(int, int, int);

int main (int argc, const char *argv[]) {
    int num_threads, num_samples, num_buckets;

    if (argc != 4) {
		printf("Invalid input! Usage: ./sharing <num_threads> <num_samples> <num_buckets> \n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
        num_buckets = atoi(argv[3]);
	}
    
    set_clock();
    perform_buckets_computation(num_threads, num_samples, num_buckets);

    printf("Using %d threads: %d operations completed in %.4gs.\n", num_threads, num_samples, elapsed_time());
    return 0;
}

/* Parallelize and optimise this function */
int perform_buckets_computation(int num_threads, int num_samples, int num_buckets) {    
    int *histogram = (int*) calloc(num_buckets, sizeof(int));
    
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        rand_gen generator = init_rand(omp_get_thread_num());
        #pragma omp for
        for(int i = 0; i < num_samples; i++){
            int val = next_rand(generator) * num_buckets;
            #pragma omp atomic
            histogram[val]++;
        }
        free_rand(generator);
    }

    free(histogram);
    return 0;
}
