/*
============================================================================
Filename    : sharing.c
Author      : Pablo Sarró Sánchez and Oscar Wohlfahrt
SCIPER		: 416086 and XXXXXX
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include <omp.h>

#define MAX_BUCKETS 2048
#define CACHE_SIZE 64

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
    /* To avoid false sharing, we would like each "temporary histogram" (the hist where
    each thread would be writing in) to be allocated in DIFFERENT CACHE LINES. This can be done by
    rounding the number of bytes needed per thread up to the nearest multiple of 64 (memory size of
    a cache line), and so in this way, no two temporary histograms will share any cache line --> no false sharing. */

    // int bytes_per_thread = num_buckets*sizeof(int);
    // int cache_lines_per_thread = (bytes_per_thread-1)/64 + 1; // Equivalent to ceil(by_per_thrd/64), from math.h
    // int total_bytes_per_thread = 64*cache_lines_per_thread;
    // int alloc_buckets_per_thread = total_bytes_per_thread / sizeof(int);
    
    int *histogram = (int*) calloc(num_buckets, sizeof(int)); // volatile not needed, since only one thread at a time writes to the "global" histogram.
    
    typedef struct { //new
        int buckets[MAX_BUCKETS]; //new
        int padding[CACHE_SIZE/sizeof(int)]; // this introduces an extra "unused" memory that exactly corresponds to a cache line size, ensuring cache line separation between threads.
    } PaddedRow; //new
    PaddedRow *tmp_hist = calloc(num_threads, sizeof(PaddedRow)); //new

    // Memory allocation check:
    if (!histogram || !tmp_hist) { 
        free(histogram); 
        free(tmp_hist); 
        return -1;
    }
    
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int *thread_tmp_hist_row = tmp_hist[thread_id].buckets; //new
        rand_gen generator = init_rand(thread_id);

        #pragma omp for
        for(int i = 0; i < num_samples; i++){
            int val = next_rand(generator) * num_buckets;
            thread_tmp_hist_row[val]++;
        }
        free_rand(generator);
        #pragma omp single
        for(int thread_id = 0; thread_id < num_threads; thread_id++){
            int *thread_tmp_hist_row = tmp_hist[thread_id].buckets; //new
            for(int i = 0; i < num_buckets; i++){
               histogram[i] += thread_tmp_hist_row[i];
            }
        }
    }

    free(tmp_hist);
    free(histogram);

    return 0;
}
