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
    
    int bytes_per_thread = num_buckets * sizeof(int);
    int cache_lines_per_thread = (bytes_per_thread-1) / CACHE_SIZE + 1; // Equivalent to ceil(by_per_thrd/64), from math.h
    int total_bytes_per_thread = CACHE_SIZE*cache_lines_per_thread;
    int alloc_buckets_per_thread = total_bytes_per_thread / sizeof(int);
    
    int *histogram = (int*) calloc(num_buckets, sizeof(int));
    int *tmp_hist = NULL;
    posix_memalign((void **)&tmp_hist, CACHE_SIZE, num_threads * total_bytes_per_thread);

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
        int *thread_tmp_hist_row = tmp_hist + thread_id * alloc_buckets_per_thread;
        rand_gen generator = init_rand(thread_id);

        // Initialise memory to 0.
        for (int i = 0; i < alloc_buckets_per_thread; i++) {
            thread_tmp_hist_row[i] = 0;
        }

        #pragma omp for
        for(int i = 0; i < num_samples; i++){
            int val = next_rand(generator) * num_buckets;
            thread_tmp_hist_row[val]++;
        }
        free_rand(generator);
        #pragma omp single
        for(int thread_id = 0; thread_id < num_threads; thread_id++){
            int *thread_tmp_hist_row = tmp_hist + thread_id * alloc_buckets_per_thread;
            for(int i = 0; i < num_buckets; i++){
               histogram[i] += thread_tmp_hist_row[i];
            }
        }
    }

    free(tmp_hist);
    free(histogram);

    return 0;
}
