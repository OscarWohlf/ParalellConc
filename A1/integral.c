/*
============================================================================
Filename    : integral.c
Authors     : Oscar Wohlfahrt and Pablo Sarró Sánchez
SCIPERs		: XXXXXX and 416086
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include "function.c"

double integrate (int num_threads, int samples, int a, int b, double (*f)(double));

int main (int argc, const char *argv[]) {

    int num_threads, num_samples, a, b;
    double integral;

    if (argc != 5) {
		printf("Invalid input! Usage: ./integral <num_threads> <num_samples> <a> <b>\n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
        a = atoi(argv[3]);
        b = atoi(argv[4]);
	}

    set_clock();

    /* You can use your self-defined funtions by replacing identity_f. */
    integral = integrate (num_threads, num_samples, a, b, custom_f);

    printf("- Using %d threads: integral on [%d,%d] = %.15g computed in %.4gs.\n", num_threads, a, b, integral, elapsed_time());

    return 0;
}


double integrate (int num_threads, int samples, int a, int b, double (*f)(double)) {
    double integral;

    omp_set_num_threads(num_threads);
    double area_sum = 0;

    #pragma omp parallel
    {
        rand_gen gen = init_rand(omp_get_thread_num());
        double x, f_x;
        double area_n = 0;

        #pragma omp for
        for (long i = 0; i < samples; i++){
            x = a + (b-a)*next_rand(gen);
            f_x = f(x);
            area_n += f_x*(b-a);
        }

        #pragma omp critical
        {
            area_sum += area_n;
        }
        free_rand(gen);
    }

    integral = (double)area_sum / (double)samples;
    return integral;
}
