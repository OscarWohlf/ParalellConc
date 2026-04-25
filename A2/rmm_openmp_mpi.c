/*
============================================================================
Filename    : rmm_openmp_mpi.c
Authors     : Oscar Wohlfahrt and Pablo Sarró Sánchez
SCIPERs 	: 416820 and 416086
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include <omp.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(argc != 6) {
        if(rank == 0) {
            printf("Usage: %s <M> <N> <K> <0|1> <nthreads>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int debug = atoi(argv[4]);
    int num_threads = atoi(argv[5]);

    if(M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
        if(rank == 0) {
            printf("M, N and K must be even\n");
        }
        MPI_Finalize();
        return 1;
    }

    int *matA[M];
    int *matB[N];
    int *matC[M/2];

    if (rank == 0) {
        init_mat(matA, M, N, 0);
        init_mat(matB, N, K, 1);
    }

    init_mat(matC, M/2, K/2, -1);

    if(debug && rank == 0) {
        display_matrix(matA, M, N, "A");
        display_matrix(matB, N, K, "B");
    }
    if (rank == =) {
         printf("Starting Computation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    set_clock();
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M / 2; i++) {
        int *c_row  = matC[i];
        int *a0_row = matA[2 * i];
        int *a1_row = matA[2 * i + 1];

        for (int k = 0; k < N; k++) {
            int a0 = a0_row[k];
            int a1 = a1_row[k];
            int *b_row = matB[k];

            for (int j = 0; j < K / 2; j++) {
                int b0 = b_row[2 * j];
                int b1 = b_row[2 * j + 1];

                c_row[j] += a0 * b0;
                c_row[j] += a0 * b1;
                c_row[j] += a1 * b0;
                c_row[j] += a1 * b1;
            }
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    double totaltime = elapsed_time();

    if (rank==0) {
        printf("Computation Done!\n");
        if(debug)
            display_matrix(matC, M/2, K/2, "C");
        printf("- Using %d procs: matC computed in %.4gs.\n", nprocs, totaltime);
        write_csv(matC, M/2, K/2, "matC.csv");
    }

    MPI_Finalize();
    return 0;
}