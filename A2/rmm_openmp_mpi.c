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
    int rank, nprocs;
    MPI_Init(&argc, &argv);
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
        init_mat(matC, M/2, K/2, -1);
    }

    // Flattening outside the timing
    int *flatA = NULL;
    int *flatB = malloc(N * K * sizeof(int));

    if (rank == 0) {
        flatA = malloc(M * N * sizeof(int));

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                flatA[i * N + j] = matA[i][j];
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                flatB[i * K + j] = matB[i][j];
            }
        }
    }

    if (rank == 0) {
        if (debug) {
            display_matrix(matA, M, N, "A");
            display_matrix(matB, N, K, "B");
        }
        printf("Starting Computation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    set_clock();

    omp_set_num_threads(num_threads);

    int total_rows = M / 2;
    int rows_each_proc = total_rows / nprocs;
    int rows_left = total_rows % nprocs;

    int num_rows_c_local = rows_each_proc + (rank < rows_left ? 1 : 0);

    int *localA = malloc(2 * num_rows_c_local * N * sizeof(int));
    int *localC = calloc(num_rows_c_local * (K / 2), sizeof(int));

    int *flatC = NULL;
    if (rank == 0) {
        flatC = malloc((M / 2) * (K / 2) * sizeof(int));
    }

    int *A_disp = NULL;
    int *A_num_send = NULL;
    int *C_recv_count = NULL;
    int *C_disp = NULL;

    if (rank == 0) {
        A_disp = malloc(nprocs * sizeof(int));
        A_num_send = malloc(nprocs * sizeof(int));
        C_recv_count = malloc(nprocs * sizeof(int));
        C_disp = malloc(nprocs * sizeof(int));

        for (int curr_rank = 0; curr_rank < nprocs; curr_rank++) {
            int rank_rows = rows_each_proc + (curr_rank < rows_left ? 1 : 0);
            int rank_start_c =
                curr_rank * rows_each_proc + (curr_rank < rows_left ? curr_rank : rows_left);

            A_num_send[curr_rank] = 2 * rank_rows * N;
            A_disp[curr_rank] = 2 * rank_start_c * N;

            C_recv_count[curr_rank] = rank_rows * (K / 2);
            C_disp[curr_rank] = rank_start_c * (K / 2);
        }
    }

    MPI_Scatterv(flatA, A_num_send, A_disp, MPI_INT, localA, 2 * num_rows_c_local * N, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(flatB, N * K, MPI_INT, 0, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_rows_c_local; i++) {
        int *c_row  = &localC[i * K / 2];
        int *a0_row = &localA[(2 * i) * N];
        int *a1_row = &localA[(2 * i + 1) * N];

        for (int k = 0; k < N; k++) {
            int a0 = a0_row[k];
            int a1 = a1_row[k];
            int *b_row = &flatB[k * K];

            for (int j = 0; j < K / 2; j++) {
                int b0 = b_row[2 * j];
                int b1 = b_row[2 * j + 1];

                c_row[j] += (a0 + a1) * (b0 + b1);
            }
        }
    }


    MPI_Gatherv(localC, num_rows_c_local * (K / 2), MPI_INT,
            flatC, C_recv_count, C_disp, MPI_INT,
            0, MPI_COMM_WORLD);

    double totaltime = elapsed_time();

    if (rank == 0) {
        for (int i = 0; i < (M / 2); i++) {
            for (int j = 0; j < (K / 2); j++) {
                matC[i][j] = flatC[i * (K / 2) + j];
            }
        }
    }

    if (rank==0) {
        printf("Computation Done!\n");
        if(debug)
            display_matrix(matC, M/2, K/2, "C");
        printf("- Using %d procs: matC computed in %.4gs.\n", nprocs, totaltime);
        write_csv(matC, M/2, K/2, "matC.csv");
    }

    free(flatA);
    free(flatB);
    free(flatC);
    free(localA);
    free(localC);
    free(A_disp);
    free(A_num_send);
    free(C_recv_count);
    free(C_disp);

    if (rank == 0) {
        for (int i = 0; i < M; i++) {
            free(matA[i]);
        }
        for (int i = 0; i < N; i++) {
            free(matB[i]);
        }
        for (int i = 0; i < M / 2; i++) {
            free(matC[i]);
        }
    }

    MPI_Finalize();
    return 0;
}