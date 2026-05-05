/*
============================================================================
Filename    : rmm.c
Authors     : Pablo Sarró Sánchez and Oscar Wohlfahrt
SCIPERs		: 416086 and 416820
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include <mpi.h>

int main(int argc, char *argv[]) {
    if(argc != 5) {
        printf("Usage: %s <M> <N> <K> <0|1>\n", argv[0]);
        return 1;
    }

    // Initialise MPI
    MPI_Init(&argc, &argv);

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int debug = atoi(argv[4]);

    /* Get the number of processes */
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
        if(rank == 0) {
            printf("M, N and K must be even\n");
        }
        MPI_Finalize();
    }

    int *matA[M];
    int *matB[N];
    int *matC[M/2];

    // Only rank 0 initialises matrices A, B and C
    if (rank==0) {
        init_mat(matA, M, N, 0);
        init_mat(matB, N, K, 1);
        init_mat(matC, M/2, K/2, -1);
    }

    // Flatten A, for easier scattering
    int *A_flat = NULL;
    if(rank == 0) {
        A_flat = (int *)malloc(M * N * sizeof(int));
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                A_flat[i*N + j] = matA[i][j];
            }
        }
    }
    // Flatten B for better cache performance
    int *B_flat = (int *)malloc(N * K * sizeof(int));
    if(rank == 0) {
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < K; j++) {
                B_flat[i*K + j] = matB[i][j];
            }
        }
    }

    if(debug) {
        if (rank == 0) {
            display_matrix(matA, M, N, "A");
            display_matrix(matB, N, K, "B");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    set_clock();

    int* sendcounts = NULL;
    int* displs = NULL;
    int total_rows = M / 2;
    int rows_per_proc = total_rows / nprocs;
    int remaining_unasign_rows = total_rows % nprocs;
    int p_rows;

    if(rank == 0) {
        sendcounts = (int *)malloc(nprocs * sizeof(int));
        displs = (int *)malloc(nprocs * sizeof(int));
        for(int p = 0; p < nprocs; p++) {
            if (p < remaining_unasign_rows) {
                p_rows = rows_per_proc + 1;
            } else {
                p_rows = rows_per_proc;
            }
            sendcounts[p] = 2 * p_rows * N;
            if (p == 0) {
                displs[p] = 0;
            } else {
                displs[p] = displs[p-1] + sendcounts[p-1];
        }
    }

    int local_rows;
    if (rank < remaining_unasign_rows) {
        local_rows = rows_per_proc + 1;
    } else {
        local_rows = rows_per_proc;
    }
    int *A_recv = (int *)malloc(2 * local_rows * N * sizeof(int));
    int recvcount = 2 * local_rows * N;

    MPI_Scatterv(
        A_flat, sendcounts, displs, 
        MPI_INT, A_recv, recvcount, 
        MPI_INT, 0, MPI_COMM_WORLD
    );

    MPI_Bcast(B_flat, N * K, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_C = (int *)calloc(local_rows * (K/2), sizeof(int));

    if (rank==0) {
        printf("Starting Computation...\n");
    }
        
    for(int m = 0; m < local_rows; m++) {
        int *c_row = &local_C[m * (K/2)];
        int *a0_row = &A_recv[2*m * N];
        int *a1_row = &A_recv[(2*m+1) * N];
        
        for(int n = 0; n < N; n++) {
            int a0 = a0_row[n];
            int a1 = a1_row[n];
            int *b_row = &B_flat[n * K];
            int a_sum = a0 + a1;

            for(int k = 0; k < K/2; k++) {
                int b0 = b_row[k * 2];
                int b1 = b_row[k * 2 + 1];

                c_row[k] += a_sum * (b0+b1);
            }
        }
    }

    int sendcount = local_rows * (K/2);

    int *recvbuf = NULL;
    if(rank == 0) {
        recvbuf = (int *)malloc((M/2) * (K/2) * sizeof(int));
    }

    int* recvcounts = NULL;

    if(rank == 0) {
        recvcounts = (int *)malloc(nprocs * sizeof(int));
        for(int p = 0; p < nprocs; p++) {
            if (p < remaining_unasign_rows) {
                p_rows = rows_per_proc + 1;
            } else {
                p_rows = rows_per_proc;
            }
            recvcounts[p] = p_rows * (K/2);
            if (p == 0) {
                displs[p] = 0;
            } else {
                displs[p] = displs[p-1] + recvcounts[p-1];
            }
        }
    }

    MPI_Gatherv(
        &local_C[0], sendcount, MPI_INT,
        recvbuf, recvcounts, displs,
        MPI_INT, 0, MPI_COMM_WORLD
    );

    double totaltime = elapsed_time();

    if (rank==0) {
        for(int i = 0; i < M/2; i++) {
            for(int j = 0; j < K/2; j++) {
                matC[i][j] = recvbuf[i*(K/2) + j];
            }
        }
        printf("Computation Done!\n");
        if(debug)
            display_matrix(matC, M/2, K/2, "C");
        printf("- Using %d procs: matC computed in %.4gs.\n", nprocs, totaltime);
        write_csv(matC, M/2, K/2, "matC.csv");
    }

    // Cleanup.
    if(rank == 0) {
        for(int i = 0; i < M; i++) {
            free(matA[i]);
        }
        for(int i = 0; i < N; i++) {
            free(matB[i]);
        }
        for(int i = 0; i < M/2; i++) {
            free(matC[i]);
        }
    }
    if (rank==0) {
        free(A_flat);
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(recvbuf);
    }
    free(B_flat);
    free(A_recv);
    free(local_C);


    MPI_Finalize();
    return 0;
}