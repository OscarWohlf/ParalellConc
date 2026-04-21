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
    
    /* Step 1: Read the values of M, N and K from the command line arguments. */
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int debug = atoi(argv[4]);

    /* Get the number of processes */
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
        printf("M, N and K must be even\n");
        return 1;
    }

    /* Step 2: Generates and initializes matrices A and B with random values. */
    int *matA[M];
    int *matB[N];
    int *matC[M/2];

    // Only rank 0
    init_mat(matA, M, N, 0);
    init_mat(matB, N, K, 1);
    init_mat(matC, M/2, K/2, -1);

    // Scatter A and broadcast B to other processes
    
    if(debug) {
        display_matrix(matA, M, N, "A");
        display_matrix(matB, N, K, "B");
    }

    /* Step 3: Computes the matrix C as the RMM of matrices A and B. */
    /* Parallelize and optimize this part only! */
    printf("Starting Computation...\n");
    set_clock();

    int total_rows = M / 2;
    int rows_per_proc = total_rows / nprocs;
    int remainder = total_rows % nprocs;
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int idx = start_row; idx < start_row + local_rows; idx++) {
        int *c_row = matC[idx];
        int *a0_row = matA[idx * 2];
        int *a1_row = matA[idx * 2 + 1];
        
        for(int kdx = 0; kdx < N; kdx++) {
            int a0 = a0_row[kdx];
            int a1 = a1_row[kdx];
            int *b_row = matB[kdx];
            
            for(int jdx = 0; jdx < K/2; jdx++) {
                int b0 = b_row[jdx * 2];
                int b1 = b_row[jdx * 2 + 1];

                c_row[jdx] += (a0+a1) * (b0+b1);
            }
        }
    }

    // Gather C from all processes
    double totaltime = elapsed_time();

    /* Step 4: Write matrix C into a csv file matC.csv and exit. */
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