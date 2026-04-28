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
    
    // --------------------------------------------------------
    // ----------- STEP 1: READ COMMAND LINE ARGS -------------
    // --------------------------------------------------------
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

    // --------------------------------------------------------
    // ------ STEP 2: GENERATE A, B AND DISTRIBUTE DATA -------
    // --------------------------------------------------------

    int *matA[M];
    int *matB[N];
    int *matC[M/2];

    // Only rank 0 initialises matrices A and C (since other processes only need specific rows of A, which will be scattered later).
    if (rank==0) {
        init_mat(matA, M, N, 0);
        init_mat(matC, M/2, K/2, -1);
    }
    // All ranks initialise matrix B (since each of them require the whole matrix).
    init_mat(matB, N, K, 1);

    // Flatten A, for easier scattering (since matA is non-contiguous in memory).
    int *A_flat = NULL;
    if(rank == 0) { // A is flattened only on rank 0, since other processes will receive the info via MPI_Scatterv.
        A_flat = (int *)malloc(M * N * sizeof(int));
        // Copy non-contiguous matA into contiguous A_flat
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                A_flat[i*N + j] = matA[i][j];
            }
        }
    }
    // Also flatten B for better cache performance (cache locality) when accessing the values during RMM computation. Done for all ranks, since every processor will need access to it.
    int *B_flat = (int *)malloc(N * K * sizeof(int));
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < K; j++) {
            B_flat[i*K + j] = matB[i][j];
        }
    }

    // --------------------------------------------------------
    // ----------------- 2.1 SCATTER A'S ROWS -----------------
    // --------------------------------------------------------

    // Input arg 1: Send buffer
    int* sendbuf_scat = A_flat;
    // Input args 2 & 3: Send counts (n_elems to be received x proc) and displacements (starting idx of elems to be sent to each proc)
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
            if (p < remaining_unasign_rows) { // we will assign an extra row to the first 'remaining_unasign_rows' processes, to ensure all of A's rows are covered
                p_rows = rows_per_proc + 1; // Number of rows given to each process (with the extra one)
                sendcounts[p] = 2 * p_rows * N; // We need to send pairs of rows (RMM) and each row of A has N elements.
            } else {
                p_rows = rows_per_proc; // Number of rows now is just the previous number (no extra)
                sendcounts[p] = 2 * p_rows * N;
            }
            if (p == 0) {
                displs[p] = 0; // First process starts at the beginning of the array
            } else {
                displs[p] = displs[p-1] + sendcounts[p-1]; // Starting index for process p is the previous displacement + previous count
            }
        }
    }
    // Input arg 4: Send type
    MPI_Datatype sendtype_scat = MPI_INT;
    // Input arg 5: Receive buffer (need to recalculate the rows per each rank/processor)
    int local_rows;
    if (rank < remaining_unasign_rows) {
        local_rows = rows_per_proc + 1;
    } else {
        local_rows = rows_per_proc;
    }
    int *A_recv = (int *)malloc(2 * local_rows * N * sizeof(int));
    // Input arg 6: Receive count (number of elements to be received)
    int recvcount = 2 * local_rows * N; // Explained previously.
    // Input arg 7: Receive type
    MPI_Datatype recvtype_scat = MPI_INT;
    // Input arg 8: Root process (the one that scatters the data)
    int root_scat = 0;
    // Input arg 9: Communicator
    MPI_Comm comm_scat = MPI_COMM_WORLD;

    MPI_Scatterv(
        sendbuf_scat, sendcounts, displs, 
        sendtype_scat, A_recv, recvcount, 
        recvtype_scat, root_scat, comm_scat
    );

    // Allocate space for a local matrix C per each process, which will store the partial results.
    int *local_C = (int *)calloc(local_rows * (K/2), sizeof(int)); // Flat 1D array.

    if(debug) {
        display_matrix(matA, M, N, "A");
        display_matrix(matB, N, K, "B");
    }

    // --------------------------------------------------------
    // ----------------- STEP 3: COMPUTE RMM ------------------
    // --------------------------------------------------------

    /* Parallelise and optimise this part only! */
    if (rank==0) {
        printf("Starting Computation...\n");
    }
    // Ensure all processes are ready to start the computation before starting the clock.
    MPI_Barrier(MPI_COMM_WORLD);
    set_clock();
        
    for(int m = 0; m < local_rows; m++) {
        int *c_row = &local_C[m * (K/2)];
        int *a0_row = &A_recv[2*m * N];
        int *a1_row = &A_recv[(2*m+1) * N];
        
        for(int n = 0; n < N; n++) {
            int a0 = a0_row[n];
            int a1 = a1_row[n];
            int *b_row = &B_flat[n * K]; // n-th row of B.
            int a_sum = a0 + a1; // Precompute a0+a1 sum outside of j loop.

            for(int k = 0; k < K/2; k++) {
                int b0 = b_row[k * 2];
                int b1 = b_row[k * 2 + 1];

                c_row[k] += a_sum * (b0+b1);
            }
        }
    }

    // --------------------------------------------------------
    // ------------- 3.1 GATHER C'S LOCAL RESULTS -------------
    // --------------------------------------------------------

    // Input arg 1: Send buffer
    int* sendbuf_gath = &local_C[0];
    // Input arg 2: Send count (n_elems to be sent per proc)
    int sendcount = local_rows * (K/2);
    // Input arg 3: Send type
    MPI_Datatype sendtype_gath = MPI_INT;
    // Input arg 4: Receive buffer (flattened C)
    int *recvbuf = NULL;
    if(rank == 0) {
        recvbuf = (int *)malloc((M/2) * (K/2) * sizeof(int));
    }
    // Input arg 5 & 6: Receive counts (n_elems to be received per proc) and displacements (starting idx of elems to be received from each proc)
    int* recvcounts = NULL;
    // displs is reused from previous scatter, hence no need to allocate it again.
    if(rank == 0) {
        recvcounts = (int *)malloc(nprocs * sizeof(int));
        for(int p = 0; p < nprocs; p++) {
            if (p < remaining_unasign_rows) {
                p_rows = rows_per_proc + 1;
                recvcounts[p] = p_rows * (K/2);
            } else {
                p_rows = rows_per_proc; // Number of rows now is just the previous number (no extra)
                recvcounts[p] = p_rows * (K/2);
            }
            if (p == 0) {
                displs[p] = 0; // First process starts at the beginning of the array
            } else {
                displs[p] = displs[p-1] + recvcounts[p-1]; // Starting index for process p is the previous displacement + previous count
            }
        }
    }
    // Input arg 7: Receive type
    MPI_Datatype recvtype_gath = MPI_INT;
    // Input arg 8: Root process (the one that gathers the data)
    int root_gath = 0;
    // Input arg 9: Communicator
    MPI_Comm comm_gath = MPI_COMM_WORLD;

    MPI_Gatherv(
        sendbuf_gath, sendcount, sendtype_gath,
        recvbuf, recvcounts, displs,
        recvtype_gath, root_gath, comm_gath
    );

    double totaltime = elapsed_time();

    // --------------------------------------------------------
    // ---------------- STEP 4: WRITE C IN CSV ----------------
    // --------------------------------------------------------

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
        for(int i = 0; i < M/2; i++) {
            free(matC[i]);
        }
    }
    for(int i = 0; i < N; i++) {
        free(matB[i]);
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