/*
============================================================================
Filename    : progD.c
Authors     : Pablo Sarró Sánchez and Oscar Wohlfahrt
SCIPERs		: 416086 and 416820
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "function.h"
#include "utility.h"
#include <mpi.h>

int main(int argc, char *argv[]) {
    int nrounds, size;

    /* Parse input arguments */
    if (argc != 3) {
        printf("Invalid input! Usage: ./progD <nrounds> <size>\n");
        return 1;
    } else {
        nrounds = atoi(argv[1]);
        size = atoi(argv[2]);
    }

    MPI_Init(&argc, &argv);

    /* Get the number of processes */
    int nprocs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    /* Initialise model */
    int *model = (int*) malloc(size * sizeof(int));
    int *new_model = malloc(size * sizeof(int));
    if (rank == 0) {
        rand_gen generator = init_rand(0);
        for(int i = 0; i < size; i++) {
            model[i] = next_rand(generator) * MAX_VAL;
        }
        free_rand(generator);
    }

    /* Prepare Scatterv metadata */
    int *counts = malloc(nprocs * sizeof(int));
    int *displs = malloc(nprocs * sizeof(int));

    int base = size / nprocs;
    int rem  = size % nprocs;

    for (int p = 0; p < nprocs; p++) {
        counts[p] = base + (p < rem ? 1 : 0);
        displs[p] = (p == 0) ? 0 : displs[p-1] + counts[p-1];
    }

    /* Local buffers */
    int local_size = counts[rank];
    int *local_in  = malloc(local_size * sizeof(int));
    int *local_out = calloc(size, sizeof(int));

    set_clock();
    
    for(int round = 0; round < nrounds; round++) {
        MPI_Scatterv(
            model, counts, displs, MPI_INT,
            local_in, local_size, MPI_INT,
            0, MPI_COMM_WORLD
        );

        for (int i = 0; i < size; i++) {
            local_out[i] = 0;
        }

        compute(local_in, local_out, local_size, size);

        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                new_model[i] = 0;
            }
        }

        MPI_Reduce(
            local_out, new_model,
            size, MPI_INT, MPI_SUM,
            0, MPI_COMM_WORLD
        );

        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                new_model[i] += model[i];
            }

            int *tmp = model;
            model = new_model;
            new_model = tmp;
        }
    }

    /* Output stats */
    if (rank == 0) {
        double totaltime = elapsed_time();
        printf("- Using %d procs for %d iterations on %d size: %.3gs.\n", nprocs, nrounds, size, totaltime);
        write_csv(&model, 1, size, "model.csv");
    }

    free(model);
    free(new_model);
    free(local_in);
    free(local_out);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}