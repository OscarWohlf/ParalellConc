/*
============================================================================
Filename    : progB.c
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
    int nrounds, size, B1, B2;

    /* Parse input arguments */
    if(argc != 5) {
        printf("Invalid input! Usage: ./progB <nrounds> <size> <B1> <B2>\n");
        return 1;
    } else {
        nrounds = atoi(argv[1]);
        size = atoi(argv[2]);
        B1 = atoi(argv[3]);
        B2 = atoi(argv[4]);
    }

    MPI_Init(&argc, &argv);

    /* Get the number of processes */
    int nprocs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    /* Initialise model */
    int *model = (int*) malloc(size * sizeof(int));
    if (rank == 0) {
        rand_gen generator = init_rand(0);
        for(int i = 0; i < size; i++) {
            model[i] = next_rand(generator) * MAX_VAL;
        }
        free_rand(generator);
    }

    int chunk_size = size / (nprocs - 1);
    int first_item = (rank - 1) * chunk_size;
    int num_items_proc = (rank == nprocs -1) ? size - first_item : chunk_size;
    int *new_model = malloc(size * sizeof(int));
    set_clock();
    for(int round = 0; round < nrounds; round++) {
        /* Write code to train for N rounds */
        if (rank == 0) {
            for (int proc = 1; proc < nprocs; proc++) {
                int proc_first_item = (proc - 1) * chunk_size;
                int proc_num_items = (proc == nprocs -1) ? size - proc_first_item : chunk_size;

                for (int item = proc_first_item; item < proc_first_item + proc_num_items; item += B1) {
                    int num_send = B1;
                    if (item + num_send > proc_first_item + proc_num_items) {
                        num_send = proc_first_item + proc_num_items - item;
                    }
                    MPI_Send(&model[item], num_send, MPI_INT, proc, 1, MPI_COMM_WORLD);
                }
            }

            for (int i = 0; i < size; i++) {
                new_model[i] = model[i];
            }

            int *recv_vals = malloc(B2 * sizeof(int));
            for (int proc = 1; proc < nprocs; proc++) {
                for (int item = 0; item < size; item += B2) {
                    int num_recv = B2;
                    if (item + num_recv > size) {
                        num_recv = size - item;
                    }
                    MPI_Recv(recv_vals, num_recv, MPI_INT, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int i = item; i < item + num_recv; i ++) {
                        new_model[i] += recv_vals[i - item];
                    }

                }
            }

            int *tmp = model;
            model = new_model;
            new_model = tmp;

        } else {
            int *local_in = malloc(num_items_proc * sizeof(int));
            int *local_out   = calloc(size, sizeof(int));
            for (int item = 0; item < num_items_proc; item += B1) {
                int num_recv = B1;
                if (item + num_recv > num_items_proc) {
                    num_recv = num_items_proc - item;
                }
                MPI_Recv(&local_in[item], num_recv, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            compute(local_in, local_out, num_items_proc, size);
            for (int item = 0; item < size; item += B2) {
                int num_send = B2;
                if (item + num_send > size) {
                    num_send = size - item;
                }
                MPI_Send(&local_out[item], num_send, MPI_INT, 0, 2, MPI_COMM_WORLD);
            }
            free(local_in);
            free(local_out);
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
    MPI_Finalize();
    return 0;
}