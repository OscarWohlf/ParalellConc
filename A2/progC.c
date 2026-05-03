/*
============================================================================
Filename    : progC.c
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
    if (argc != 5) {
        printf("Invalid input! Usage: ./progC <nrounds> <size> <B1> <B2>\n");
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
    int *new_model = malloc(size * sizeof(int));
    if (rank == 0) {
        rand_gen generator = init_rand(0);
        for(int i = 0; i < size; i++) {
            model[i] = next_rand(generator) * MAX_VAL;
        }
        free_rand(generator);
    }

    int chunk_size = size / (nprocs-1);
    int first_item = (rank-1) * chunk_size;
    int num_items_proc = (rank == nprocs-1) ? size - first_item : chunk_size;
    
    set_clock();
    for(int round = 0; round < nrounds; round++) {
        // -------------------- SERVER (rank == 0) -------------------- //
        if (rank == 0) {
            
            // ---------------- SEND (to workers) ---------------- //
            MPI_Request *send_reqs = malloc((nprocs - 1) * (chunk_size + B1 - 1) / B1 * sizeof(MPI_Request)); // Slightly over-allocate for safety
            int req_idx = 0;
            
            for (int proc = 1; proc < nprocs; proc++) {
                int start = (proc-1) * chunk_size;
                int count = (proc == nprocs-1) ? size - start : chunk_size;

                for (int item = start; item < start + count; item += B1) {
                    int num_send = B1;
                    if (item + num_send > start + count) {
                        num_send = start + count - item;
                    }
                    MPI_Isend(
                        &model[item], num_send, MPI_INT,
                        proc, 1, MPI_COMM_WORLD, &send_reqs[req_idx]
                    );
                    req_idx++;
                }
            }
            int num_send_reqs = req_idx;

            // ---------------- COMPUTE ---------------- //
            // Initialise new model while sends are in flight
            for (int i = 0; i < size; i++) {
                new_model[i] = model[i];
            }

            // Wait for all sends to complete
            if (num_send_reqs > 0) {
                MPI_Waitall(num_send_reqs, send_reqs, MPI_STATUSES_IGNORE);
            }

            // ---------------- RECEIVE (from workers) ---------------- //
            // Use sliding window to avoid a large number of (expensive) requests
            int *recv_buf = malloc(B2 * sizeof(int)); // Only 1 chunk per worker at a time
            int window_size = nprocs-1; // Length of sliding window
            MPI_Request *recv_reqs = malloc(window_size * sizeof(MPI_Request)); // One request per chunk per worker (hence, n_workers=nprocs-1 requests)
            int *recv_items = malloc(window_size * sizeof(int)); // Saving the item positions for each worker's chunk
            int *recv_src = malloc(window_size * sizeof(int));   // Saving the ranks of each chunk
            int *next_item = calloc(nprocs, sizeof(int));   // Next position to receive from each worker
            
            int num_chunks_per_worker = (size + B2 - 1) / B2;
            int total_recvs = (nprocs-1) * num_chunks_per_worker;
            
            // Post one receive per worker
            for (int proc = 1; proc < nprocs; proc++) {
                int num_recv = B2;
                if (num_recv > size) {
                    num_recv = size;
                }
                MPI_Irecv(
                    recv_buf, num_recv, MPI_INT, 
                    proc, 2, MPI_COMM_WORLD, &recv_reqs[proc-1]
                );
                recv_items[proc-1] = 0;
                recv_src[proc-1] = proc;
            }
            
            // Process receives using sliding window
            for (int completed = 0; completed < total_recvs; completed++) {
                int idx;
                MPI_Waitany(window_size, recv_reqs, &idx, MPI_STATUS_IGNORE);
                
                int item = recv_items[idx];
                int proc = recv_src[idx];
                int num_recv = B2;
                if (item + num_recv > size) {
                    num_recv = size - item;
                }
                
                // Aggregate received data
                for (int k = 0; k < num_recv; k++) {
                    new_model[item + k] += recv_buf[k];
                }
                
                next_item[proc] = item + num_recv;
                
                // Post next receive for this worker if there's more
                if (next_item[proc] < size) {
                    int next_recv = B2;
                    if (next_item[proc] + next_recv > size) {
                        next_recv = size - next_item[proc];
                    }
                    MPI_Irecv(
                        recv_buf, next_recv, MPI_INT,
                        proc, 2, MPI_COMM_WORLD, &recv_reqs[idx]
                    );
                    recv_items[idx] = next_item[proc];
                    recv_src[idx] = proc;
                }
            }
            
            free(recv_buf);
            free(recv_reqs);
            free(recv_items);
            free(recv_src);
            free(next_item);
            free(send_reqs);

            int *tmp = model;
            model = new_model;
            new_model = tmp;

        // -------------------- WORKERS (rank > 0) -------------------- //
        } else {

            // ---------------- RECEIVE (from server) ---------------- //
            int *local_in = malloc(num_items_proc * sizeof(int));
            int *local_out = calloc(size, sizeof(int));

            int max_recv_reqs = (num_items_proc + B1 - 1) / B1;
            MPI_Request *recv_reqs = malloc(max_recv_reqs * sizeof(MPI_Request));
            int recv_idx = 0;

            for (int item = 0; item < num_items_proc; item += B1) {
                int num_recv = B1;
                if (item + num_recv > num_items_proc) {
                    num_recv = num_items_proc - item;
                }
                MPI_Irecv(
                    &local_in[item], num_recv, MPI_INT,
                    0, 1, MPI_COMM_WORLD, &recv_reqs[recv_idx]
                );
                recv_idx++;
            }
            int num_recv_reqs = recv_idx;

            // Wait for all receives to complete
            if (num_recv_reqs > 0) {
                MPI_Waitall(num_recv_reqs, recv_reqs, MPI_STATUSES_IGNORE);
            }

            // ---------------- COMPUTE (on local data) ---------------- //
            compute(local_in, local_out, num_items_proc, size);

            // ---------------- SEND (to server) ---------------- //
            int max_send_reqs = (size + B2 - 1) / B2; // = ceil(size / B2)
            MPI_Request *send_reqs = malloc(max_send_reqs * sizeof(MPI_Request));
            int send_idx = 0;

            for (int item = 0; item < size; item += B2) {
                int num_send = B2;
                if (item + num_send > size) {
                    num_send = size - item;
                }
                MPI_Isend(
                    &local_out[item], num_send, MPI_INT,
                    0, 2, MPI_COMM_WORLD, &send_reqs[send_idx]
                );
                send_idx++;
            }
            int num_send_reqs = send_idx;

            // Wait for all sends to complete before freeing memory
            if (num_send_reqs > 0) {
                MPI_Waitall(num_send_reqs, send_reqs, MPI_STATUSES_IGNORE);
            }

            free(local_in);
            free(local_out);
            free(recv_reqs);
            free(send_reqs);
        }
    }

    /* Output stats */
    if (rank == 0) {
        double totaltime = elapsed_time();
        printf("- Using %d procs for %d iterations on %d size: %.3gs.\n", nprocs, nrounds, size, totaltime);
        write_csv(&model, 1, size, "model.csv");
    }
    
    // long long checksum = 0;
    // for (int i = 0; i < size; i++) checksum += model[i];
    
    // if (rank == 0) printf("CHECKSUM: %lld\n", checksum);
        
    free(model);
    free(new_model);
    MPI_Finalize();
    return 0;
}