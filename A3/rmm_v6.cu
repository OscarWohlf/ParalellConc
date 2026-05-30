/*
============================================================================
Filename    : rmm_v6.cu
Authors     : Pablo Sarró Sánchez and Oscar Wohlfahrt
SCIPERs		: 416086 and 416820
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

#ifndef TILE
#define TILE 32
#endif

/* CPU Baseline */
void rmm_cpu(int *matA, int *matB, int *matC, int M, int N, int K)
{
    for(int idx = 0; idx < M/2; idx++) {
        for(int jdx = 0; jdx < K/2; jdx++) {
            matC[idx*(K/2) + jdx] = 0;
            for(int aoff = 0; aoff < 2; aoff++) {
                for(int boff = 0; boff < 2; boff++) {
                    for(int kdx = 0; kdx < N; kdx++) {
                        matC[idx*(K/2) + jdx] += matA[(idx*2 + aoff)*N + kdx] * matB[kdx*K + jdx*2 + boff];
                    }
                }
            }
        }
    }
}



__global__ void rmm_kernel(int *matA, int *matB, int *matC, int M, int N, int K) {
    __shared__ int tileA[2*TILE][TILE];  // 2*TILE rows: covers 2 row-patches
    __shared__ int tileB[TILE][2*TILE];  // 2*TILE cols: covers 2 col-patches

    // Each thread is responsible for a 2x2 block in output C
    // Grid is (K/2)/(2*TILE) x (M/2)/(2*TILE)
    unsigned int col  = blockIdx.x * (2*TILE) + threadIdx.x;  // base col in C
    unsigned int row  = blockIdx.y * (2*TILE) + threadIdx.y;  // base row in C

    int sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;

    for (unsigned int tileStart = 0; tileStart < N; tileStart += TILE) {

        // --- Load tileA: 2*TILE rows, TILE cols ---
        // Thread (ty, tx) loads row `row` and row `row + TILE`
        if (row < M/2 && tileStart + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = matA[2*row*N + tileStart + threadIdx.x] + matA[(2*row+1)*N + tileStart + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }
        unsigned int row2 = row + TILE;
        if (row2 < M/2 && tileStart + threadIdx.x < N) {
            tileA[threadIdx.y + TILE][threadIdx.x] = matA[2*row2*N + tileStart + threadIdx.x] + matA[(2*row2+1)*N + tileStart + threadIdx.x];
        } else {
            tileA[threadIdx.y + TILE][threadIdx.x] = 0;
        }

        // --- Load tileB: TILE rows, 2*TILE cols ---
        if (col < K/2 && tileStart + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x] = matB[(tileStart+threadIdx.y)*K + 2*col] + matB[(tileStart+threadIdx.y)*K + 2*col + 1];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }
        unsigned int col2 = col + TILE;
        if (col2 < K/2 && tileStart + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x + TILE] = matB[(tileStart+threadIdx.y)*K + 2*col2] + matB[(tileStart+threadIdx.y)*K + 2*col2 + 1];
        } else {
            tileB[threadIdx.y][threadIdx.x + TILE] = 0;
        }

        __syncthreads();

        // --- Compute 2x2 register tile ---
        for (unsigned int kdx = 0; kdx < TILE; kdx++) {
            int a0 = tileA[threadIdx.y][kdx];
            int a1 = tileA[threadIdx.y + TILE][kdx];
            int b0 = tileB[kdx][threadIdx.x];
            int b1 = tileB[kdx][threadIdx.x + TILE];
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        __syncthreads();
    }

    // --- Write 2x2 outputs ---
    unsigned int row2 = row + TILE;
    unsigned int col2 = col + TILE;
    unsigned int halfK = K/2;

    if (row  < M/2 && col  < K/2) matC[row  * halfK + col ] = sum00;
    if (row  < M/2 && col2 < K/2) matC[row  * halfK + col2] = sum01;
    if (row2 < M/2 && col  < K/2) matC[row2 * halfK + col ] = sum10;
    if (row2 < M/2 && col2 < K/2) matC[row2 * halfK + col2] = sum11;
}




/* GPU Optimized Function */
void rmm_gpu(int *matA, int *matB, int *matC, int M, int N, int K)
{
    /* Cuda events for calculating elapsed time */
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);

    /* Preprocessing (if any) goes here */
    int *matA_d;
    int *matB_d;
    int *matC_d;

    cudaMalloc((void**) &matA_d, M * N * sizeof(int));
    cudaMalloc((void**) &matB_d, K * N * sizeof(int));
    cudaMalloc((void**) &matC_d, (M/2) * (K/2) * sizeof(int));

    cudaEventRecord(cpy_H2D_start);
    /* Copying array(s) from   to device goes here */
    cudaMemcpy(matA_d, matA, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matB_d, matB, N*K * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    cudaEventRecord(comp_start);
    /* Launching the GPU kernel to do the computation goes here */
    dim3 numThreadsPerBlock(TILE, TILE);
    // Grid covers 2*TILE output elements per block in each dimension
    dim3 numBlocks(
        ((K/2) + 2*TILE - 1) / (2*TILE),
        ((M/2) + 2*TILE - 1) / (2*TILE)
    );
    rmm_kernel<<<numBlocks, numThreadsPerBlock>>>(matA_d, matB_d, matC_d, M, N, K);

    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);



    cudaEventRecord(cpy_D2H_start);
    /* Copying array(s) from device to host goes here */
    cudaMemcpy(matC, matC_d, (M/2) * (K/2) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);


    /* Postprocessing (if any) goes here */
    cudaFree(matA_d);
    cudaFree(matB_d);
    cudaFree(matC_d);
    /* Display timing statistics */
    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout << "Host to Device MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout << "RMM operation takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout << "Device to Host MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;
}
