/*
============================================================================
Filename    : rmm_v1.cu
Authors     : Pablo Sarró Sánchez and Oscar Wohlfahrt
SCIPERs		: 416086 and 416820
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ((M / 2) * (K / 2))) {
        int row = idx / (K / 2);
        int col = idx % (K/2);
        int sum = 0;
        for(int aoff = 0; aoff < 2; aoff++) {
            for(int boff = 0; boff < 2; boff++) {
                for(int kdx = 0; kdx < N; kdx++) {
                    sum += matA[(row*2 + aoff)*N + kdx] * matB[kdx*K + col*2 + boff];
                }
            }
        }
        matC[row*(K/2) + col] = sum;
    }
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
    int numThreadsPerBlock, numBlocks;

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
    numThreadsPerBlock = 16 * 16;
    numBlocks = (M/2 * K/2 + numThreadsPerBlock - 1) / numThreadsPerBlock;
    rmm_kernel <<< numBlocks, numThreadsPerBlock >>> (matA_d, matB_d, matC_d, M, N ,K);

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
