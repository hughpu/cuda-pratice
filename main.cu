#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mykernel() {
    printf("Hello, World!\n");
}

int main()
{
    mykernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    return 0;
}