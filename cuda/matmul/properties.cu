// nvcc ./properties.cu -o properties && ./properties 

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
 
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <stdio.h>
#include <cmath>
#include <iostream>

 
int main()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
 
    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);
    printf("%32s  %d / %d\n", "device count", devNo, deviceCount);
    printf("%32s  %s\n", "device", iProp.name);
    printf("%32s  %d\n", "multiprocessor count", iProp.multiProcessorCount);
    printf("%32s  %d\n", "clock rate", iProp.clockRate);
    printf("%32s  %d.%d\n", "compute capability", iProp.major, iProp.minor);
    printf("%32s  %zd MiB\n", "total global memory", iProp.totalGlobalMem / 1024 / 1024);
    printf("%32s  %zd KiB\n", "total constant memory", iProp.totalConstMem / 1024);
    printf("%32s  %zd KiB\n", "shared memory per block", iProp.sharedMemPerBlock / 1024);
    printf("%32s  %d\n", "max threads per block", iProp.maxThreadsPerBlock);
    printf("%32s  [%d, %d, %d]\n", "max grid size", iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);
    printf("%32s  [%d, %d, %d]\n", "max thread dims", iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
    printf("%32s  %d\n", "warp size", iProp.warpSize);
    printf("%32s  %d\n", "max threads per warp", iProp.maxThreadsPerBlock);
}
