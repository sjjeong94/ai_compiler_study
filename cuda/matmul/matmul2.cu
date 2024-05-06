// nvcc -O3 --use_fast_math -Xcompiler -fopenmp ./matmul.cu -o matmul
// OMP_NUM_THREADS=32 ./matmul

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

void matmul_cpu(float *A, float *B, float *C, int M, int N, int K) {
#pragma omp parallel for collapse(2)
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float val = 0.0f;
      for (int k = 0; k < K; ++k) {
        val += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = val;
    }
  }
}

#define BLOCKSIZE 64
#define TILESIZE 8
#define TILENUM (BLOCKSIZE / TILESIZE)
#define BLOCKDIM (TILENUM * TILENUM)

__global__ void matmul_kernel_1(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;
  const int m = cRow * BLOCKSIZE + threadRow;
  const int n = cCol * BLOCKSIZE + threadCol;

  if (m < M && n < N) {
    float val = 0.0f;
    for (int k = 0; k < K; ++k) {
      val += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = val;
  }
}
void matmul_cuda_1(float *A, float *B, float *C, int M, int N, int K) {
  dim3 gridDim(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  matmul_kernel_1<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  cudaCheck(cudaGetLastError());
}

__global__ void matmul_kernel_2(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float val = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      val += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = val;
}
void matmul_cuda_2(float *A, float *B, float *C, int M, int N, int K) {
  dim3 gridDim(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  matmul_kernel_2<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  cudaCheck(cudaGetLastError());
}

__global__ void matmul_kernel_3(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / (BLOCKSIZE / TILESIZE);
  const int threadCol = threadIdx.x % (BLOCKSIZE / TILESIZE);
  const int copySize = BLOCKDIM / TILESIZE;
  const int copyUnit = BLOCKSIZE / copySize;
  const int innerCol = threadIdx.x;

  __shared__ float As[TILESIZE * BLOCKSIZE];
  __shared__ float Bs[TILESIZE * BLOCKSIZE];

  float threadResults[TILESIZE * TILESIZE] = {0.0f};
  float regM[TILESIZE] = {0.0f};
  float regN[TILESIZE] = {0.0f};

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  for (int bkIdx = 0; bkIdx < K; bkIdx += TILESIZE) {
    for (int loadOffset = 0; loadOffset < copyUnit; ++loadOffset) {
      As[loadOffset * BLOCKSIZE + innerCol] = A[innerCol * K + loadOffset];
    }
    for (int loadOffset = 0; loadOffset < copyUnit; ++loadOffset) {
      Bs[loadOffset * BLOCKSIZE + innerCol] = B[loadOffset * N + innerCol];
    }
    __syncthreads();

    A += TILESIZE;
    B += TILESIZE * N;

    for (int dotIdx = 0; dotIdx < TILESIZE; ++dotIdx) {
      for (int i = 0; i < TILESIZE; ++i) {
        regM[i] = As[dotIdx * BLOCKSIZE + threadRow * TILESIZE + i];
      }
      for (int i = 0; i < TILESIZE; ++i) {
        regN[i] = Bs[dotIdx * BLOCKSIZE + threadCol * TILESIZE + i];
      }
      for (int resIdxM = 0; resIdxM < TILESIZE; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TILESIZE; ++resIdxN) {
          threadResults[resIdxM * TILESIZE + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (int resIdxM = 0; resIdxM < TILESIZE; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < TILESIZE; ++resIdxN) {
      C[(threadRow * TILESIZE + resIdxM) * N + threadCol * TILESIZE + resIdxN] =
          threadResults[resIdxM * TILESIZE + resIdxN];
    }
  }
}
void matmul_cuda_3(float *A, float *B, float *C, int M, int N, int K) {
  dim3 gridDim(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  dim3 blockDim(BLOCKDIM);
  matmul_kernel_3<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  cudaCheck(cudaGetLastError());
}

__global__ void matmul_kernel_4(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / (BLOCKSIZE / TILESIZE);
  const int threadCol = threadIdx.x % (BLOCKSIZE / TILESIZE);
  const int copySize = BLOCKDIM / TILESIZE;
  const int copyUnit = BLOCKSIZE / copySize;
  const int innerCol = threadIdx.x;

  __shared__ float As[TILESIZE * BLOCKSIZE];
  __shared__ float Bs[TILESIZE * BLOCKSIZE];

  float threadResults[TILESIZE * TILESIZE] = {0.0f};
  float regM[TILESIZE] = {0.0f};
  float regN[TILESIZE] = {0.0f};

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  for (int bkIdx = 0; bkIdx < K; bkIdx += TILESIZE) {
    for (int loadOffset = 0; loadOffset < copyUnit; loadOffset += 4) {
      float4 tmp = reinterpret_cast<float4 *>(&A[innerCol * K + loadOffset])[0];
      As[(loadOffset + 0) * BLOCKSIZE + innerCol] = tmp.x;
      As[(loadOffset + 1) * BLOCKSIZE + innerCol] = tmp.y;
      As[(loadOffset + 2) * BLOCKSIZE + innerCol] = tmp.z;
      As[(loadOffset + 3) * BLOCKSIZE + innerCol] = tmp.w;
    }
    for (int loadOffset = 0; loadOffset < copyUnit; ++loadOffset) {
      Bs[loadOffset * BLOCKSIZE + innerCol] = B[loadOffset * N + innerCol];
    }
    __syncthreads();

    A += TILESIZE;
    B += TILESIZE * N;

    for (int dotIdx = 0; dotIdx < TILESIZE; ++dotIdx) {
      for (int i = 0; i < TILESIZE; ++i) {
        regM[i] = As[dotIdx * BLOCKSIZE + threadRow * TILESIZE + i];
      }
      for (int i = 0; i < TILESIZE; ++i) {
        regN[i] = Bs[dotIdx * BLOCKSIZE + threadCol * TILESIZE + i];
      }
      for (int resIdxM = 0; resIdxM < TILESIZE; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TILESIZE; ++resIdxN) {
          threadResults[resIdxM * TILESIZE + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (int resIdxM = 0; resIdxM < TILESIZE; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < TILESIZE; ++resIdxN) {
      C[(threadRow * TILESIZE + resIdxM) * N + threadCol * TILESIZE + resIdxN] =
          threadResults[resIdxM * TILESIZE + resIdxN];
    }
  }
}
void matmul_cuda_4(float *A, float *B, float *C, int M, int N, int K) {
  dim3 gridDim(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  dim3 blockDim(BLOCKDIM);
  matmul_kernel_4<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  cudaCheck(cudaGetLastError());
}

int main(int argc, char **argv) {
  srand(0);

  int M = 1024 * 8;
  int N = 768 * 4;
  int K = 768;
  printf("Matrix shape -> M = %d, N = %d, K = %d\n", M, N, K);

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceIdx);
  printf("Device %d: %s\n", deviceIdx, deviceProp.name);

  float *A = make_random_float(M * K);
  float *B = make_random_float(K * N);
  float *C = (float *)malloc(M * N * sizeof(float));

  float *A_d;
  float *B_d;
  float *C_d;

  cudaCheck(cudaMalloc(&A_d, M * K * sizeof(float)));
  cudaCheck(cudaMalloc(&B_d, K * N * sizeof(float)));
  cudaCheck(cudaMalloc(&C_d, M * N * sizeof(float)));
  cudaCheck(cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  matmul_cpu(A, B, C, M, N, K);
  matmul_cuda_4(A_d, B_d, C_d, M, N, K);
  validate_result(C_d, C, "out", M * N, 1e-1f);

  int repeat_times = 100;
  float elapsed_time =
      benchmark_kernel(repeat_times, matmul_cuda_4, A_d, B_d, C_d, M, N, K);

  float tflops = (float)M * N * K * 2 / elapsed_time * 1e3f / 1e12f;
  float max_tflops = 31.24f;
  printf("time %.4f ms | tflops %.2f (%.2f %%)\n", elapsed_time, tflops,
         (tflops / max_tflops * 100.0f));

  free(A);
  free(B);
  free(C);
  cudaCheck(cudaFree(A_d));
  cudaCheck(cudaFree(B_d));
  cudaCheck(cudaFree(C_d));

  return 0;
}