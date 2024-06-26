/*
https://github.com/karpathy/llm.c/blob/master/dev/cuda/attention_forward.cu

Kernels for matmul forward pass.
It's advised to use OpenMP here because the CPU implementation is fairly slow otherwise

Compile example:
nvcc -O3 --use_fast_math -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
OMP_NUM_THREADS=32 ./matmul_forward 1

version 2 calls cuBLAS, very fast
OMP_NUM_THREADS=32 ./matmul_forward 2

version 3 calls cuBLASLt, should be even faster
OMP_NUM_THREADS=32 ./matmul_forward 3
*/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CUDA setup

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static cublasHandle_t cublas_handle;
static cublasLtHandle_t cublaslt_handle;
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1: naive kernel, every thread handles one output element, direct global memory access
__global__ void matmul_forward_kernel1(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of out
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (bt < BT && oc < OC) {
        int b = bt / BT;
        int t = bt % BT;
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        const float* wrow = weight + oc*C;
        const float* inp_bt = inp + b * BT * C + t * C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out[bt * OC + oc] = val;
    }
}

// is there no better way other than just adding bias with a whole separate kernel?
// this is a highly memory-bound operation, should be fused into the matmul kernel
// but i can't seem to find a cuBLAS function that does this
__global__ void add_bias(float* out, const float* bias, int B, int T, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// kernel 1 is the most naive matmul kernel
void matmul_forward1(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    dim3 gridDim(ceil_div(B * T, sqrt_block_size), ceil_div(OC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel1<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    cudaCheck(cudaGetLastError());
}

// kernel 2 calls cuBLAS, which should be very efficient
void matmul_forward2(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // for reference API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)
    // for us, inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
    // cuBLAS does C = alpha * A * B + beta * C
    // where A is mxk, B is kxn, C is mxn
    // now, because we use row-major storage, cuBLAS (which is column-major) sees our matrices transposed.
    // algorithmically / in e.g. PyTorch we want to do: out = inp @ weight.T
    // but because cuBLAS is column-major, we actually want to get it to calculate out.T . Mathematically, this is:
    // out.T = weight @ inp.T
    // but again, our variables look transposed, so using the actual weight/inp we have here in this function, this becomes
    // out.T = weight.T @ inp
    // so we need to get cuBLAS to calculate weight.T @ inp (the variables here are the actual ones in this function)
    // => need to call cuBLAS with A = weight, B = inp
    // => need to call cuBLAS with transa = CUBLAS_OP_T, transb = CUBLAS_OP_N

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC));
    // and now we still have to add the bias... (ew)
    if (bias != NULL) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = ceil_div(OC * B * T, block_size);
        add_bias<<<grid_size, block_size>>>(out, bias, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

// uses cublasLt to fuse the bias and gelu
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
void matmul_forward3(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    int has_bias = (bias != NULL);
    int has_gelu = 0;

    // check bias alignment
    if(((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    int returnedResults = 0;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayout;
    cublasLtMatrixLayout_t inputLayout;
    cublasLtMatrixLayout_t outputLayout;
    cublasLtMatrixLayout_t biasLayout;
    cublasLtMatmulHeuristicResult_t heuristic;

    // create the operation descriptor
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_DEFAULT;
    if (has_bias && has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU_BIAS;
    } else if (has_bias) {
        epilogueBias = CUBLASLT_EPILOGUE_BIAS;
    } else if (has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU;
    }
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // define matrix layouts
    cublasCheck(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, C, OC, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, C, B*T, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, OC, B*T, OC));
    cublasCheck(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, OC, 1, OC));

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // find a suitable algorithm
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc,
        weightLayout, inputLayout, outputLayout, outputLayout,
        preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n",
            B, T, C, OC, has_bias, has_gelu);
        exit(EXIT_FAILURE);
    }

    // call the matmul
    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
        &alpha, weight, weightLayout, inp, inputLayout, &beta,
        out, outputLayout, out, outputLayout, &heuristic.algo,
        cublaslt_workspace, cublaslt_workspace_size, 0));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
}

// ----------------------------------------------------------------------------
// newly added kernels
// out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
// inp is (B,T,C), weight is (OC, C), bias is (OC)

__global__ void matmul_forward_kernel11(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC) {
    const int bt = blockIdx.x * blockDim.x + threadIdx.x;
    const int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        for (int i = 0; i < C; i++) {
            val += inp[bt * C + i] * weight[oc * C + i];
        }
        out[bt * OC + oc] = val;
    }
}
void matmul_forward11(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    
    dim3 gridDim(ceil_div(B * T, sqrt_block_size), ceil_div(OC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel11<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    cudaCheck(cudaGetLastError());
}

__global__ void matmul_forward_kernel12(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC,
                                       unsigned int block_size) {
    const int bt = blockIdx.x * block_size + (threadIdx.x / block_size);
    const int oc = blockIdx.y * block_size + (threadIdx.x % block_size);
    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        for (int i = 0; i < C; i++) {
            val += inp[bt * C + i] * weight[oc * C + i];
        }
        out[bt * OC + oc] = val;
    }
}
void matmul_forward12(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    dim3 gridDim(ceil_div(B * T, sqrt_block_size), ceil_div(OC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size * sqrt_block_size);
    matmul_forward_kernel12<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC, sqrt_block_size);
    cudaCheck(cudaGetLastError());
}

#define BLOCKSIZE 16
__global__ void matmul_forward_kernel13(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC) {
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const int threadRow = threadIdx.x / BLOCKSIZE;
    const int threadCol = threadIdx.x % BLOCKSIZE;

    inp += cRow * BLOCKSIZE * C;
    weight += cCol * BLOCKSIZE * C;
    out += cRow * BLOCKSIZE * OC + cCol * BLOCKSIZE;

    const int oc = cCol * BLOCKSIZE + threadCol;
    float val = (bias != NULL) ? bias[oc] : 0.0f;
    for (int bkIdx = 0; bkIdx < C; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = inp[threadRow * C + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = weight[threadRow * C + threadCol];
        __syncthreads();

        inp += BLOCKSIZE;
        weight += BLOCKSIZE;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            val += As[threadRow * BLOCKSIZE + dotIdx] * Bs[threadCol * BLOCKSIZE + dotIdx];
        }
        __syncthreads();
    }
    out[threadRow * OC + threadCol] = val;
}
void matmul_forward13(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    dim3 gridDim(ceil_div(B * T, BLOCKSIZE), ceil_div(OC, BLOCKSIZE));
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    cudaFuncSetAttribute(matmul_forward_kernel13,
                        cudaFuncAttributePreferredSharedMemoryCarveout,
                        cudaSharedmemCarveoutMaxShared);
    matmul_forward_kernel13<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    cudaCheck(cudaGetLastError());
}

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8
__global__ void matmul_forward_kernel14(float* C,
                                       const float* A, const float* B, const float* bias,
                                       int M, int K, int N) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / BN;
    const int threadCol = threadIdx.x % BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN * K;
    C += cRow * BM * N + cCol * BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    
    float threadResults[TM] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerColB * K + innerRowB];
        __syncthreads();

        A += BK;
        B += BK;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    const int oc = cCol * BN + threadCol;
    float val = (bias != NULL) ? bias[oc] : 0.0f;
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx] + val;
    }
}
void matmul_forward14(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    dim3 gridDim(ceil_div(OC, BN), ceil_div(B * T, BM));
    dim3 blockDim((BM * BN) / TM);
    matmul_forward_kernel14<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    cudaCheck(cudaGetLastError());
}

__global__ void matmul_forward_kernel15(float* C,
                                       const float* A, const float* B, const float* bias,
                                       int M, int K, int N) {
    
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int totalResultsBlocktile = BM * BN;

    const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN * K;
    C += cRow * BM * N + cCol * BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;

    const int strideA = numThreadsBlocktile / BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    const int strideB = numThreadsBlocktile / BN;

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[innerColB * K + innerRowB + loadOffset];
        }
        __syncthreads();

        A += BK;
        B += BK;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int oc = cCol * BN + threadCol * TN + resIdxN;
            float val = (bias != NULL) ? bias[oc] : 0.0f;
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN] + val;
        }
    }
}
void matmul_forward15(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    dim3 gridDim(ceil_div(OC, BN), ceil_div(B * T, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    matmul_forward_kernel15<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    cudaCheck(cudaGetLastError());
}

__global__ void matmul_forward_kernel16(float* C,
                                       float* A, float* B, float* bias,
                                       int M, int K, int N) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN * K;
    C += cRow * BM * N + cCol * BN;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BK / 4);
    const int innerColB = threadIdx.x % (BK / 4);

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        float4 tmp;
        tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
        tmp = reinterpret_cast<float4 *>(&A[(innerRowA + 32) * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA + 32] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + 32] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + 32] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + 32] = tmp.w;

        tmp = reinterpret_cast<float4 *>(&B[innerRowB * K + innerColB * 4])[0];
        Bs[(innerColB * 4 + 0) * BN + innerRowB] = tmp.x;
        Bs[(innerColB * 4 + 1) * BN + innerRowB] = tmp.y;
        Bs[(innerColB * 4 + 2) * BN + innerRowB] = tmp.z;
        Bs[(innerColB * 4 + 3) * BN + innerRowB] = tmp.w;
        tmp = reinterpret_cast<float4 *>(&B[(innerRowB + 32) * K + innerColB * 4])[0];
        Bs[(innerColB * 4 + 0) * BN + innerRowB + 32] = tmp.x;
        Bs[(innerColB * 4 + 1) * BN + innerRowB + 32] = tmp.y;
        Bs[(innerColB * 4 + 2) * BN + innerRowB + 32] = tmp.z;
        Bs[(innerColB * 4 + 3) * BN + innerRowB + 32] = tmp.w;
        __syncthreads();

        A += BK;
        B += BK;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            for (int i = 0; i < TN; ++i){
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for (int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            float4 tmp;
            int oc = cCol * BN + threadCol * TN + resIdxN;
            tmp.x = threadResults[resIdxM * TN + resIdxN];
            tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
            tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
            tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
            if ((bias != NULL) && tmp.x != 0) {
                tmp.x += bias[oc];
                tmp.y += bias[oc + 1];
                tmp.z += bias[oc + 2];
                tmp.w += bias[oc + 3];
            }
            reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
        }
    }
}
void matmul_forward16(float* out,
                     float* inp, float* weight, float* bias,
                     int B, int T, int C, int OC) {
    dim3 gridDim(ceil_div(OC, BN), ceil_div(B * T, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    matmul_forward_kernel16<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

// kernel version dispatch
void matmul_forward(int kernel_num,
                    float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC,
                    const int sqrt_block_size) {
    switch (kernel_num) {
        case 1:
            matmul_forward1(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 2:
            matmul_forward2(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 3:
            matmul_forward3(out, inp, weight, bias, B, T, C, OC);
            break;
        case 11:
            matmul_forward11(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 12:
            matmul_forward12(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 13:
            matmul_forward13(out, inp, weight, bias, B, T, C, OC);
            break;
        case 14:
            matmul_forward14(out, inp, weight, bias, B, T, C, OC);
            break;
        case 15:
            matmul_forward15(out, inp, weight, bias, B, T, C, OC);
            break;
        case 16:
            matmul_forward16(out, inp, weight, bias, B, T, C, OC);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    // setup the (global) cuBLASLt workspace
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, OC * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    // time the kernel at different block sizes
    int sqrt_block_sizes[] = {4, 8, 16, 32};

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];
        printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
        matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
        validate_result(d_out, out, "out", B * T * OC, 1e-1f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
                                              kernel_num, d_out, d_inp, d_weight, d_bias,
                                              B, T, C, OC, sqrt_block_size);

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time, tflops);
    }

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasDestroy(cublas_handle));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    return 0;
}