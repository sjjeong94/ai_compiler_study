nvcc simplest_kernel.cu -o simplest_kernel
nvcc cublas_gemm.cu -o cublas_gemm -lcublas
nvcc -O3 --use_fast_math -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt
