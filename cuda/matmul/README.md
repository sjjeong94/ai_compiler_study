# Matmul Implementations

GPU: A10 (31.24 TFLOPS)

| Kernel      | TFLOPS | Config                        |
| ----------- | ------ | ----------------------------- |
| Naive       | 1.33   | BLOCKSIZE = 32                |
| Block Tile  | 1.52   | BLOCKSIZE = 32                |
| Thread Tile | 8.04   | BLOCKSIZE = 64, TILESIZE = 8  |
| Vectorize   | 9.87   | BLOCKSIZE = 64, TILESIZE = 8  |
| Vectorize   | 12.00  | BLOCKSIZE = 128, TILESIZE = 8 |
| cuBLAS      | 14.68  | CUBLAS_DEFAULT_MATH           |
| cuBLAS      | 35.33  | CUBLAS_TF32_TENSOR_OP_MATH    |
