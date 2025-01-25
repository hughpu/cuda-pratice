#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include <cstdio>
#include <cudapractice/helper_cuda.cuh>
#include <map>
#include <string>
#include <vector>

#define USAGE                                                                \
  "USAGE:\n    ./main <demo_name>\n    demo_name: hello_world, reduce_sum, " \
  "transpose, gemm"

#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define FETCH_FLOAT4(element) (reinterpret_cast<float4 *>(&(element))[0])
// cublas API error checking
#define CUBLAS_CHECK(err)                                                  \
  do {                                                                     \
    cublasStatus_t err_ = (err);                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                            \
    }                                                                      \
  } while (0)

template <class T>
class CudaAllocator {
 public:
  using value_type = T;

  T *allocate(size_t size) {
    T *ptr = nullptr;
    checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
    return ptr;
  }

  void deallocate(T *ptr, size_t size) { checkCudaErrors(cudaFree(ptr)); }

  template <class... Args>
  void construct(T *ptr, Args &&...args) {
    if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>)) {
      ::new (ptr) T(std::forward<Args>(args)...);
    }
  }
};

template <class Func>
__global__ void MyKernel(int n, Func func) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    func(i);
  }
}

template <int blockSize, class T>
__global__ void ReduceSumKernel(int n, T *sum, T *arr) {
  __shared__ volatile T local_sum[blockSize];
  int i = blockIdx.x;
  int j = threadIdx.x;
  T temp_sum = 0;
  for (int t = j + blockSize * i; t < n; t += blockSize * gridDim.x) {
    temp_sum += arr[t];
  }

  local_sum[j] = temp_sum;
  __syncthreads();

  if constexpr (blockSize >= 1024) {
    if (j < 512) {
      local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
  }
  if constexpr (blockSize >= 512) {
    if (j < 256) {
      local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
  }
  if constexpr (blockSize >= 256) {
    if (j < 128) {
      local_sum[j] += local_sum[j + 128];
    }
  }
  if constexpr (blockSize >= 128) {
    if (j < 64) {
      local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
  }
  if (j < 32) {
    if constexpr (blockSize >= 64) {
      local_sum[j] += local_sum[j + 32];
    }
    if constexpr (blockSize >= 32) {
      local_sum[j] += local_sum[j + 16];
    }
    if constexpr (blockSize >= 16) {
      local_sum[j] += local_sum[j + 8];
    }
    if constexpr (blockSize >= 8) {
      local_sum[j] += local_sum[j + 4];
    }
    if constexpr (blockSize >= 4) {
      local_sum[j] += local_sum[j + 2];
    }
    if (j == 0) {
      sum[i] = local_sum[0] + local_sum[1];
    }
  }
}

void HelloWorld() {
  int n = 65536;
  std::vector<float, CudaAllocator<float>> y(n);
  thrust::host_vector<float> x_host(n);

  float a = 3.14f;

  for (int i = 0; i < n; ++i) {
    y[i] = std::rand() * (1.f / RAND_MAX);
  }

  thrust::generate(x_host.begin(), x_host.end(),
                   []() { return std::rand() * (1.f / RAND_MAX); });

  thrust::device_vector<float> x_dev = x_host;
  MyKernel<<<32, 128>>>(n, [a, x = x_dev.data(), y = y.data()] __device__(
                               int i) { x[i] = a * x[i] + y[i] + __sinf(i); });

  // checkCudaErrors(cudaDeviceSynchronize());
  x_host = x_dev;

  for (int i = 0; i < n; i++) {
    printf("arr[%d]: %f\n", i, x_host[i]);
  }
}

template <int reduceScale = 4096, int blockSize = 256, class T>
T ReduceSum(thrust::universal_vector<T> &arr, int n) {
  thrust::universal_vector<T> x(n);
  thrust::universal_vector<T> sum(n / reduceScale);
  ReduceSumKernel<blockSize>
      <<<n / reduceScale, blockSize>>>(n, sum.data().get(), arr.data().get());
  checkCudaErrors(cudaDeviceSynchronize());
  T final_sum = 0;
  for (int i = 0; i < n / reduceScale; ++i) {
    final_sum += sum[i];
  }
  return final_sum;
}

void ReduceSum() {
  int n = 1 << 24;
  thrust::universal_vector<int> arr(n);
  for (int i = 0; i < n; ++i) {
    arr[i] = std::rand() % 4;
  }

  float final_sum = ReduceSum(arr, n);
  printf("final sum: %f\n", final_sum);
}

template <int TBlockDimX, int TBlockDimY, typename TValue>
__global__ void TransPoseKernel(TValue *in_mat, TValue *out_mat, int nx,
                                int ny) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > nx || y > ny) return;

  __shared__ int block_share_in[TBlockDimX * TBlockDimY];

  constexpr int bank_confict_diff = 1;
  int blockScopeIdx = threadIdx.y * blockDim.x + threadIdx.x;

  int rx = blockIdx.y * blockDim.y + blockScopeIdx % blockDim.y;
  int ry = blockIdx.x * blockDim.x + blockScopeIdx / blockDim.y;

  block_share_in[blockScopeIdx + blockScopeIdx / blockDim.y *
                                     bank_confict_diff] = in_mat[ry * ny + rx];
  __syncthreads();

  out_mat[y * nx + x] =
      block_share_in[threadIdx.x * (blockDim.y + bank_confict_diff) +
                     threadIdx.y];
}

void TransPose() {
  int nx = 1 << 12, ny = 1 << 11;
  thrust::universal_vector<int> in_mat(ny * nx);
  thrust::universal_vector<int> out_mat(nx * ny);

  for (int i = 0; i < nx * ny; ++i) {
    in_mat[i] = i;
  }

  TransPoseKernel<64, 16><<<dim3(nx / 64, ny / 16, 1), dim3(64, 16, 1)>>>(
      thrust::raw_pointer_cast(in_mat.data()),
      thrust::raw_pointer_cast(out_mat.data()), nx, ny);

  checkCudaErrors(cudaDeviceSynchronize());

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      if (in_mat[x * ny + y] != out_mat[y * nx + x]) {
        char buffer[20];
        sprintf(buffer, "mismatch at (%d, %d): %d != %d\n", x, y,
                in_mat[x * ny + y], out_mat[y * nx + x]);
        printf(buffer);

        throw std::runtime_error(buffer);
      }
    }
  }

  printf("---corresponding in samples---");
  for (int x = 0; x < 4; ++x) {
    printf("\n");
    for (int y = 0; y < 8; ++y) {
      printf("%4d, ", in_mat[x * ny + y]);
    }
  }

  printf("\n\n---corresponding out samples---");
  for (int y = 0; y < 8; ++y) {
    printf("\n");
    for (int x = 0; x < 4; ++x) {
      printf("%4d, ", out_mat[y * nx + x]);
    }
  }

  printf("\n\nAll corrected!");
}

template <const int TBlockSizeM, const int TBlockSizeN, const int TBlockSizeK,
          const int TThreadSizeM, const int TThreadSizeN>
__global__ void GemmKernel(float *__restrict__ A, float *__restrict__ B,
                           float *__restrict__ C, const int M, const int N,
                           const int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  constexpr int kNumThreadsInX = TBlockSizeN / TThreadSizeN;
  constexpr int kNumThreadsInY = TBlockSizeM / TThreadSizeM;
  constexpr int kNumThreads = kNumThreadsInX * kNumThreadsInY;

  int tid = ty * kNumThreadsInX + tx;

  __shared__ float a_share[2][TBlockSizeK][TBlockSizeM];
  __shared__ float b_share[2][TBlockSizeK][TBlockSizeN];

  float accum[TThreadSizeM][TThreadSizeN] = {0.0f};
  float reg_a[2][TThreadSizeM];
  float reg_b[2][TThreadSizeN];

  constexpr int kFloatsPerCopy = 4;
  constexpr int a_share_copy_threads_per_row = TBlockSizeK / kFloatsPerCopy;
  constexpr int a_share_copy_row_stride =
      kNumThreads / a_share_copy_threads_per_row;
  constexpr int b_share_copy_threads_per_row = TBlockSizeN / kFloatsPerCopy;
  constexpr int b_share_copy_row_stride =
      kNumThreads / b_share_copy_threads_per_row;
  float ldg_a[TBlockSizeM / a_share_copy_row_stride * kFloatsPerCopy];

  int a_share_copy_start_row = tid / a_share_copy_threads_per_row;
  int a_share_copy_col = tid % a_share_copy_threads_per_row * kFloatsPerCopy;
  int b_share_copy_start_row = tid / b_share_copy_threads_per_row;
  int b_share_copy_col = tid % b_share_copy_threads_per_row * kFloatsPerCopy;

#pragma unroll
  for (int i = 0; i < TBlockSizeM; i += a_share_copy_row_stride) {
    int ldg_index = i / a_share_copy_row_stride * kFloatsPerCopy;
    FETCH_FLOAT4(ldg_a[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
        i + a_share_copy_start_row + by * TBlockSizeM, a_share_copy_col, K)]);
    a_share[0][a_share_copy_col][a_share_copy_start_row + i] = ldg_a[ldg_index];
    a_share[0][a_share_copy_col + 1][a_share_copy_start_row + i] =
        ldg_a[ldg_index + 1];
    a_share[0][a_share_copy_col + 2][a_share_copy_start_row + i] =
        ldg_a[ldg_index + 2];
    a_share[0][a_share_copy_col + 3][a_share_copy_start_row + i] =
        ldg_a[ldg_index + 3];
  }

#pragma unroll
  for (int i = b_share_copy_start_row; i < TBlockSizeK;
       i += b_share_copy_row_stride) {
    FETCH_FLOAT4(b_share[0][i][b_share_copy_col]) =
        FETCH_FLOAT4(B[OFFSET(i, b_share_copy_col + bx * TBlockSizeN, N)]);
  }

  __syncthreads();

#pragma unroll
  for (int y = 0; y < TThreadSizeM; y += kFloatsPerCopy) {
    FETCH_FLOAT4(reg_a[0][y]) =
        FETCH_FLOAT4(a_share[0][0][ty * TThreadSizeM + y]);
  }

#pragma unroll
  for (int x = 0; x < TThreadSizeN; x += kFloatsPerCopy) {
    FETCH_FLOAT4(reg_b[0][x]) =
        FETCH_FLOAT4(b_share[0][0][tx * TThreadSizeN + x]);
  }

  // outer iteration with blockK stride
  int write_stage_idx = 1;
  int tile_k_idx = 0;
  do {
    tile_k_idx += TBlockSizeK;

    if (tile_k_idx < K) {
#pragma unroll
      for (int i = 0; i < TBlockSizeM; i += a_share_copy_row_stride) {
        int ldg_index = i / a_share_copy_row_stride * kFloatsPerCopy;
        FETCH_FLOAT4(ldg_a[ldg_index]) =
            FETCH_FLOAT4(A[OFFSET(i + a_share_copy_start_row + by * TBlockSizeM,
                                  a_share_copy_col + tile_k_idx, K)]);
      }
    }

    int load_stage_idx = write_stage_idx ^ 1;
// inner iteration with single step stride on k
#pragma unroll
    for (int j = 0; j < TBlockSizeK - 1; ++j) {
#pragma unroll
      for (int y = 0; y < TThreadSizeM; y += kFloatsPerCopy) {
        FETCH_FLOAT4(reg_a[(j + 1) % 2][y]) =
            FETCH_FLOAT4(a_share[load_stage_idx][j + 1][ty * TThreadSizeM + y]);
      }

#pragma unroll
      for (int x = 0; x < TThreadSizeN; x += kFloatsPerCopy) {
        FETCH_FLOAT4(reg_b[(j + 1) % 2][x]) =
            FETCH_FLOAT4(b_share[load_stage_idx][j + 1][tx * TThreadSizeN + x]);
      }

#pragma unroll
      for (int y = 0; y < TThreadSizeM; ++y) {
#pragma unroll
        for (int x = 0; x < TThreadSizeN; ++x) {
          accum[y][x] += reg_a[j % 2][y] * reg_b[j % 2][x];
        }
      }
    }

    if (tile_k_idx < K) {
#pragma unroll
      for (int i = 0; i < TBlockSizeM; i += a_share_copy_row_stride) {
        int ldg_index = i / a_share_copy_row_stride * kFloatsPerCopy;
        a_share[write_stage_idx][a_share_copy_col][a_share_copy_start_row + i] =
            ldg_a[ldg_index];
        a_share[write_stage_idx][a_share_copy_col + 1]
               [a_share_copy_start_row + i] = ldg_a[ldg_index + 1];
        a_share[write_stage_idx][a_share_copy_col + 2]
               [a_share_copy_start_row + i] = ldg_a[ldg_index + 2];
        a_share[write_stage_idx][a_share_copy_col + 3]
               [a_share_copy_start_row + i] = ldg_a[ldg_index + 3];
      }

#pragma unroll
      for (int i = b_share_copy_start_row; i < TBlockSizeK;
           i += b_share_copy_row_stride) {
        FETCH_FLOAT4(b_share[write_stage_idx][i][b_share_copy_col]) =
            FETCH_FLOAT4(B[OFFSET(i + tile_k_idx,
                                  b_share_copy_col + bx * TBlockSizeN, N)]);
      }
      __syncthreads();
      write_stage_idx ^= 1;
    }

#pragma unroll
    for (int y = 0; y < TThreadSizeM; y += kFloatsPerCopy) {
      FETCH_FLOAT4(reg_a[0][y]) =
          FETCH_FLOAT4(a_share[load_stage_idx ^ 1][0][ty * TThreadSizeM + y]);
    }

#pragma unroll
    for (int x = 0; x < TThreadSizeN; x += kFloatsPerCopy) {
      FETCH_FLOAT4(reg_b[0][x]) =
          FETCH_FLOAT4(b_share[load_stage_idx ^ 1][0][tx * TThreadSizeN + x]);
    }

#pragma unroll
    for (int y = 0; y < TThreadSizeM; ++y) {
#pragma unroll
      for (int x = 0; x < TThreadSizeN; ++x) {
        accum[y][x] +=
            reg_a[(TBlockSizeK - 1) % 2][y] * reg_b[(TBlockSizeK - 1) % 2][x];
      }
    }
  } while (tile_k_idx < K);

#pragma unroll
  for (int y = 0; y < TThreadSizeM; ++y) {
#pragma unroll
    for (int x = 0; x < TThreadSizeN; x += kFloatsPerCopy) {
      FETCH_FLOAT4(C[OFFSET(TBlockSizeM * by + TThreadSizeM * ty + y,
                            TBlockSizeN * bx + TThreadSizeN * tx + x, N)]) =
          FETCH_FLOAT4(accum[y][x]);
    }
  }
}

void Gemm() {
  const int M = 1 << 12;
  const int N = 1 << 13;
  const int K = 1 << 8;
  thrust::universal_vector<float> A(M * K);
  thrust::universal_vector<float> B(K * N);
  thrust::universal_vector<float> C(M * N);
  thrust::universal_vector<float> ACompare(M * K);
  thrust::universal_vector<float> BCompare(K * N);
  thrust::universal_vector<float> CCompare(M * N);

  float a_val, b_val;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      a_val = ((std::rand() & ((1 << 6) - 1)) - (1 << 5)) * 1.f / (1 << 8);
      A[i * K + j] = a_val;
      ACompare[j * M + i] = a_val;
    }
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      b_val = ((std::rand() & ((1 << 6) - 1)) - (1 << 5)) * 1.f / (1 << 8);
      B[i * N + j] = b_val;
      BCompare[j * K + i] = b_val;
    }
  }

  checkCudaErrors(cudaDeviceSynchronize());

  dim3 dim_block(128 / 8, 128 / 8);
  dim3 dim_grid(N / 128, M / 128);

  cudaStream_t stream1, stream2;
  checkCudaErrors(cudaStreamCreate(&stream1));
  checkCudaErrors(cudaStreamCreate(&stream2));

  cudaEvent_t start1, stop1, start2, stop2;
  checkCudaErrors(cudaEventCreate(&start1));
  checkCudaErrors(cudaEventCreate(&stop1));
  checkCudaErrors(cudaEventCreate(&start2));
  checkCudaErrors(cudaEventCreate(&stop2));

  // start cublas gemm
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasHandle_t handle = NULL;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream2));

  checkCudaErrors(cudaStreamSynchronize(stream2));
  checkCudaErrors(cudaEventRecord(start2, stream2));

  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                           thrust::raw_pointer_cast(ACompare.data()), M,
                           thrust::raw_pointer_cast(BCompare.data()), K, &beta,
                           thrust::raw_pointer_cast(CCompare.data()), M));
  checkCudaErrors(cudaEventRecord(stop2, stream2));

  checkCudaErrors(cudaStreamSynchronize(stream2));

  // start self implemented gemm
  checkCudaErrors(cudaStreamSynchronize(stream1));
  checkCudaErrors(cudaEventRecord(start1, stream1));
  GemmKernel<128, 128, 16, 8, 8><<<dim_grid, dim_block, 0, stream1>>>(
      thrust::raw_pointer_cast(A.data()), thrust::raw_pointer_cast(B.data()),
      thrust::raw_pointer_cast(C.data()), M, N, K);
  checkCudaErrors(cudaEventRecord(stop1, stream1));
  checkCudaErrors(cudaStreamSynchronize(stream1));

  // perf counting
  float elapsed_time1 = 0.0f, elapsed_time2 = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time1, start1, stop1));
  printf("self implemented gemm cost %.3f ms\n", elapsed_time1);

  checkCudaErrors(cudaEventElapsedTime(&elapsed_time2, start2, stop2));
  printf("cublas gemm cost %.3f ms\n", elapsed_time2);

  checkCudaErrors(cudaDeviceSynchronize());

  float acc = 0.f;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      acc += C[i * N + j];
    }
  }

  fprintf(stderr, "self implemented gemm: c = %.3f\n", acc);

  acc = 0.f;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      acc += CCompare[j * M + i];
    }
  }

  fprintf(stderr, "cublas sgemm: c = %.3f\n", acc);
  fprintf(stderr,
          "!!! cublass cost more perhaps caused to self implemented gemm "
          "ignoring the copy "
          "of C matric and scaling with alpha and beta coeffient\n");

  CUBLAS_CHECK(cublasDestroy(handle));
  checkCudaErrors(cudaEventDestroy(start1));
  checkCudaErrors(cudaEventDestroy(stop1));
  checkCudaErrors(cudaEventDestroy(start2));
  checkCudaErrors(cudaEventDestroy(stop2));

  checkCudaErrors(cudaStreamDestroy(stream1));
  checkCudaErrors(cudaStreamDestroy(stream2));
}

enum Demo {
  REDUCE_SUM,
  HELLO_WORLD,
  GEMM,
  TRANSPOSE,
};

int main(int argc, char *argv[]) {
  std::map<std::string, Demo> demo_map = {{
                                              "reduce_sum",
                                              REDUCE_SUM,
                                          },
                                          {

                                              "hello_world",
                                              HELLO_WORLD,
                                          },
                                          {
                                              "gemm",
                                              GEMM,
                                          },
                                          {
                                              "transpose",
                                              TRANSPOSE,
                                          }};
  if (argc != 2) {
    printf(USAGE);
    return 1;
  }

  if (demo_map.find(std::string(argv[1])) == demo_map.end()) {
    printf(USAGE);
    return 1;
  }

  Demo demo_name = demo_map[std::string(argv[1])];

  switch (demo_name) {
    case Demo::REDUCE_SUM:
      ReduceSum();
      break;
    case Demo::HELLO_WORLD:
      HelloWorld();
      break;
    case Demo::TRANSPOSE:
      TransPose();
      break;
    case Demo::GEMM:
      Gemm();
      break;
    default:
      printf(USAGE);
      return 1;
  }

  return 0;
}