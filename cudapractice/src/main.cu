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
      printf("transpose is not implemented yet");
      break;
    case Demo::GEMM:
      printf("gemm is not implemented yet");
      break;
    default:
      printf(USAGE);
      return 1;
  }

  return 0;
}