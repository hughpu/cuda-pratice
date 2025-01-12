#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include <cstdio>
#include <cudapractice/helper_cuda.cuh>
#include <vector>

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

int main() {
  HelloWorld();

  return 0;
}