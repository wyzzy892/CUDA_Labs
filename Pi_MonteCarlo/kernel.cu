#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <random>
#include <math.h>

using namespace std;

__global__ void setup_kernel(curandState* state)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(123456789, index, 0, &state[index]);
}

__global__ void Monte_Carlo(curandState* state, int* count, int m)
{
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	
	__shared__ int cache[256];
	cache[threadIdx.x] = 0;
	__syncthreads();

	unsigned int temp = 0;
	while (temp < m) {
		float x = curand_uniform(&state[index]);
		float y = curand_uniform(&state[index]);
		float r = x * x + y * y;

		if (r <= 1) {
			cache[threadIdx.x]++;
		}
		temp++;
	}
	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}

		i /= 2;
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicAdd(count, cache[0]);
	}
}

int main()
{
	setlocale(LC_ALL, "rus");
	unsigned int n = 256 * 256;
	unsigned int m = 20000;
	int* host_count;
	int* dev_count;
	curandState* dev_state;
	float pi;

	// выделяем память
	host_count = (int*)malloc(n * sizeof(int));
	cudaMalloc((void**)&dev_count, n * sizeof(int));
	cudaMalloc((void**)&dev_state, n * sizeof(curandState));
	cudaMemset(dev_count, 0, sizeof(int));

	// Переменные для отслеживания времени вычислений
	float gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);

	dim3 gridSize = 256;
	dim3 blockSize = 256;
	setup_kernel << < gridSize, blockSize >> > (dev_state);

	// Метод Монте-Карло
	Monte_Carlo << <gridSize, blockSize >> > (dev_state, dev_count, m);

	// Копируем результаты обратно на хост
	cudaMemcpy(host_count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	// Вывод результатов
	pi = *host_count * 4.0 / (n * m);
	cout << "Число Pi равно: " << pi << endl<<"Время вычислений: " << gpu_elapsed_time << endl;

	// Освобождаем память
	free(host_count);
	cudaFree(dev_count);
	cudaFree(dev_state);
	return 0;
}