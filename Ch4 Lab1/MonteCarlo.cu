#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <iostream>
#include <ctime>

#define N 360
#define M 500

using namespace std;

// Функция инициализации начальных состояний генератора случайных чисел для каждого потока
__global__ void init(curandState* state) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(1234, index, 0, &state[index]);
}

//Функция генерерующая случайное число из равномерного распределения 
__device__ double generate(curandState* state, int index) {
	curandState local_state = state[index];
	double random_value = curand_uniform_double(&local_state);
	state[index] = local_state;
	return random_value;
}

__global__ void MonteCarlo(curandState* state, int* n, int* n_circle) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N * M) {
		double x = generate(state, index);
		double y = generate(state, index);
		if (x * x + y * y <= 1)
			atomicAdd(n_circle, 1);
		atomicAdd(n, 1);
	}
}

int main()
{
	setlocale(LC_ALL, "rus");
	curandState* states;
	cudaMalloc((void**)&states, N * M * sizeof(curandState));
	int n = 0, n_circle = 0;
	int* dev_n, * dev_n_circle;
	cudaMalloc((void**)&dev_n, sizeof(int));
	cudaMalloc((void**)&dev_n_circle, sizeof(int));
	cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_n_circle, &n_circle, sizeof(int), cudaMemcpyHostToDevice);

	// замеряем время
	float gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);

	//инициализируем состояния
	init << <N, M >> > (states);

	//считаем точки
	MonteCarlo << <N, M >> > (states, dev_n, dev_n_circle);
	cudaMemcpy(&n, dev_n, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&n_circle, dev_n_circle, sizeof(int), cudaMemcpyDeviceToHost);

	// замеряем время
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	// считаем число pi
	cout << "Число пи приближенно равно " << 4.0 * n_circle / n<<endl<<gpu_elapsed_time/1000<<"";

	//Освобождаем память
	cudaFree(dev_n);
	cudaFree(dev_n_circle);
	return 0;
}