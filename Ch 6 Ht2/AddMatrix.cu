#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>


#define M 20
#define N 20
#define BLOCK_SIZE 20
#define BASE_TYPE int 


using namespace std;

// kernel
__global__ void Add(const BASE_TYPE* a, const BASE_TYPE* b, BASE_TYPE* c)
{
	int idx = N * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	c[idx] = a[idx] + b[idx];
}


int main()
{
	// переменные на CPU
	BASE_TYPE a[M][N] = { 0 };
	BASE_TYPE b[M][N] = { 0 };
	BASE_TYPE c[M][N] = { 0 };

	// инициализация rand для float
	srand(time(0));
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i][j] = (BASE_TYPE)rand() %100+14;
			b[i][j] = (BASE_TYPE)rand() %100+23;
		}
	}

	BASE_TYPE* dev_a = NULL, * dev_b = NULL, *dev_c = NULL;

	//выделение памяти на GPU
	size_t size = N * M * sizeof(BASE_TYPE);
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);

	// копирование информации с CPU на GPU
	cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);


	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(N / BLOCK_SIZE, M / BLOCK_SIZE);

	// вызов ядра
	Add << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_c);

	// копирование результата работы ядра с GPU на CPU
	cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);

	// вывод информации
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
			cout << c[i][j] << ' ';
		cout << endl;
	}
	cout << endl;

	// очищаем память на GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}