#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <iostream>
#include <ctime>

#define N 100

using namespace std;

// считаем y = sqrt(1-x^2)
__global__ void dzeta(double* zeta, double *s) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	zeta[index] = 1.0 / powf((index + 1), *s);
}


int main()
{
	// переменные на CPU
	double zeta[N]; //zeta
	double s=5; //s - степень
	double sum = 0; // для вычисления суммы

	// переменные на GPU
	double* dev_zeta, * dev_s;

	// выделяем память на GPU
	cudaMalloc((void**)&dev_zeta, N*sizeof(double));
	cudaMalloc((void**)&dev_s, sizeof(double));

	// копирование информации с CPU на GPU
	cudaMemcpy(dev_s, &s, sizeof(double), cudaMemcpyHostToDevice);


	// вызов ядра
	dzeta << < 1, N >> > (dev_zeta, dev_s);

	// копирование результата работы ядра с GPU на CPU
	cudaMemcpy(&zeta, dev_zeta, N*sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i)
		sum += zeta[i];

	// вывод информации
	cout << "Zeta functions is: " << sum << endl;
	// очищение памяти на GPU
	cudaFree(dev_zeta);
	cudaFree(dev_s);

	return 0;
}