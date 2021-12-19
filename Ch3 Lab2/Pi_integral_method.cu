#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <iostream>
#include <ctime>

#define N 1000

using namespace std;

// считаем y = sqrt(1-x^2)
__global__ void f(double* mas) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	double x = double(index) / double(N);
	mas[index] = sqrt(1 - x*x);
}


int main()
{
	double mas[N];
	double result=0;
	double* dev_mas;
	cudaMalloc((void**)&dev_mas, N*sizeof(double));

	//вызов ядра
	f << <1, N >> > (dev_mas);

	cudaMemcpy(&mas, dev_mas, N*sizeof(double), cudaMemcpyDeviceToHost);
	//проверка на ошибку
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("%s ", cudaGetErrorString(err));

	// копирование результата работы ядра с GPU на CPU
	cudaMemcpy(&mas, dev_mas, N * sizeof(double), cudaMemcpyDeviceToHost);
	//Считаем площадь
	for (int i = 0; i < N; i++) {
		result += 2*mas[i];
	}
	// вывод результата
	cout<<"Pi = "<<4.0 * result / (2*N)<<endl;
	// очищение памяти на GPU
	cudaFree(dev_mas);
	return 0;
}