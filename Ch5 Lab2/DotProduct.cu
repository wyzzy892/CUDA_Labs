#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>

#define N 50

using namespace std;

__global__ void DotProduct(int* a, int* b, int* res) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        int component = a[index] * b[index];
        atomicAdd(res, component);
    }
}


int main()
{
    int a[N], b[N], res;  // объявляем массивы для CPU
    int* dev_a, * dev_b, * dev_res;  // объявляем массивы для GPU

    // инициализируем массивы a, b значениями
    for (int i = 0; i < N; i++) {
        a[i] = pow(i + 1, 2);
        b[i] = pow(i + 1, 2);
    }
    res = 0;

    //выделяем память на GPU
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_res, sizeof(int));

    //копируем данные из CPU на GPU
    cudaMemcpy(dev_a, &a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res, &res, sizeof(int), cudaMemcpyHostToDevice);

    //вызываем функцию ядра с 5 блоками по 10 нитей
    DotProduct << <5, 10 >> > (dev_a, dev_b, dev_res);

    //копируем данные обратно на хост
    cudaMemcpy(&res, dev_res, sizeof(int), cudaMemcpyDeviceToHost);

    //Вывод значения
    cout << res<<endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    return 0;
}
