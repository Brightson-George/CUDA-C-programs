#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <common.h>

__global__ void sum_two_array_gpu(int* a, int* b, int* c, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size) {
		c[gid] = a[gid] + b[gid];
	}
}

void sum_two_array_cpu(int* a, int* b, int* c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}


int main() {
	int size = 10000;
	int byte_size = sizeof(int) * size;
	int block_size = 128;

	int *h_a, *h_b, *gpu_results, *h_c;
	h_a = (int*)malloc(byte_size);
	h_b = (int*)malloc(byte_size);
	h_c = (int*)malloc(byte_size);
	gpu_results = (int*)malloc(byte_size);

	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < size; i++) {
		h_a[i] = (int)(rand() & 0xff);
		h_b[i] = (int)(rand() & 0xff);
	}

	sum_two_array_cpu(h_a, h_b, h_c, size);

	int* d_a, *d_b, *d_c;
	cudaMalloc((void**)&d_a, byte_size);
	cudaMalloc((void**)&d_b, byte_size);
	cudaMalloc((void**)&d_c, byte_size);

	cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice);

	dim3 block(block_size);
	dim3 grid(size / block_size + 1);

	sum_two_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);
	cudaDeviceSynchronize();

	cudaMemcpy(gpu_results, d_c, byte_size, cudaMemcpyDeviceToHost);

	// cpu and gpu array_sum comparison
	compare_arrays(gpu_results, h_c, size);

	/*for (int i = 0; i < size; i++) {
		if (gpu_results[i] == h_c[i]) {
			printf("the arrays are not same");
		}
	}*/

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(gpu_results);
	cudaDeviceReset();
	return 0;
}