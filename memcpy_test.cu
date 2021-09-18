#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

using namespace std;

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <cstring>

__global__ void mem_transfer_test(int* input) {
	int tid = threadIdx.x;
	int offset = blockDim.x * blockIdx.x;

	int gid = offset + tid;
	printf("threadIdx_X: %d, value : %d, GlobalIdx_X: %d \n", tid, input[gid], gid);
}

int main() {
	int size = 128;
	int array_byte_size = sizeof(int) * size;

	int * h_input;
	h_input = (int*)malloc(array_byte_size);

	time_t t;
	srand((unsigned)time(&t));
	//	srand(time(NULL));

	for (int i = 0; i < size; i++) {
		h_input[i] = (int)(rand() & 0xff);
	}
	//cout << h_input;
	int * d_input;
	cudaMalloc((void**)&d_input, array_byte_size);
	
	cudaMemcpy(d_input, h_input, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(64);
	dim3 grid(2);

	mem_transfer_test << <grid, block >> > (d_input);
	cudaDeviceSynchronize();

	cudaFree(d_input);
	free(h_input);

	cudaDeviceReset();
	cout << *h_input;
	return 0;
	
}