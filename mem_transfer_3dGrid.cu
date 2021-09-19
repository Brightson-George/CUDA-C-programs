#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void mem_transfer_3dGrid_3dBlock(int* input) {
	int tid = threadIdx.x + (blockDim.x * threadIdx.y) + (threadIdx.z * blockDim.x * blockDim.y);

	int num_of_thread_in_block = blockDim.x * blockDim.y * blockDim.z;
	int block_offset = num_of_thread_in_block * blockIdx.x;

	int num_of_thread_in_rowblocks = gridDim.x * blockDim.x * blockDim.y * blockDim.z;
	int row_offset = num_of_thread_in_rowblocks * blockIdx.y;

	int num_of_thread_in_colxrow = gridDim.x * gridDim.y * blockDim.x * blockDim.y * blockDim.z;
	int z_offset = num_of_thread_in_colxrow * blockIdx.z;

	int gid = block_offset + row_offset + z_offset + tid;

	printf("threadIdx.x: %d, GlobalIdx: %d, value: %d \n", tid, gid, input[gid]);
}

int main() {
	int array_size = 64;
	int array_byte_size = sizeof(int) * array_size;

	int* h_input;
	h_input = (int*)malloc(array_byte_size); // memory allocation in host device 

	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < array_size; i++) {
		h_input[i] = (int)(rand() & 0xff);
	}

	int* d_input;
	cudaMalloc((void**)&d_input, array_byte_size);

	cudaMemcpy(d_input, h_input, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(2, 2, 2);
	dim3 grid(2, 2, 2);

	mem_transfer_3dGrid_3dBlock << <grid, block >> > (d_input);

	cudaDeviceSynchronize();
	
	cudaFree(d_input);
	free(h_input);

	cudaDeviceReset();

	return 0;




	
}

