#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void print_threadIDs() {
	printf("threadIdx_x : %d, threadIdx : %d, threadIdx : %d \n",
		threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
	int nx, ny;
	nx = 16;
	ny = 16;

	dim3 block(8, 8);
	dim3 grid(nx / block.x, ny / block.y);

	print_threadIDs << <grid, block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();

	return 0;
}