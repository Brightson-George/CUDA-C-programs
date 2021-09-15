#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void print_parameters() {
	printf("blockIdx_x: %d, blockIdx_y: %d, blockIdx_z: %d, blockDim_x: %d, blockDim_y: %d, GridDim_x: %d, GridDim_y: %d \n",
		blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
}

int main() {
	int nx, ny;
	nx = 16;
	ny = 16;

	dim3 block(8, 8);
	dim3 grid(nx / block.x, ny / block.y);

	print_parameters << <grid, block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}