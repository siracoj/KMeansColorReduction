/* Author: Jared Siraco
   4/22/2014
*/

#include <stdio.h>
#include <stdlib.h>

#include <vector_types.h>
#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_devptrs.hpp> 



//CUDA
#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"


using namespace cv;


__global__ void fillClusters(gpu::PtrStepSz<uchar3> src, uchar4* d_cluster)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
	int y = threadIdx.y + blockIdx.y * blockDim.y; 
	int offset = x + y * blockDim.x * gridDim.x;

	uchar3 temp = src(x, y);

	d_cluster[offset].w = 0;
	d_cluster[offset].x = temp.x;
	d_cluster[offset].y = temp.y;
	d_cluster[offset].z = temp.z;
}

__global__ void colorChange(gpu::PtrStep<uchar3> dst, uchar4* cluster, uchar3* kpoints) 
{ 
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
	int y = threadIdx.y + blockIdx.y * blockDim.y; 
	int offset = x + y * blockDim.x * gridDim.x;
 
	int k = cluster[offset].w;

	dst(x, y) = kpoints[k];
} 

__global__ void kmeansInParallel_kernel(uchar4* cluster, uchar3* kpoints) 
{ 
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
	int y = threadIdx.y + blockIdx.y * blockDim.y; 
	int offset = x + y * blockDim.x * gridDim.x;
 
	int minDist = 255*255 + 255*255 + 255*255;
	int k = cluster[offset].w;

	for(int i =0; i<8; i++){
		int rdiff = cluster[offset].x - kpoints[i].x;
		int gdiff = cluster[offset].y - kpoints[i].y;
		int bdiff = cluster[offset].z - kpoints[i].z;
		int dist = rdiff*rdiff + gdiff*gdiff + bdiff*bdiff;
		if( dist < minDist){
			minDist = dist;
			k = i;
		}
	}
	cluster[offset].w = k;
} 
//uchar4* cluster = (uchar4*)malloc(320*240);
uchar4* d_cluster;

uchar3* kpoints = (uchar3*)malloc(8);
uchar3* d_kpoints;

void average()
{
	uchar3 sums[8] = {(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)};
	for(int i = 0;  i<9600*8; i++){
		int k = d_cluster[i].w;
		sums[k].x += d_cluster[i].x;
		sums[k].y += d_cluster[i].y;
		sums[k].z += d_cluster[i].z;
	}
	for(int i = 0; i<8; i++){
		sums[i].x = sums[i].x/9600;
		sums[i].y = sums[i].y/9600;
		sums[i].z = sums[i].z/9600;
	}
}

void kmeansInParallel_caller(uchar* src, uchar* dst, int width, int height) 
{ 
	dim3 grids(width/16, height/16); 
	dim3 threads(16,16);
	
	
	for(int i = 0; i<8; i++){
		int div = i+1;
		kpoints[i] = src(rows/div,cols/div);
	}

	cudaMalloc((void**)&d_cluster, sizeof(uchar4)*320*240);
	fillClusters<<<grids, threads>>>(src,d_cluster);

	cudaMalloc((void**)&d_kpoints, sizeof(uchar3)*8);
		
	for(int i = 0; i<10; i++){
		cudaMemcpy(d_kpoints, kpoints, sizeof(uchar3)*8, cudaMemcpyHostToDevice );
		
		kmeansInParallel_kernel<<<grids, threads>>>(d_cluster, d_kpoints); 
		average();
	}

	colorChange<<<grids, threads>>>(dst, d_cluster, d_kpoints);
	
 
	cudaDeviceSynchronize(); 
} 