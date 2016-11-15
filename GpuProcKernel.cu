
//#include <cuda.h>
//#include <cuda_runtime.h>
#ifndef _GPU_PROC_KERNEL_CU_
#define _GPU_PROC_KERNEL_CU_

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <atomic>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//__global__ void rgb2GrayKernel(unsigned char* dest_data, unsigned char* src_data, int rows, int cols, int chunkRows, int offset);

__global__ void rgb2GrayKernel(unsigned char* dest_data, unsigned char* src_data, int rows, int cols, int chunkRows, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	src_data += 3 * offset;
	if (idx < rows * cols){
		dest_data[idx] = 0.299 * src_data[3 * idx + 2] + 0.587 * src_data[3 * idx + 1] + 0.114 * src_data[3 * idx];
	}
}

__global__ void reverseKernel(unsigned char* dest_data, unsigned char* src_data, int srcN, int chunkRows, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx + (offset * 3) < srcN){
		dest_data[idx] = 255 - src_data[idx + (offset * 3)];
	}
}

__global__ void gammaCorrectionKernel(unsigned char* dest_data, unsigned char* src_data, int srcRows, int srcCols, double gamma, int chunkRows, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double gammaCorrect = 1.00 / gamma;

	if (idx + offset < 3 * srcRows * srcCols){
		double color = (double)src_data[idx + offset];
		unsigned char val = 255 * pow((color / 255), gammaCorrect);
		dest_data[idx] = val;
	}
}

__global__ void directResizeKernel(unsigned char* dest_data, unsigned char* src_data, int srcRows, int srcCols, int destRows, int destCols, int chunkRows, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double rRow = (double)srcRows / destRows;
	double rCol = (double)srcCols / destCols;

	if (idx + offset < 3 * destRows * destCols){
		int sRow = (((idx + offset) / 3) / destCols) * rRow;
		int sCol = (((idx + offset) / 3) % destCols) * rCol;
		dest_data[idx] = src_data[3 * (sRow * srcCols + sCol) + (idx + offset) % 3];
	}
}

__global__ void linearResizeKernel(unsigned char* dest_data, unsigned char* src_data, int srcRows, int srcCols, int destRows, int destCols, int chunkRows, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double rRow = (double)srcRows / destRows;
	double rCol = (double)srcCols / destCols;

	if (idx < 3 * destRows * destCols){
		double dsRow = (((offset + idx) / 3) / destCols) * rRow;
		double dsCol = (((offset + idx) / 3) % destCols) * rCol;
		int sRow = (int)dsRow;
		int sCol = (int)dsCol;

		double deltaI = dsRow - sRow;
		double deltaJ = dsCol - sCol;

		if (deltaI + deltaJ < 0.0000000001){
			dest_data[idx] = src_data[3 * (sRow * srcCols + sCol) + (idx + offset) % 3];
		}
		else{
			unsigned char val = 0;
			double area1 = (1 - deltaI) * (1 - deltaJ);
			double area2 = deltaI * (1 - deltaJ);
			double area3 = deltaI * deltaJ;
			double area4 = (1 - deltaI) * deltaJ;

			val += area1 * src_data[3 * (sRow * srcCols + sCol) + (idx + offset) % 3];
			val += area2 * src_data[3 * ((sRow + 1) * srcCols + sCol) + (idx + offset) % 3];
			val += area3 * src_data[3 * ((sRow + 1) * srcCols + sCol + 1) + (idx + offset) % 3];
			val += area4 * src_data[3 * (sRow * srcCols + sCol + 1) + (idx + offset) % 3];

			dest_data[idx] = val;
		}
	}
}

__global__ void gaussianFilterKernel(unsigned char* dest_data, unsigned char* src_data, double* gKernel, int filter_size, int rows, int cols, int chunkRows, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	/*if ((idx / 3) / cols > filter_size &&
	(idx / 3) / cols < rows - filter_size &&
	(idx / 3) % cols > filter_size &&
	(idx / 3) % cols < cols - filter_size ){*/

	int i = ((idx + offset) / 3) / cols;
	int j = ((idx + offset) / 3) % cols;

	int maxk = min(filter_size, rows - i);
	int mink = min(filter_size, i);
	int maxl = min(filter_size, cols - j);
	int minl = min(filter_size, j);
	unsigned char tmp = 0;
	for (int k = -mink; k <= maxk; k++){
		for (int l = -minl; l <= maxl; l++){
			tmp += *(gKernel + (k + filter_size) * (2 * filter_size + 1) + (l + filter_size)) * src_data[(idx + offset) + 3 * (k * cols + l)];
		}
	}
	dest_data[idx] = tmp;
	//}
	/*else{
	dest_data[idx] = src_data[idx];
	}*/
}

__global__ void sobelGradientKernel(short* temp_data, unsigned char* src_data, int* sobel_x, int* sobel_y, double* rangeMin, double* rangeMax, int srcRows, int srcCols, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = ((idx + offset) / 3) / srcCols;
	int j = ((idx + offset) / 3) % srcCols;

	float tmpx = 0;
	float tmpy = 0;
	int maxk = min(1, srcRows - i);
	int mink = min(1, i);
	int maxl = min(1, srcCols - j);
	int minl = min(1, j);

	for (int k = -mink; k <= maxk; k++){
		for (int l = -minl; l <= maxl; l++){
			//if (k > 1 || k < -1 || l > 1 || l < -1)
			//printf("Error in Sobel!\n");
			tmpx = tmpx + sobel_x[3 * (k + 1) + (l + 1)] * src_data[(idx + offset) + 3 * (k * srcCols + l)];
			tmpy = tmpy + sobel_y[3 * (k + 1) + (l + 1)] * src_data[(idx + offset) + 3 * (k * srcCols + l)];
		}
	}
	int value = sqrt((tmpx*tmpx) + (tmpy*tmpy));
	//printf("Value: %d\n",value);

	temp_data[idx] = value;

}

__global__ void sobelRangeKernel(unsigned char* dest_data, short* temp_data, double rangeMin, double rangeMax, double thresh_min, double thresh_max, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	dest_data[idx] = 255;

	int value = 255 * ((temp_data[idx] + rangeMin) / (rangeMax - rangeMin));
	//int value = 255 * ((temp_data[idx + offset] + rangeMin) / (rangeMax - rangeMin));
	if (value >= thresh_max){
		value = 255;
	}
	else if (value < thresh_min){
		value = 0;
	}
	dest_data[idx] = value;
}

__global__ void kMeansCountingKernel(unsigned char* src_data, unsigned char* k_index, int* k_count, float* k_colors, bool* convergence, int k_means, int srcRows, int srcCols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = (idx) / srcCols;
	int j = (idx) % srcCols;

	if (i >= srcRows)
		return;

	float b2 = src_data[3 * (i * srcCols + j)];
	float g2 = src_data[3 * (i * srcCols + j) + 1];
	float r2 = src_data[3 * (i * srcCols + j) + 2];
	float min_dist = FLT_MAX;
	unsigned char new_index = k_index[i * srcCols + j];

	for (int group = 0; group < k_means; group++){
		float val = 0;
		val += (b2 - k_colors[3 * group]) * (b2 - k_colors[3 * group]);
		val += (g2 - k_colors[3 * group + 1]) * (g2 - k_colors[3 * group + 1]);
		val += (r2 - k_colors[3 * group + 2]) * (r2 - k_colors[3 * group + 2]);

		float dist = sqrtf(val);
		if (dist < min_dist){
			min_dist = dist;
			new_index = group;
		}
	}
	if (k_index[i * srcCols + j] != new_index){
		k_index[i * srcCols + j] = new_index;
		//printf("New Index: %d", new_index);
		//atomicAdd((hits + new_index), 1);
		//if (iter > 60)
		//	printf(" (%d, %d) \n",i,j);
		convergence[0] = false;
	}
	atomicAdd((k_count + new_index), 1);
}

__global__ void kMeansCountingKernelOld(unsigned char* src_data, unsigned char* k_index, int* k_count, float* k_colors, bool* convergence, int k_means, int srcRows, int srcCols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = (idx) / srcCols;
	int j = (idx) % srcCols;

	float b2 = src_data[3 * (i * srcCols + j)];
	float g2 = src_data[3 * (i * srcCols + j) + 1];
	float r2 = src_data[3 * (i * srcCols + j) + 2];
	float min_dist = FLT_MAX;
	unsigned char new_index;

	for (int group = 0; group < k_means; group++){
		float b1 = k_colors[3 * group];
		float g1 = k_colors[3 * group + 1];
		float r1 = k_colors[3 * group + 2];

		float dist = std::sqrt(pow(r2 - r1, 2) + pow(g2 - g1, 2) + pow(b2 - b1, 2));	//Combination of pow and sqrt is too much
		//float dist = 0;
		if (dist < min_dist){
			min_dist = dist;
			//k_index[i * srcCols + j] = group;
			new_index = group;
		}

	}
	if (k_index[i * srcCols + j] != new_index){
		k_index[i * srcCols + j] = new_index;
		//printf("New Index: %d", new_index);
		*convergence = false;
	}
	//k_count[new_index] += 1;
	atomicAdd((k_count + new_index), 1);
}

__global__ void kMeansGroupAdjustKernel(unsigned char* src_data, unsigned char* k_index, int* k_count, float* k_colors, int k_means, int srcRows, int srcCols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = (idx) / srcCols;
	int j = (idx) % srcCols;

	int group = k_index[i * srcCols + j];
	for (int color = 0; color < 3; color++){
		float src_val = src_data[3 * (i * srcCols + j) + color];
		int group_count = k_count[group];
		float val = src_val / group_count;
		//k_colors[3 * group + color] += val;
		atomicAdd((k_colors + (3 * group + color)), val);
	}
}

__global__ void kMeansOutputKernel(unsigned char* dest_data, unsigned char* k_index, float* k_colors, int srcRows, int srcCols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = (idx) / srcCols;
	int j = (idx) % srcCols;

	int group = k_index[i * srcCols + j];
	for (int color = 0; color < 3; color++){
		dest_data[3 * (i * srcCols + j) + color] = (unsigned char)k_colors[3 * group + color];
	}
}

#endif

