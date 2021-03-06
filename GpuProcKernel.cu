
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
#include "nvt_events.hpp"

#define M_PI				3.14159265358979323846  /* pi */
#define REGION_SIZE			4
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//int FIXED = 16;
//int ONE = 1 << FIXED;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	//fprintf(stderr, "Checking\n");
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		getchar();
		if (abort) exit(code);
	}
}

//__global__ void rgb2GrayKernel(unsigned char* dest_data, unsigned char* src_data, int rows, int cols, int chunkRows, int offset);

__global__ void rgb2GrayKernel(unsigned char* dest_data, unsigned char* src_data, int rows, int cols, int chunkRows, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	src_data += 3 * offset;
	if (idx < rows * cols){
		dest_data[idx] = 0.299 * src_data[3 * idx + 2] + 0.587 * src_data[3 * idx + 1] + 0.114 * src_data[3 * idx];
	}
}

__global__ void frgb2GrayKernel(float* dest_data, unsigned char* src_data, int rows, int cols, int chunkRows, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//src_data += 3 * offset;
	dest_data[idx] = 0.299 * src_data[3 * idx + 2] + 0.587 * src_data[3 * idx + 1] + 0.114 * src_data[3 * idx];
	return;
}

__global__ void reverseKernel(unsigned char* dest_data, unsigned char* src_data, int srcN, int chunkRows, int offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (3 * idx + (offset * 3) < srcN){
		dest_data[3 * idx + 0] = 255 - src_data[3 * idx + (offset * 3) + 0];
		dest_data[3 * idx + 1] = 255 - src_data[3 * idx + (offset * 3) + 1];
		dest_data[3 * idx + 2] = 255 - src_data[3 * idx + (offset * 3) + 2];
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

__global__ void fdirectResizeKernel(float* dest_data, float* src_data, int srcRows, int srcCols, int destRows, int destCols, int pix_begin, int src_begin){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int pix_local = pix_begin + idx;
	float rRow = (float)srcRows / destRows;
	float rCol = (float)srcCols / destCols;

	int sRow = ((pix_local) / destCols) * rRow;
	int sCol = ((pix_local) % destCols) * rCol;

	int src_local = (sRow * srcCols + sCol) - src_begin;
	dest_data[idx] = src_data[src_local];
	return;
	//if ((pix_local) / destCols == 11 && (pix_local) % destCols <= 307){
	//	printf("    pix_local: %d, %d;  sRow: %d, sCol: %d\n", (pix_local / destCols), (pix_local % destCols), sRow, sCol);
	//	dest_data[idx] = 255.0;
	//}

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
	return;
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
	float tmp = 0;
	for (int k = -mink; k <= maxk; k++){
		for (int l = -minl; l <= maxl; l++){
			tmp += *(gKernel + (k + filter_size) * (2 * filter_size + 1) + (l + filter_size)) * src_data[(idx + offset) + 3 * (k * cols + l)];
		}
	}
	dest_data[idx] = (unsigned char)tmp;
	//}
	/*else{
	dest_data[idx] = src_data[idx];
	}*/
}

__global__ void fGaussianFilterKernel(float* dest_data, float* src_data, double* gKernel, int filter_size, int rows, int cols, int src_begin, int pix_begin){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	/*if ((idx / 3) / cols > filter_size &&
	(idx / 3) / cols < rows - filter_size &&
	(idx / 3) % cols > filter_size &&
	(idx / 3) % cols < cols - filter_size ){*/
	int offset = 0;

	int i = ((idx + pix_begin)) / cols;
	int j = ((idx + pix_begin)) % cols;

	int local = pix_begin - src_begin;

	int maxk = min(filter_size, rows - i);
	int mink = min(filter_size, i);
	int maxl = min(filter_size, cols - j);
	int minl = min(filter_size, j);
	float tmp = 0;
	for (int k = -mink; k <= maxk; k++){
		for (int l = -minl; l <= maxl; l++){
			tmp += gKernel[(k + filter_size) * (2 * filter_size + 1) + (l + filter_size)] * src_data[local + idx + (k * cols + l)];
			//tmp += *(gKernel + (k + filter_size) * (2 * filter_size + 1) + (l + filter_size)) * src_data[(idx + offset) + (k * cols + l)];
		}
	}
	dest_data[idx] = tmp;
	return;
	//}
	/*else{
	dest_data[idx] = src_data[idx];
	}*/
}

__global__ void myConv2Kernel(float* deviceTemp, float* deviceLarge, float* deviceSmall, int tRows, int tCols, int lRows, int lCols, int sRows, int sCols, int pix_begin, int lrg_begin){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float val = 0;
	int value = 0;
	int FIXED = 16;
	int ONE = 1 << FIXED;

	int stRow, sdRow, stCol, sdCol;
	int mtRow, mdRow, mtCol, mdCol;
	int ks, km, ls, lm;
	int k, l, m, n;
	int loc_m, loc_n, loc_k, loc_l;
	int loc_t, loc_s;

	int i = (pix_begin + idx) / tCols;
	int j = (pix_begin + idx) % tCols;

	if (pix_begin + idx > tRows * tCols){
		return;
	}

	stRow = i - sRows + 1;
	sdRow = max(0, stRow);
	stCol = j - sCols + 1;
	sdCol = max(0, stCol);
	mtRow = stRow + sRows;
	mdRow = min(lRows, stRow + sRows);
	mtCol = stCol + sCols;
	mdCol = min(lCols, stCol + sCols);
	ks = sdRow - stRow; km = sRows - (mtRow - mdRow);
	ls = sdCol - stCol; lm = sCols - (mtCol - mdCol);

	for (k = ks, m = sdRow; k < km; k++, m++){
		for (l = ls, n = sdCol; l < lm; l++, n++){
			//float largeval = large.at<float>(m, n);
			//float smallval = small.at<float>(k, l);
			loc_t = (m * lCols + n) - lrg_begin;
			loc_s = (sRows - k - 1) * sCols + (sCols - l - 1);

			//val += large.at<float>(m, n) * small.at<float>(sRows - k - 1, sCols - l - 1);
			//val += deviceLarge[loc_t] * deviceSmall[loc_s];
			value += (deviceLarge[loc_t] * deviceSmall[loc_s]) * ONE;
		}
	}

	//deviceTemp[idx] = val;
	deviceTemp[idx] = value / ONE;

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

__global__ void kMeansCountingKernelFixed(unsigned char* src_data, unsigned char* k_index, int* k_count, int* k_colors, bool* convergence, int k_means, int srcRows, int srcCols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int FIXED = 16;
	int ONE = 1 << FIXED;

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
		val += (b2 - ((float)k_colors[3 * group + 0] / ONE)) * (b2 - ((float)k_colors[3 * group + 0] / ONE));
		val += (g2 - ((float)k_colors[3 * group + 1] / ONE)) * (g2 - ((float)k_colors[3 * group + 1] / ONE));
		val += (r2 - ((float)k_colors[3 * group + 2] / ONE)) * (r2 - ((float)k_colors[3 * group + 2] / ONE));

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

__global__ void kMeansGroupAdjustKernelFixed(unsigned char* src_data, unsigned char* k_index, int* k_count, int* k_colors, int k_means, int srcRows, int srcCols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int FIXED = 16;
	int ONE = 1 << FIXED;

	int i = (idx) / srcCols;
	int j = (idx) % srcCols;

	int group = k_index[i * srcCols + j];
	int group_count = k_count[group];
	for (int color = 0; color < 3; color++){
		//float src_val = src_data[3 * (i * srcCols + j) + color];
		int src_val = src_data[3 * (i * srcCols + j) + color] * ONE;
		int val = src_val / (group_count);
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

__global__ void kMeansOutputKernelFixed(unsigned char* dest_data, unsigned char* k_index, int* k_colors, int srcRows, int srcCols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int FIXED = 16;
	int ONE = 1 << FIXED;

	int i = (idx) / srcCols;
	int j = (idx) % srcCols;

	int group = k_index[i * srcCols + j];
	for (int color = 0; color < 3; color++){
		dest_data[3 * (i * srcCols + j) + color] = (unsigned char)(k_colors[3 * group + color] / ONE);
	}
}

__global__ void mySiftDOGKernel(float* curr_data, float* next_data, float* dog_data){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	dog_data[idx] = 0.0 + (next_data[idx] - curr_data[idx]);
	return;
}

__global__ void testKernel(){
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	return;
}

__global__ void mySiftKeypointsKernel(float* prev_data, float* curr_data, float* next_data, char* answers, int curRows, int curCols, int pix_begin, int src_begin, int block_begin, int keybits){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int pix_loc = pix_begin + idx;
	int src_loc = pix_loc - src_begin;
	int i = pix_loc / curCols, j = pix_loc % curCols;

	if ( i != 0 && i != curRows - 1 && j != 0 && j != curCols - 1 ){
	//if (true){
		int block_slot = (pix_loc / keybits) - block_begin;
		int block_loc = (pix_loc % keybits);
		float val = curr_data[src_loc];
		//unsigned int match = (unsigned int)exp2f(block_loc);

		int counter = 0;

		float val_c = 0;
		for (int k = -1; k <= 1; k++){
			for (int l = -1; l <= 1; l++){
				val_c = prev_data[src_loc + (k * curCols) + l] - val;
				if (val_c > 0) counter += 1;
				else if (val_c < 0) counter -= 1;

				val_c = curr_data[src_loc + (k * curCols) + l] - val;
				if (val_c > 0) counter += 1;
				else if (val_c < 0) counter -= 1;

				val_c = next_data[src_loc + (k * curCols) + l] - val;
				if (val_c > 0) counter += 1;
				else if (val_c < 0) counter -= 1;
			}
		}

		if (abs(counter) == 26){
			answers[idx] = 1;
			return;
		}
		else{
			answers[idx] = 0;
			return;
		}

	}
	else{
		answers[idx] = 0;
		return;
	}

}

__global__ void mySiftOrMagKernel(float* curr_data, float* or_mag, int curRows, int curCols, int pix_begin, int pix_end, int src_begin, int src_end){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int pix_local = idx + pix_begin;

	int src_local = pix_local - src_begin;

	int i = pix_local / curCols;
	int j = pix_local % curCols;

	int pi = 1, pj = 1, mi = 1, mj = 1;

	if (i == 0) mi = 0;
	else if (i == curRows - 1) pi = 0;

	if (j == 0) mj = 0;
	else if (j == curCols - 1) pj = 0;

	//float val = pow(curr_data[src_local + (pi * curCols)] - curr_data[src_local - (mi * curCols)], 2) + pow(curr_data[src_local + pj] - curr_data[src_local - mj], 2);
	float val = 0;
	val += (curr_data[src_local + (pi * curCols)] - curr_data[src_local - (mi * curCols)]) * (curr_data[src_local + (pi * curCols)] - curr_data[src_local - (mi * curCols)]);
	val += (curr_data[src_local + pj] - curr_data[src_local - mj]) * (curr_data[src_local + pj] - curr_data[src_local - mj]);
	//val = sqrt(val);
	or_mag[2 * idx + 1] = sqrt(val);

	float val1 = curr_data[src_local + (pi * curCols)] - curr_data[src_local - (mi * curCols)];
	float val2 = curr_data[src_local + pj] - curr_data[src_local - mj];
	val = atan2f(val2, val1);
	if (val < 0){
		val = (2 * M_PI) + val;
	}
	else if (val > 2 * M_PI){
		val = val - (2 * M_PI);
	}

	or_mag[2 * idx] = val;
	
}

__global__ void mySiftCountingKernel(unsigned int* data, int* count, int exp, int d){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < d){
		atomicAdd((count + ((data[idx] / exp) % 10)), 1);
	}
}

__global__ void mySiftCountSortKernel(unsigned int* data, unsigned int* out_data, unsigned int* index, unsigned int* out_index, int* count, int exp, int d){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < d){
		int loc = atomicAdd(count + ((data[idx] / exp) % 10), -1);
		loc -= 1;
		out_data[loc] = data[idx];
		out_index[loc] = index[idx];
	}
}

__global__ void mySiftCountSortSwitchKernel(unsigned int* data, unsigned int* out_data, unsigned int* index, unsigned int* out_index, int* count, int exp, int d){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < d){
		data[idx] = out_data[idx];
		index[idx] = out_index[idx];
	}
}

#endif

