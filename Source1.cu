#include "ImgProc.hpp"
#include "GpuProc.cu"
#include <stdio.h>
#include <opencv2\\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <chrono>


using namespace cv;
using namespace std;

#define FILTER_SIZE			3
#define M_PI				3.14159265358979323846  /* pi */
#define IMG_CHUNK			3110400	/* (1920 x 1080 x 3) / 2 */
#define THREADS_PER_BLOCK	256


//Boat from https://homepages.cae.wisc.edu/~ece533/images/boat.png;

//-- Cuda Device Code

/*__global__ void rgb2GrayKernel(uchar* dest_data, uchar* src_data, int rows, int cols, int chunkRows, int offset){
int idx = blockIdx.x * blockDim.x + threadIdx.x;

src_data += 3 * offset;
if (idx < rows * cols){
dest_data[idx] = 0.299 * src_data[3 * idx + 2] + 0.587 * src_data[3 * idx + 1] + 0.114 * src_data[3 * idx];
}
}*/

/*__global__ void reverseKernel(uchar* dest_data, uchar* src_data, int srcN, int chunkRows, int offset){
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx + (offset * 3) < srcN){
dest_data[idx] = 255 - src_data[idx + (offset * 3)];
}
}*/

/*__global__ void gammaCorrectionKernel(uchar* dest_data, uchar* src_data, int srcRows, int srcCols, double gamma, int chunkRows, int offset){
int idx = blockIdx.x * blockDim.x + threadIdx.x;
double gammaCorrect = 1.00 / gamma;

if (idx + offset < 3 * srcRows * srcCols){
double color = (double)src_data[idx + offset];
uchar val = 255 * pow((color / 255), gammaCorrect);
dest_data[idx] = val;
}
}*/

/*__global__ void directResizeKernel(uchar* dest_data, uchar* src_data, int srcRows, int srcCols, int destRows, int destCols, int chunkRows, int offset){
int idx = blockIdx.x * blockDim.x + threadIdx.x;
double rRow = (double)srcRows / destRows;
double rCol = (double)srcCols / destCols;

if (idx + offset < 3 * destRows * destCols){
int sRow = (((idx + offset) / 3) / destCols) * rRow;
int sCol = (((idx + offset) / 3) % destCols) * rCol;
dest_data[idx] = src_data[3 * (sRow * srcCols + sCol) + (idx + offset) % 3];
}
}*/

/*__global__ void linearResizeKernel(uchar* dest_data, uchar* src_data, int srcRows, int srcCols, int destRows, int destCols, int chunkRows, int offset){
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
uchar val = 0;
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
}*/

/*__global__ void gaussianFilterKernel(uchar* dest_data, uchar* src_data, double* gKernel, int filter_size, int rows, int cols, int chunkRows, int offset){
int idx = blockIdx.x * blockDim.x + threadIdx.x;

int i = ((idx + offset) / 3) / cols;
int j = ((idx + offset) / 3) % cols;

int maxk = min(filter_size, rows - i);
int mink = min(filter_size, i);
int maxl = min(filter_size, cols - j);
int minl = min(filter_size, j);
uchar tmp = 0;
for (int k = -mink; k <= maxk; k++){
for (int l = -minl; l <= maxl; l++){
tmp += *(gKernel + (k + filter_size) * (2 * filter_size + 1) + (l + filter_size)) * src_data[(idx + offset) + 3 * (k * cols + l)];
}
}
dest_data[idx] = tmp;
//}
}*/

/*__global__ void sobelGradientKernel(short* temp_data, uchar* src_data, int* sobel_x, int* sobel_y, double* rangeMin, double* rangeMax, int srcRows, int srcCols, int offset){
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

}*/

/*__global__ void sobelRangeKernel(uchar* dest_data, short* temp_data, double rangeMin, double rangeMax, double thresh_min, double thresh_max, int offset){
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
}*/

//__global__ void kMeansCountingKernel(uchar* src_data, uchar* k_index, int* k_count, int* hits, float* k_colors, bool* convergence, int k_means, int srcRows, int srcCols, int iter){
/*__global__ void kMeansCountingKernel(uchar* src_data, uchar* k_index, int* k_count, float* k_colors, bool* convergence, int k_means, int srcRows, int srcCols){
int idx = blockIdx.x * blockDim.x + threadIdx.x;

int i = (idx) / srcCols;
int j = (idx) % srcCols;

if (i >= srcRows)
return;

float b2 = src_data[3 * (i * srcCols + j)];
float g2 = src_data[3 * (i * srcCols + j) + 1];
float r2 = src_data[3 * (i * srcCols + j) + 2];
float min_dist = FLT_MAX;
uchar new_index = k_index[i * srcCols + j];

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
}*/

/*__global__ void kMeansCountingKernelOld(uchar* src_data, uchar* k_index, int* k_count, float* k_colors, bool* convergence, int k_means, int srcRows, int srcCols){
int idx = blockIdx.x * blockDim.x + threadIdx.x;

int i = (idx) / srcCols;
int j = (idx) % srcCols;

float b2 = src_data[3 * (i * srcCols + j)];
float g2 = src_data[3 * (i * srcCols + j) + 1];
float r2 = src_data[3 * (i * srcCols + j) + 2];
float min_dist = FLT_MAX;
uchar new_index;

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
}*/

/*__global__ void kMeansGroupAdjustKernel(uchar* src_data, uchar* k_index, int* k_count, float* k_colors, int k_means, int srcRows, int srcCols){
int idx = blockIdx.x * blockDim.x + threadIdx.x;

int i = (idx) / srcCols;
int j = (idx) % srcCols;

int group = k_index[i * srcCols + j];
for (int color = 0; color < 3; color++){
float src_val = src_data[3 * (i * srcCols + j) + color];
int group_count = k_count[group];
float val = src_val / group_count;
//k_colors[3 * group + color] += val;
atomicAdd((k_colors + (3 * group + color)),val);
}
}*/

/*__global__ void kMeansOutputKernel(uchar* dest_data, uchar* k_index, float* k_colors, int srcRows, int srcCols){
int idx = blockIdx.x * blockDim.x + threadIdx.x;

int i = (idx) / srcCols;
int j = (idx) % srcCols;

int group = k_index[i * srcCols + j];
for (int color = 0; color < 3; color++){
dest_data[3 * (i * srcCols + j) + color] = (uchar)k_colors[3 * group + color];
}
}*/

//-- Cuda Host Code

/*void cudaRgb2Gray(uchar* input, uchar* output, int srcRows, int srcCols){
uchar* deviceSrcData;
uchar* deviceDestData;
int threadsPerBlock = THREADS_PER_BLOCK;
int blocks = 0;
int chunkRows = 0;
int offset = 0;
int srcN = 3 * srcRows * srcCols;
cudaMalloc(&deviceSrcData, srcN*sizeof(uchar));
cudaMemcpy(deviceSrcData, input, srcN*sizeof(uchar), cudaMemcpyHostToDevice);

chunkRows = IMG_CHUNK / srcCols;
if (chunkRows == 0){
chunkRows = srcRows;
}
int rounds = ceil(srcRows / (double)chunkRows);

//int destN = min(1920 * 1080, srcRows * srcCols);
for (int step = 0; step < rounds; step++){
int destN = min(chunkRows * srcCols, srcRows * srcCols - offset);
if (destN <= 0)
break;

blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceDestData, destN*sizeof(uchar));
rgb2GrayKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, srcRows, srcCols, chunkRows, offset);
cudaMemcpy(output + offset, deviceDestData, destN*sizeof(uchar), cudaMemcpyDeviceToHost);
cudaFree(deviceDestData);

offset += destN;
}

cudaFree(deviceSrcData);
}*/

/*void cudaReverse(uchar* input, uchar* output, int srcRows, int srcCols){
uchar* deviceSrcData;
uchar* deviceDestData;
int threadsPerBlock = THREADS_PER_BLOCK;
int blocks = 0;
int chunkRows = 0;
int offset = 0;
int srcN = 3 * srcRows * srcCols;
cudaMalloc(&deviceSrcData, srcN*sizeof(uchar));
cudaMemcpy(deviceSrcData, input, srcN*sizeof(uchar), cudaMemcpyHostToDevice);

chunkRows = IMG_CHUNK / srcCols;
if (chunkRows == 0){
chunkRows = srcRows;
}
int rounds = ceil(srcRows / (double)chunkRows);
//printf("Rounds: %d \n", rounds);

for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * srcCols, 3 * srcRows * srcCols - (offset * 3));
if (destN <= 0){
//printf("Broken!\n");
break;
}

blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceDestData, destN*sizeof(uchar));

reverseKernel <<<blocks, threadsPerBlock >>>(deviceDestData, deviceSrcData, srcN, chunkRows, offset);
cudaMemcpy(output + (3 * offset), deviceDestData, destN*sizeof(uchar), cudaMemcpyDeviceToHost);
cudaFree(deviceDestData);

offset += destN / 3;
}


cudaFree(deviceSrcData);

}*/

/*void cudaGammaCorrection(uchar* input, uchar* output, double gamma, int srcRows, int srcCols){
uchar* deviceSrcData;
uchar* deviceDestData;
int threadsPerBlock = THREADS_PER_BLOCK;
int blocks = 0;
int chunkRows = 0;
int offset = 0;
int srcN = 3 * srcRows * srcCols;
cudaMalloc(&deviceSrcData, srcN*sizeof(uchar));
cudaMemcpy(deviceSrcData, input, srcN*sizeof(uchar), cudaMemcpyHostToDevice);

chunkRows = IMG_CHUNK / srcCols;
if (chunkRows == 0){
chunkRows = srcRows;
}
int rounds = ceil(srcRows / (double)chunkRows);

for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
if (destN <= 0){
break;
}

blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceDestData, destN*sizeof(uchar));

gammaCorrectionKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, srcRows, srcCols, gamma, chunkRows, offset);

cudaMemcpy(output + offset, deviceDestData, destN*sizeof(uchar), cudaMemcpyDeviceToHost);
cudaFree(deviceDestData);

offset += destN;
}

cudaFree(deviceSrcData);

}*/

/*void cudaDirectResize(uchar* input, uchar* output, int srcRows, int srcCols, int destRows, int destCols){
uchar* deviceSrcData;
uchar* deviceDestData;
int threadsPerBlock = THREADS_PER_BLOCK;
int blocks = 0;
int chunkRows = 0;
int offset = 0;
int srcN = 3 * srcRows * srcCols;
cudaMalloc(&deviceSrcData, srcN*sizeof(uchar));
cudaMemcpy(deviceSrcData, input, srcN*sizeof(uchar), cudaMemcpyHostToDevice);

chunkRows = IMG_CHUNK / destCols;
if (chunkRows == 0){
chunkRows = destRows;
}
int rounds = ceil(destRows / (double)chunkRows);

for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * destCols, 3 * destRows * destCols - offset);
if (destN <= 0){
break;
}

blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceDestData, destN*sizeof(uchar));

directResizeKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, srcRows, srcCols, destRows, destCols, chunkRows, offset);
cudaMemcpy(output + offset, deviceDestData, destN*sizeof(uchar), cudaMemcpyDeviceToHost);
cudaFree(deviceDestData);

offset += destN;
}



cudaFree(deviceSrcData);


}*/

/*void cudaLinearResize(uchar* input, uchar* output, int srcRows, int srcCols, int destRows, int destCols){
uchar* deviceSrcData;
uchar* deviceDestData;
int threadsPerBlock = THREADS_PER_BLOCK;
int blocks = 0;
int chunkRows = 0;
int offset = 0;
int srcN = 3 * srcRows * srcCols;
cudaMalloc(&deviceSrcData, srcN*sizeof(uchar));
cudaMemcpy(deviceSrcData, input, srcN*sizeof(uchar), cudaMemcpyHostToDevice);

chunkRows = IMG_CHUNK / destCols;
if (chunkRows == 0){
chunkRows = destRows;
}
int rounds = ceil(destRows / (double)chunkRows);

for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * destCols, 3 * destRows * destCols - offset);
if (destN <= 0){
break;
}

blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceDestData, destN*sizeof(uchar));

linearResizeKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, srcRows, srcCols, destRows, destCols, chunkRows, offset);

cudaMemcpy(output + offset, deviceDestData, destN*sizeof(uchar), cudaMemcpyDeviceToHost);
cudaFree(deviceDestData);

offset += destN;
}

cudaFree(deviceSrcData);
}*/

/*void cudaGaussianFilter(uchar* input, uchar* output, double gKernel[][2 * FILTER_SIZE + 1], int srcRows, int srcCols){
uchar* deviceSrcData;
uchar* deviceDestData;
double* deviceFilter;
int threadsPerBlock = THREADS_PER_BLOCK;
int blocks = 0;
int chunkRows = 0;
int offset = 0;
int srcN = 3 * srcRows * srcCols;
cudaMalloc(&deviceSrcData, srcN*sizeof(uchar));
cudaMalloc(&deviceFilter, (2 * FILTER_SIZE + 1) * (2 * FILTER_SIZE + 1) * sizeof(double));
cudaMemcpy(deviceSrcData, input, srcN*sizeof(uchar), cudaMemcpyHostToDevice);
cudaMemcpy(deviceFilter, gKernel, (2 * FILTER_SIZE + 1)*(2 * FILTER_SIZE + 1)*sizeof(double), cudaMemcpyHostToDevice);

chunkRows = IMG_CHUNK / srcCols;
if (chunkRows == 0){
chunkRows = srcRows;
}
int rounds = ceil(srcRows / (double)chunkRows);

for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
if (destN <= 0){
break;
}

blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceDestData, destN*sizeof(uchar));

gaussianFilterKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, deviceFilter, FILTER_SIZE, srcRows, srcCols, chunkRows, offset);

cudaMemcpy(output + offset, deviceDestData, destN*sizeof(uchar), cudaMemcpyDeviceToHost);
cudaFree(deviceDestData);

offset += destN;
}

cudaFree(deviceSrcData);
cudaFree(deviceFilter);
}*/

/*void cudaSobelFilter(uchar* input, uchar* output, int srcRows, int srcCols){
uchar* deviceSrcData;
uchar* deviceDestData;
short* deviceTempData;
int* deviceSobel_x;
int* deviceSobel_y;
int srcN = 3 * srcRows * srcCols;
double* deviceRangeMin;
double* deviceRangeMax;
double rangeMin[1] = { 0.0 };
double rangeMax[1] = { 0.0 };
int threadsPerBlock = THREADS_PER_BLOCK;
int blocks = 0;
int chunkRows = 0;
int offset = 0;

int sobel_x[9], sobel_y[9];

sobel_x[0] = -1; sobel_x[1] = 0; sobel_x[2] = 1;
sobel_x[3] = -2; sobel_x[4] = 0; sobel_x[5] = 2;
sobel_x[6] = -1; sobel_x[7] = 0; sobel_x[8] = 1;

sobel_y[0] = -1; sobel_y[1] = -2; sobel_y[2] = -1;
sobel_y[3] = 0; sobel_y[4] = 0; sobel_y[5] = 0;
sobel_y[6] = 1; sobel_y[7] = 2; sobel_y[8] = 1;

//int threadsPerBlock = 512;
//blocks = (srcN + threadsPerBlock - 1) / threadsPerBlock;
//printf("Blocks: %d\n", blocks);

cudaMalloc(&deviceSrcData, srcN*sizeof(uchar));
cudaMalloc(&deviceSobel_x, 9 * sizeof(int));
cudaMalloc(&deviceSobel_y, 9 * sizeof(int));
cudaMalloc(&deviceRangeMin, sizeof(double));
cudaMalloc(&deviceRangeMax, sizeof(double));

cudaMemcpy(deviceSrcData, input, srcN*sizeof(uchar), cudaMemcpyHostToDevice);
cudaMemcpy(deviceSobel_x, sobel_x, 9 * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(deviceSobel_y, sobel_y, 9 * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(deviceRangeMin, rangeMin, sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(deviceRangeMax, rangeMax, sizeof(double), cudaMemcpyHostToDevice);

chunkRows = IMG_CHUNK / srcCols;
if (chunkRows == 0){
chunkRows = srcRows;
}
int rounds = ceil(srcRows / (double)chunkRows);

short* temp_data = new short[3 * srcRows * srcCols];

for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
if (destN <= 0){
break;
}
blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceTempData, destN*sizeof(short));

sobelGradientKernel << <blocks, threadsPerBlock >> > (deviceTempData, deviceSrcData, deviceSobel_x, deviceSobel_y, deviceRangeMin, deviceRangeMax, srcRows, srcCols, offset);

cudaMemcpy(temp_data + offset, deviceTempData, destN*sizeof(short), cudaMemcpyDeviceToHost);
cudaFree(deviceTempData);

offset += destN;
}
cudaFree(deviceSrcData);
cudaFree(deviceSobel_x);
cudaFree(deviceSobel_y);
cudaFree(deviceRangeMin);
cudaFree(deviceRangeMax);
//printf("Works!\n");

//cudaMemcpy(rangeMin, deviceRangeMin, sizeof(double), cudaMemcpyDeviceToHost);
//cudaMemcpy(rangeMax, deviceRangeMax, sizeof(double), cudaMemcpyDeviceToHost);

//printf("Host temp data done");

for (int i = 0; i < srcRows; i++){
for (int j = 0; j < srcCols; j++){
for (int color = 0; color < 3; color++){
double value = temp_data[3 * (i * srcCols + j) + color];;
rangeMin[0] = std::min(value, rangeMin[0]);
rangeMax[0] = std::max(value, rangeMax[0]);
}
}
}


//printf("Got here!\n");
//output = (uchar*)temp_data;
//return;

//printf("Range Min: %f, Range Max: %f \n", rangeMin[0], rangeMax[0]);

//blocks = (srcN + threadsPerBlock - 1) / threadsPerBlock;
offset = 0;

for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
if (destN <= 0){
break;
}
blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceTempData, destN*sizeof(short));
cudaMemcpy(deviceTempData, temp_data + offset, destN*sizeof(short), cudaMemcpyHostToDevice);

cudaMalloc(&deviceDestData, destN*sizeof(uchar));
sobelRangeKernel << <blocks, threadsPerBlock >> >(deviceDestData, deviceTempData, rangeMin[0], rangeMax[0], 20, 60, offset);

cudaMemcpy(output + offset, deviceDestData, destN*sizeof(uchar), cudaMemcpyDeviceToHost);

cudaFree(deviceDestData);
cudaFree(deviceTempData);
offset += destN;
}

//uchar minThresh = 20;
//uchar maxThresh = 60;

/*for (int i = 0; i < srcRows; i++){
for (int j = 0; j < srcCols; j++){
for (int color = 0; color < 3; color++){
int idx = 3 * (i * srcCols + j) + color;

int value = 255 * ((temp_data[idx] + rangeMin[0])/(rangeMax[0] - rangeMin[0]));
if (value >= maxThresh){
value = 255;
}
else if (value < minThresh){
value = 0;
}
output[idx] = value;
}
}
}


//cudaFree(deviceTempData);
//cudaFree(deviceDestData);

delete[] temp_data;
}*/

/*void cudaKMeans(uchar* input, uchar* output, int srcRows, int srcCols, int k_means){
int threadsPerBlock = THREADS_PER_BLOCK;
int blocks = 0;
int chunkRows = 0;
int offset = 0;

uchar* deviceSrcData;
uchar* deviceDestData;
float* device_k_colors;
int* device_k_count;
//int* device_hits;
uchar* device_k_index;
bool* device_convergence;

float* k_colors = new float[k_means * 3];
uchar* k_index = new uchar[srcRows * srcCols];
int* k_count = new int[k_means];
//* hits = new int[k_means];

int srcN = srcRows * srcCols * 3;

for (int pix = 0; pix < k_means; pix++){
int i = rand() % srcRows;
int j = rand() % srcCols;
for (int color = 0; color < 3; color++){
k_colors[3 * pix + color] = input[3 * (i * srcCols + j) + color];
}
}
cudaMalloc(&device_k_colors, (3 * k_means)*sizeof(float));
cudaMemcpy(device_k_colors, k_colors, 3 * k_means *sizeof(float), cudaMemcpyHostToDevice);

//printf("=== START ===\n");
//for (int group = 0; group < k_means; group++){
//printf("Color Group %d: R=%f, G=%f, B=%f \n", group + 1, k_colors[3 * group + 2], k_colors[3 * group + 1], k_colors[3 * group]);
//}

bool convergence[1] = { false };

for (int k = 0; k < srcRows * srcCols; k++){
k_index[k] = 0;
}

chunkRows = (IMG_CHUNK * 0.5) / srcCols;
if (chunkRows == 0){
chunkRows = srcRows;
}

cudaMalloc(&device_k_count, (k_means)*sizeof(int));
//cudaMalloc(&device_hits, (k_means)*sizeof(int));
//cudaMalloc(&device_k_colors, (3 * k_means)*sizeof(float));
cudaMalloc(&device_convergence, sizeof(bool));

//cudaMalloc(&device_k_index, (srcRows * srcCols)*sizeof(uchar));
//cudaMemcpy(device_k_index, k_index, srcRows * srcCols *sizeof(uchar), cudaMemcpyHostToDevice);
int count = 0;

while (!convergence[0]){
convergence[0] = true;
cudaMemcpy(device_convergence, convergence, sizeof(bool), cudaMemcpyHostToDevice);
for (int k = 0; k < k_means; k++){
k_count[k] = 0;
//hits[k] = 0;
}
cudaMemcpy(device_k_count, k_count, k_means * sizeof(int), cudaMemcpyHostToDevice);
//cudaMemcpy(device_hits, hits, k_means * sizeof(int), cudaMemcpyHostToDevice);
//printf("Count: %d\n",count);

int rounds = ceil(srcRows / (float)chunkRows);

offset = 0;
for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
if (destN <= 0){
break;
}
blocks = ((destN/3) + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceSrcData, destN*sizeof(uchar));
cudaMemcpy(deviceSrcData, input + offset, destN*sizeof(uchar), cudaMemcpyHostToDevice);
cudaMalloc(&device_k_index, destN*sizeof(uchar)/3);
cudaMemcpy(device_k_index, k_index + (offset / 3), destN*sizeof(uchar) / 3, cudaMemcpyHostToDevice);

//kernel
//kMeansCountingKernel << <blocks, threadsPerBlock >> > (deviceSrcData, device_k_index, device_k_count, device_hits, device_k_colors, device_convergence, k_means, srcRows, srcCols,count);
kMeansCountingKernel << <blocks, threadsPerBlock >> > (deviceSrcData, device_k_index, device_k_count, device_k_colors, device_convergence, k_means, srcRows, srcCols);

cudaMemcpy(k_index + (offset / 3), device_k_index, destN*sizeof(uchar) / 3, cudaMemcpyDeviceToHost);

cudaFree(deviceSrcData);
cudaFree(device_k_index);
offset += destN;
}
cudaMemcpy(k_count, device_k_count, (k_means)*sizeof(int), cudaMemcpyDeviceToHost);
//cudaMemcpy(hits, device_hits, (k_means)*sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(convergence, device_convergence, sizeof(bool), cudaMemcpyDeviceToHost);

//printf("Group Count, step %d::\n",count);
//for (int i = 0; i < k_means; i++){
//	printf("Group %d: %d\n",i,k_count[i]);
//}

//convergence[0] = true;	//Stopper

//if (convergence[0])		//RE-ENABLE WHEN ACTUALLY USING
//break;

if (count == 200){
//printf("Stopped at %d!\n",count);
break;
}
count++;

for (int k = 0; k < 3 * k_means; k++){
k_colors[k] = 0;
}
cudaMemcpy(device_k_colors, k_colors, 3 * k_means *sizeof(float), cudaMemcpyHostToDevice);

rounds = ceil(srcRows / (float)chunkRows);

offset = 0;
for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
if (destN <= 0){
break;
}
blocks = ((destN / 3) + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceSrcData, destN*sizeof(uchar));
cudaMemcpy(deviceSrcData, input + offset, destN*sizeof(uchar), cudaMemcpyHostToDevice);
cudaMalloc(&device_k_index, destN*sizeof(uchar) / 3);
cudaMemcpy(device_k_index, k_index + (offset / 3), destN*sizeof(uchar) / 3, cudaMemcpyHostToDevice);

kMeansGroupAdjustKernel << <blocks, threadsPerBlock >> > (deviceSrcData, device_k_index, device_k_count, device_k_colors, k_means, srcRows, srcCols);

cudaFree(deviceSrcData);
cudaFree(device_k_index);

offset += destN;
}
cudaMemcpy(k_colors, device_k_colors, 3 * k_means * sizeof(float), cudaMemcpyDeviceToHost);
//kernel
}

int rounds = ceil(srcRows / (float)chunkRows);
offset = 0;
for (int step = 0; step < rounds; step++){
int destN = min(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
if (destN <= 0){
break;
}
blocks = ((destN / 3) + threadsPerBlock - 1) / threadsPerBlock;

cudaMalloc(&deviceDestData, destN*sizeof(uchar));
cudaMalloc(&device_k_index, destN*sizeof(uchar) / 3);
cudaMemcpy(device_k_index, k_index + (offset / 3), destN*sizeof(uchar) / 3, cudaMemcpyHostToDevice);

//kernel
kMeansOutputKernel << <blocks, threadsPerBlock >> > (deviceDestData, device_k_index, device_k_colors, srcRows, srcCols);
cudaMemcpy(output + offset, deviceDestData, destN*sizeof(uchar), cudaMemcpyDeviceToHost);

cudaFree(deviceDestData);
cudaFree(device_k_index);

offset += destN;
}

//printf("Count: %d\n", count);

cudaFree(device_k_colors);
cudaFree(device_k_count);
cudaFree(device_convergence);

delete[] k_colors;
delete[] k_index;
delete[] k_count;

}*/

/*void cudaKMeansOld(uchar* input, uchar* output, int srcRows, int srcCols, int k_means){
int threadsPerBlock = 512;
int blocks = ((srcRows * srcCols) + threadsPerBlock - 1) / threadsPerBlock;
uchar* deviceSrcData;
uchar* deviceDestData;
float* device_k_colors;
int* device_k_count;
uchar* device_k_index;
bool* device_convergence;
//int srcN = 3 * srcRows * srcCols;
int srcN = min(3 * 1920 * 1080, 3 * srcRows * srcCols);
cudaMalloc(&deviceSrcData, srcN*sizeof(uchar));
cudaMalloc(&deviceDestData, srcN*sizeof(uchar));
cudaMalloc(&device_k_colors, (3 * k_means)*sizeof(float));
cudaMalloc(&device_k_index, (srcRows * srcCols)*sizeof(uchar));
cudaMalloc(&device_k_count, (k_means)*sizeof(int));
cudaMalloc(&device_convergence, sizeof(bool));

cudaMemcpy(deviceSrcData, input, srcN*sizeof(uchar), cudaMemcpyHostToDevice);

float* k_colors = new float[3 * k_means];
uchar* k_index = new uchar[srcRows * srcCols];
int* k_count = new int[k_means];

for (int pix = 0; pix < k_means; pix++){
int i = rand() % srcRows;
int j = rand() % srcCols;
for (int color = 0; color < 3; color++){
k_colors[3 * pix + color] = input[3 * (i * srcCols + j) + color];
}

}
cudaMemcpy(device_k_colors, k_colors, 3 * k_means *sizeof(float), cudaMemcpyHostToDevice);

printf("=== START ===\n");
for (int group = 0; group < k_means; group++){
printf("Color Group %d: R=%f, G=%f, B=%f \n", group + 1, k_colors[3 * group + 2], k_colors[3 * group + 1], k_colors[3 * group]);
}

bool convergence[1] = { false };

for (int k = 0; k < srcRows * srcCols; k++){
k_index[k] = 0;
}
cudaMemcpy(device_k_index, k_index, srcRows * srcCols *sizeof(uchar), cudaMemcpyHostToDevice);
int count = 0;

while (!convergence[0]){
convergence[0] = true;
cudaMemcpy(device_convergence, convergence, sizeof(bool), cudaMemcpyHostToDevice);
for (int k = 0; k < k_means; k++){
k_count[k] = 0;
}
cudaMemcpy(device_k_count, k_count, k_means * sizeof(int), cudaMemcpyHostToDevice);

kMeansCountingKernelOld<<<blocks,threadsPerBlock>>> (deviceSrcData,device_k_index,device_k_count,device_k_colors, device_convergence, k_means,srcRows,srcCols);
cudaMemcpy(k_index, device_k_index, (srcRows*srcCols)*sizeof(uchar), cudaMemcpyDeviceToHost);
cudaMemcpy(k_count, device_k_count, (k_means)*sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(convergence, device_convergence, sizeof(bool), cudaMemcpyDeviceToHost);

if (count == 400){
printf("Stopped at 400!\n");
break;
}
count++;
//printf("Bogey::\n");
if (convergence[0])
break;
for (int k = 0; k < 3 * k_means; k++){
k_colors[k] = 0;
}
cudaMemcpy(device_k_colors, k_colors, 3 * k_means *sizeof(float), cudaMemcpyHostToDevice);

//kMeansGroupAdjustKernel<<<blocks,threadsPerBlock>>> (deviceSrcData,device_k_index,device_k_count,device_k_colors,k_means,srcRows,srcCols);
cudaMemcpy(k_colors, device_k_colors, 3 * k_means * sizeof(float), cudaMemcpyDeviceToHost);

}
cudaMemcpy(device_k_colors, k_colors, 3 * k_means *sizeof(float), cudaMemcpyHostToDevice);


printf("=== END ===\n");
//for (int group = 0; group < k_means; group++){
//printf("Color Group %d: R=%f, G=%f, B=%f \n", group + 1, k_colors[3 * group + 2], k_colors[3 * group + 1], k_colors[3 * group]);
//}
kMeansOutputKernel<<<blocks,threadsPerBlock>>> (deviceDestData,device_k_index,device_k_colors,srcRows,srcCols);
cudaMemcpy(output, deviceDestData, srcN*sizeof(uchar), cudaMemcpyDeviceToHost);

cudaFree(deviceSrcData);
cudaFree(deviceDestData);
cudaFree(device_k_colors);
cudaFree(device_k_index);
cudaFree(device_k_count);
cudaFree(device_convergence);

delete[] k_colors;
delete[] k_index;
delete[] k_count;

}*/

//-- OpenCV Handling

/*Mat rgb2Gray(Mat image){
Mat out = Mat(image.rows, image.cols, CV_8UC1);

uchar* input = (uchar*)image.datastart;
uchar* output = (uchar*)out.datastart;
cudaRgb2Gray(input, output, image.rows, image.cols);

return out;
}*/

/*Mat reverse(Mat image){
Mat out = Mat(image.rows, image.cols, image.type());

uchar* input = (uchar*)image.datastart;
uchar* output = (uchar*)out.datastart;
cudaReverse(input, output, image.rows, image.cols);

return out;
}*/

/*Mat gammaCorrection(Mat image, double gamma){
Mat out = Mat(image.rows, image.cols, image.type());

uchar* input = (uchar*)image.datastart;
uchar* output = (uchar*)out.datastart;
cudaGammaCorrection(input, output, gamma, image.rows, image.cols);

return out;
}*/

/*Mat directResize(Mat image, int rows, int cols){
Mat out = Mat(rows, cols, image.type());

uchar* input = (uchar*)image.datastart;
uchar* output = (uchar*)out.datastart;
cudaDirectResize(input, output, image.rows, image.cols, rows, cols);

return out;
}*/

/*Mat linearResize(Mat image, int rows, int cols){
Mat out = Mat(rows, cols, image.type());

uchar* input = (uchar*)image.datastart;
uchar* output = (uchar*)out.datastart;
cudaLinearResize(input, output, image.rows, image.cols, rows, cols);

return out;
}*/

/*void createFilter(double gKernel[][2 * FILTER_SIZE + 1], double inputSigma){
//standard deviation to 1.0
double sigma = inputSigma;
double r, s = 2.0 * sigma * sigma;
double sum = 0.0;

for (int x = -FILTER_SIZE; x <= FILTER_SIZE; x++){
for (int y = -FILTER_SIZE; y <= FILTER_SIZE; y++){
r = sqrt(x*x + y*y);
gKernel[x + FILTER_SIZE][y + FILTER_SIZE] = exp(-(r*r) / s) / (M_PI * s);
sum += gKernel[x + FILTER_SIZE][y + FILTER_SIZE];
}
}

for (int i = 0; i < 2 * FILTER_SIZE + 1; ++i){
for (int j = 0; j < 2 * FILTER_SIZE + 1; ++j){
gKernel[i][j] /= sum;
}
}
}*/

/*Mat gaussianFilter(Mat image, double sigma){
double gKernel[2 * FILTER_SIZE + 1][2 * FILTER_SIZE + 1];
createFilter(gKernel, sigma);

Mat out = Mat(image.rows, image.cols, image.type());
uchar* input = (uchar*)image.datastart;
uchar* output = (uchar*)out.datastart;
cudaGaussianFilter(input,output,gKernel,image.rows,image.cols);

//delete[] gKernel;

return out;
//return image;
}*/

/*Mat sobelFilter(Mat image){
Mat out(image.rows, image.cols, image.type());
//Mat temp(image.rows, image.cols, image.type());
uchar* input = (uchar*)image.datastart;
uchar* output = (uchar*)out.datastart;
cudaSobelFilter(input,output,image.rows,image.cols);

return out;
}*/

/*Mat kMeans(Mat image, int k_means){
srand(6000);
if (k_means > 256){
printf("Error: Max number of groups exceeded (256)\n");
exit(-1);
}
Mat out(image.rows, image.cols, image.type());
uchar* input = (uchar*)image.datastart;
uchar* output = (uchar*)out.datastart;
cudaKMeans(input,output,image.rows,image.cols,k_means);

return out;
}*/

/*Mat gaussianPyramid(cv::Mat image, uchar levels, float scale){
if (scale > 0.5){
printf("Error: Scale > 0.5\n");
exit(-1);
}

int srcRows = image.rows;
int srcCols = image.cols;
Mat output(srcRows + (srcRows * scale + 1), srcCols, image.type());
uchar* src_data = (uchar*)image.datastart;
uchar* dest_data = (uchar*)output.datastart;

for (int i = 0; i < srcRows; i++){
for (int j = 0; j < srcCols; j++){
for (int color = 0; color < 3; color++){
int idx = 3 * (i * srcCols + j) + color;
dest_data[idx] = src_data[idx];
}
}
}
dest_data += srcRows * srcCols * 3;
int newRows = srcRows * scale;
int newCols = srcCols * scale;
float newScale = scale;
int offset = 0;
for (int level = 1; level < levels; level++){
image = gaussianFilter(image, 1.0);
image = linearResize(image, newRows, newCols);
src_data = (uchar*)image.datastart;

for (int i = 0; i < newRows; i++){
for (int j = 0; j < newCols; j++){
for (int color = 0; color < 3; color++){
int idx = 3 * (i * srcCols + j + offset) + color;
dest_data[idx] = src_data[3 * (i * newCols + j) + color];
}
}
}
offset += newCols;

newRows *= scale;
newCols *= scale;
newScale *= scale;

}

return output;
}*/

//-- Main

void simulate(cv::Mat& im){
	//im = frgb2Gray(im);
	img_proc::mySift(im);
	mySift(im);
}

int main(void){		// [MAIN]

	//printf("%d\n", (unsigned int)powf(2, 2) & 4);
	//img_proc::makeFilter(1.7148);
	//printf("Done!\n");
	//getchar();
	//return;


	cv::Mat im, im0, im1;
	im = imread("D://School//Summer 2016//Research//nausicaa.jpg");
	im = img_proc::linearResize(im, 640, 640);
	//im = img_proc::frgb2Gray(im);
	//im = imread("D://School//Summer 2016//Research//Stereo//storage//z_im0.png");


	//im0 = imread("D://School//Summer 2016//Research//kMeans//nausicaa_float_cpu.png");
	//im0 = img_proc::frgb2Gray(im0);
	//im1 = imread("D://School//Summer 2016//Research//kMeans//nausicaa_float_gpu.png");
	//im1 = img_proc::frgb2Gray(im1);


	//im = imread("D://School//Summer 2016//Research//bike.jpg");
	//im = imread("D://School//Summer 2016//Research//gray//einstein_gray.png");
	//im0 = imread("D://School//Summer 2016//Research//Stereo//storage//ims0.png");
	//im1 = imread("D://School//Summer 2016//Research//Stereo//storage//ims1.png");

	if (im.empty()){
		printf("Error!\n");
		getchar();
		return -1;
	}

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	int versionFish = 0;

	//img_proc::diff_count(im0, im1);

	//im = img_proc::frgb2Gray(im);
	//im = img_proc::fGaussianFilterSep(im, 1.6);
	//im = img_proc::mySift(im);
	//im = fGaussianFilterSep(im, 1.6);
	//im = reverse(im);
	//im = img_proc::mySift(im);
	//im = myConv2(im, filter, 0);
	//im1 = kMeansFixed(im, 4);
	//im = img_proc::linearResize(im, im.rows / 4, im.cols / 4);

	//im0 = imread("D://School//Summer 2016//Research//mySift//audrey_conv_cpu.png");
	//im1 = imread("D://School//Summer 2016//Research//mySift//audrey_conv_gpu.png");
	//im0 = img_proc::frgb2Gray(im0);
	//im1 = img_proc::frgb2Gray(im1);
	//im = img_proc::diff_count(im0, im1);
	

	//imwrite("D://School//Summer 2016//Research//mySift//reverse.png", im, compression_params);
	//imwrite("D://School//Summer 2016//Research//Stereo//storage//z_im1.png", im, compression_params);

	//im0 = img_proc::frgb2Gray(im0);
	//im1 = img_proc::frgb2Gray(im1);

	//img_proc::diff_count(im0, im1);


	//im = img_proc::depthFromStereo(im0, im1, 0, 0, 0);
	//im = img_proc::kMeans(im, 8);
	//im = kMeans(im, 8);
	//im = img_proc::kMeansFixed(im, 8);

	//imwrite("D://School//Summer 2016//Research//kMeans//nausicaa_float_gpu.png", im, compression_params);

	//cudaRuntimeGetVersion(&versionFish);

	//printf("Version: %f\n", CV_VERSION);
	//cout << CV_VERSION << endl;

	//Vec3b intensity = im.at<Vec3b>(20, 20);

	//printf("rows: %d, columns: %d, %d \n", im.rows, im.cols,intensity.val[2]);

	//get_nvtAttrib("Bike", 0xFF222222);
	//img_proc::mySift(im);
	//mySift(im);
	//nvtxRangePop();
	//Mat resized = gaussianFilter(im,2.0);
	//Mat resized = img_proc::kMeans(im,4);
	//Mat temp = frgb2Gray(im);
	//Mat resized2 =  fdirectResize(temp, temp.rows / 4, temp.cols / 4);
	//Mat resized = img_proc::fdirectResize(temp, temp.rows / 4, temp.cols / 4);
	//auto time1 = std::chrono::high_resolution_clock::now();
	//Mat resized = img_proc::mySift(im);
	//nvtxEventAttributes_t eventAttrib = get_nvtAttrib("Fish", 0xFF880000);
	//nvtxMarkEx(&eventAttrib);
	//eventAttrib.message.ascii = "Fish";
	//eventAttrib.color = 0xFF880000;
	//nvtxRangePushEx(&eventAttrib);
	

	//Mat resized, resized2;

	//cudaLagSetup();

	//im = imread("D://School//Summer 2016//Research//audrey.jpg");
	//get_nvtAttrib("Audrey", 0xFF222222);
	//simulate(im);
	//nvtxRangePop();

	/*im = imread("D://School//Summer 2016//Research//einstein.png");
	get_nvtAttrib("Einstein", 0xFF222222);
	//im = frgb2Gray(im);
	//img_proc::fgaussianFilter(im, 2.0);
	//fGaussianFilter(im, 2.0);
	simulate(im);
	nvtxRangePop();

	im = imread("D://School//Summer 2016//Research//boat.png");
	get_nvtAttrib("Boat", 0xFF222222);
	//im = frgb2Gray(im);
	//img_proc::fgaussianFilter(im, 2.0);
	//fGaussianFilter(im, 2.0);
	simulate(im);
	nvtxRangePop();

	im = imread("D://School//Summer 2016//Research//bike.jpg");
	get_nvtAttrib("Bike", 0xFF222222);
	//im = frgb2Gray(im);
	//img_proc::fgaussianFilter(im, 2.0);
	//fGaussianFilter(im, 2.0);
	simulate(im);
	nvtxRangePop();

	im = imread("D://School//Summer 2016//Research//castle.png");
	get_nvtAttrib("Castle", 0xFF222222);
	//im = frgb2Gray(im);
	//img_proc::fgaussianFilter(im, 2.0);
	//fGaussianFilter(im, 2.0);
	simulate(im);
	nvtxRangePop();

	im = imread("D://School//Summer 2016//Research//lena.png");
	get_nvtAttrib("Lena", 0xFF222222);
	//im = frgb2Gray(im);
	//img_proc::fgaussianFilter(im, 2.0);
	//fGaussianFilter(im, 2.0);
	simulate(im);
	nvtxRangePop();

	im = imread("D://School//Summer 2016//Research//valve.png");
	get_nvtAttrib("Valve", 0xFF222222);
	//im = frgb2Gray(im);
	//img_proc::fgaussianFilter(im, 2.0);
	//fGaussianFilter(im, 2.0);
	simulate(im);
	nvtxRangePop();

	im = imread("D://School//Summer 2016//Research//koala.png");
	get_nvtAttrib("Koala", 0xFF222222);
	//im = frgb2Gray(im);
	//img_proc::fgaussianFilter(im, 2.0);
	//fGaussianFilter(im, 2.0);
	simulate(im);
	nvtxRangePop();

	im = imread("D://School//Summer 2016//Research//audrey.jpg");
	get_nvtAttrib("Audrey", 0xFF222222);
	//im = frgb2Gray(im);
	//img_proc::fgaussianFilter(im, 2.0);
	//fGaussianFilter(im, 2.0);
	simulate(im);
	nvtxRangePop();

	im = imread("D://School//Summer 2016//Research//nausicaa.jpg");
	get_nvtAttrib("Nausicaa", 0xFF222222);
	//im = frgb2Gray(im);
	//img_proc::fgaussianFilter(im, 2.0);
	//fGaussianFilter(im, 2.0);
	simulate(im);
	nvtxRangePop();*/

	//********//

	//printf("Done!\n");
	//getchar();
	//return 0;

	//********//


	ofstream avg_file, det_file, allc_file, allg_file, spd_file, thc_file, thg_file;
	uchar micro = 181;
	string name = "";
	double filesize = 0;

	allc_file.open("D://School//Summer 2016//Research//Textfiles//averages__all_cpu.csv");
	allc_file << micro << "s, rgb2Gray,reverse,gamma correction,direct resize(x2),direct resize(x0.5),linear resize(x2),linear resize(x0.5),gaussian filter,edge detection,k-means,gaussian pyramid" << endl;
	allg_file.open("D://School//Summer 2016//Research//Textfiles//averages__all_gpu.csv");
	allg_file << micro << "s, rgb2Gray,reverse,gamma correction,direct resize(x2),direct resize(x0.5),linear resize(x2),linear resize(x0.5),gaussian filter,edge detection,k-means,gaussian pyramid" << endl;

	spd_file.open("D://School//Summer 2016//Research//Textfiles//averages__speedups.csv");
	spd_file << "CPU/GPU, rgb2Gray,reverse,gamma correction,direct resize(x2),direct resize(x0.5),linear resize(x2),linear resize(x0.5),gaussian filter,edge detection,k-means,gaussian pyramid" << endl;

	thc_file.open("D://School//Summer 2016//Research//Textfiles//averages__throughput_cpu.csv");
	thc_file << "MB/s, rgb2Gray,reverse,gamma correction,direct resize(x2),direct resize(x0.5),linear resize(x2),linear resize(x0.5),gaussian filter,edge detection,k-means,gaussian pyramid" << endl;
	thg_file.open("D://School//Summer 2016//Research//Textfiles//averages__throughput_gpu.csv");
	thg_file << "MB/s, rgb2Gray,reverse,gamma correction,direct resize(x2),direct resize(x0.5),linear resize(x2),linear resize(x0.5),gaussian filter,edge detection,k-means,gaussian pyramid" << endl;

	//printf("Reading started... ");

	for (int pics = 0; pics <= 8; pics++){
		Mat im1;
		if (pics == 0){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_einstein.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_einstein.txt");
			im1 = imread("D://School//Summer 2016//Research//einstein.png");
			name = "einstein";
		}
		else if (pics == 1){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_castle.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_castle.txt");
			im1 = imread("D://School//Summer 2016//Research//castle.png");
			name = "castle";
		}
		else if (pics == 2){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_lena.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_lena.txt");
			im1 = imread("D://School//Summer 2016//Research//lena.png");
			name = "lena";
		}
		else if (pics == 3){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_boat.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_boat.txt");
			im1 = imread("D://School//Summer 2016//Research//boat.png");
			name = "boat";
		}
		else if (pics == 4){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_bike.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_bike.txt");
			im1 = imread("D://School//Summer 2016//Research//bike.jpg");
			name = "bike";
		}
		else if (pics == 5){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_valve.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_valve.txt");
			im1 = imread("D://School//Summer 2016//Research//valve.png");
			name = "valve";
		}
		else if (pics == 6){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_koala.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_koala.txt");
			im1 = imread("D://School//Summer 2016//Research//koala.png");
			name = "koala";//[STOP HERE]
		}
		else if (pics == 7){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_nausicaa.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_nausicaa.txt");
			im1 = imread("D://School//Summer 2016//Research//nausicaa.jpg");
			name = "nausicaa";
		}
		else if (pics == 8){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_audrey.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_audrey.txt");
			im1 = imread("D://School//Summer 2016//Research//audrey.jpg");
			name = "oranges";
		}
		else if (pics == 9){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_oranges.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_oranges.txt");
			im1 = imread("D://School//Summer 2016//Research//oranges.jpg");
			name = "oranges";
		}
		else if (pics == 10){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_mountains.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_mountains.txt");
			im1 = imread("D://School//Summer 2016//Research//mountains.jpg");
			name = "mountains";
		}
		else if (pics == 11){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_tiger.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_tiger.txt");
			im1 = imread("D://School//Summer 2016//Research//tiger.jpg");
			name = "tiger";
		}
		else{
			printf("Error!\n");
			exit(-1);
		}

		filesize = im1.rows * im1.cols * 3;
		det_file << name << ": " << im1.cols << "x" << im1.rows << endl << "========" << endl;
		avg_file << name << ": " << im1.cols << "x" << im1.rows << endl << "========" << endl;
		allc_file << name;
		allg_file << name;
		spd_file << name;
		thc_file << name;
		thg_file << name;
		//printf("Reading done\n");

		//RGB 2 Gray

		int rounds = 10;

		printf("RGB2Gray\n");
		double cpu_duration = 0, gpu_duration = 0;
		det_file << "RGB 2 GRAY" << endl;
		avg_file << "RGB 2 GRAY" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::rgb2Gray(im1);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			rgb2Gray(im1);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		//avg_file.close();
		//det_file.close();
		//continue;

		//Reverse

		printf("Reverse\n");
		cpu_duration = 0; gpu_duration = 0;
		//Mat im1 = imread("D://School//Summer 2016//Research//valve.png");
		det_file << "REVERSE" << endl;
		avg_file << "REVERSE" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::reverse(im1);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			reverse(im1);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		//Gamma Correction

		printf("Gamma Correction\n");
		cpu_duration = 0; gpu_duration = 0;
		//Mat im1 = imread("D://School//Summer 2016//Research//valve.png");
		det_file << "GAMMA CORRECTION" << endl;
		avg_file << "GAMMA CORRECTION" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::gammaCorrection(im1, 2.0);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			gammaCorrection(im1, 2.0);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		//Direct Resize

		printf("Direct Resize x2\n");
		cpu_duration = 0; gpu_duration = 0;
		//Mat im1 = imread("D://School//Summer 2016//Research//valve.png");
		det_file << "DIRECT RESIZE (X2)" << endl;
		avg_file << "DIRECT RESIZE (X2)" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::directResize(im1, im1.rows * 2, im1.cols * 2);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			directResize(im1, im1.rows * 2, im1.cols * 2);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		printf("Direct Resize x0.5\n");
		cpu_duration = 0; gpu_duration = 0;
		//Mat im1 = imread("D://School//Summer 2016//Research//valve.png");
		det_file << "DIRECT RESIZE (X0.5)" << endl;
		avg_file << "DIRECT RESIZE (X0.5)" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::directResize(im1, im1.rows * 0.5, im1.cols * 0.5);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			directResize(im1, im1.rows * 0.5, im1.cols * 0.5);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		//Linear Resize

		printf("Linear Resize x2\n");
		cpu_duration = 0; gpu_duration = 0;
		//Mat im1 = imread("D://School//Summer 2016//Research//valve.png");
		det_file << "LINEAR RESIZE (X2)" << endl;
		avg_file << "LINEAR RESIZE (X2)" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::linearResize(im1, im1.rows * 2, im1.cols * 2);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			linearResize(im1, im1.rows * 2, im1.cols * 2);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		cpu_duration = 0; gpu_duration = 0;
		//Mat im1 = imread("D://School//Summer 2016//Research//valve.png");
		printf("Linear Resize x0.5\n");
		det_file << "LINEAR RESIZE (X0.5)" << endl;
		avg_file << "LINEAR RESIZE (X0.5)" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::linearResize(im1, im1.rows * 0.5, im1.cols * 0.5);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			linearResize(im1, im1.rows * 0.5, im1.cols * 0.5);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		//Gaussian Filter

		rounds = 5;
		cpu_duration = 0; gpu_duration = 0;
		printf("Gaussian Filter Sep\n");
		Mat img_temp = frgb2Gray(im1);
		//Mat im1 = imread("D://School//Summer 2016//Research//valve.png");
		det_file << "GAUSSIAN FILTER SEP (SIZE 3, SIGMA 1.0)" << endl;
		avg_file << "GAUSSIAN FILTER SEP (SIZE 3, SIGMA 1.0)" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			//img_proc::gaussianFilter(im1, 1.0);
			img_proc::fGaussianFilterSep(img_temp, 1.0);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			fGaussianFilterSep(img_temp, 1.0);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		//Edge Filter

		rounds = 1;
		cpu_duration = 0; gpu_duration = 0;
		//Mat im2 = imread("D://School//Summer 2016//Research//valve_gray.png");
		printf("Edge Filter\n");
		det_file << "EDGE DETECTION" << endl;
		avg_file << "EDGE DETECTION" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::sobelFilter(im1);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			sobelFilter(im1);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		//k-Means 8, 200 rounds

		rounds = 1;
		cpu_duration = 0; gpu_duration = 0;
		//Mat im2 = imread("D://School//Summer 2016//Research//valve_gray.png");
		det_file << "K MEANS 8" << endl;
		avg_file << "K MEANS 8" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::kMeansFixed(im1, 8);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			kMeansFixed(im1, 8);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);

		//Gaussian Pyramid

		rounds = 1;
		cpu_duration = 0; gpu_duration = 0;
		//Mat im2 = imread("D://School//Summer 2016//Research//valve_gray.png");
		det_file << "GAUSSIAN PYRAMID 8 0.5x" << endl;
		avg_file << "GAUSSIAN PYRAMID 8 0.5x" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::gaussianPyramid(im1, 8, 0.5);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			gaussianPyramid(im1, 8, 0.5);;
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);


		//SIFT

		rounds = 1;
		cpu_duration = 0; gpu_duration = 0;
		printf("SIFT\n");
		//Mat im2 = imread("D://School//Summer 2016//Research//valve_gray.png");
		int lesser = min(img_temp.rows, img_temp.cols);
		Mat img_sqtmp = fdirectResize(img_temp,lesser,lesser);
		det_file << "SIFT" << endl;
		avg_file << "SIFT" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::mySift(img_sqtmp);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			mySift(img_sqtmp);
			auto t4 = std::chrono::high_resolution_clock::now();
			gpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t4 - t3).count();
			det_file << "GPU Run: " << fixed << gpu_time << micro << "s" << endl;
			gpu_duration += gpu_time;
		}
		det_file << "========" << endl;
		avg_file << "CPU Average: " << fixed << cpu_duration / rounds << micro << "s" << endl;
		avg_file << "GPU Average: " << fixed << gpu_duration / rounds << micro << "s" << endl;
		avg_file << "CPU:GPU:  " << fixed << cpu_duration / gpu_duration << endl;
		avg_file << "========" << endl;
		allc_file << "," << fixed << cpu_duration / rounds;
		allg_file << "," << fixed << gpu_duration / rounds;
		spd_file << "," << fixed << cpu_duration / gpu_duration;
		thc_file << "," << fixed << filesize / (cpu_duration / rounds);
		thg_file << "," << fixed << filesize / (gpu_duration / rounds);


		avg_file.close();
		det_file.close();

		allc_file << endl;
		allg_file << endl;
		spd_file << endl;
		thc_file << endl;
		thg_file << endl;
	}

	allc_file.close();
	allg_file.close();
	spd_file.close();
	thc_file.close();
	thg_file.close();
	return 0;
}