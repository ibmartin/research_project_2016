#ifndef _GPU_PROC_HOST_CU_
#define _GPU_PROC_HOST_CU_

#include <atomic>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GpuProcKernel.cu"

#define FILTER_SIZE			3
#define M_PI				3.14159265358979323846  /* pi */
#define IMG_CHUNK			3110400	/* (1920 x 1080 x 3) / 2 */
#define THREADS_PER_BLOCK	256
//#define MEM_CAP				131072	// (2 ^ 17)
#define MEM_CAP				65536 //64 KB as a power of 2 (2 ^ 16)
//#define MEM_CAP				32768 //32 KB as a power of 2 (2 ^ 15)
//#define MEM_CAP				16384 //16 KB as a power of 2 (2 ^ 14)

#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}

int FIXED = 16;
int ONE = 1 << FIXED;

void cudaLagSetup(){
	get_nvtAttrib("Cuda Setup", 0xFFFFFFFF);
	unsigned char* nothing;
	cudaMalloc(&nothing, sizeof(unsigned char));
	cudaFree(nothing);
	nvtxRangePop();
}

void cudaRgb2Gray(unsigned char* input, unsigned char* output, int srcRows, int srcCols){
	get_nvtAttrib("rgb2Gray CPU", 0xFF222222);
	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;
	int srcN = 3 * srcRows * srcCols;
	cudaMalloc(&deviceSrcData, srcN*sizeof(unsigned char));
	cudaMemcpy(deviceSrcData, input, srcN*sizeof(unsigned char), cudaMemcpyHostToDevice);

	chunkRows = IMG_CHUNK / srcCols;
	if (chunkRows == 0){
		chunkRows = srcRows;
	}
	int rounds = ceil(srcRows / (double)chunkRows);

	//int destN = min(1920 * 1080, srcRows * srcCols);
	for (int step = 0; step < rounds; step++){
		int destN = fmin(chunkRows * srcCols, srcRows * srcCols - offset);
		if (destN <= 0)
			break;

		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));
		rgb2GrayKernel <<<blocks, threadsPerBlock >>> (deviceDestData, deviceSrcData, srcRows, srcCols, chunkRows, offset);
		cudaMemcpy(output + offset, deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += destN;
	}

	cudaFree(deviceSrcData);
	nvtxRangePop();
}

void cudafRgb2Gray(unsigned char* input, float* output, int srcRows, int srcCols){
	get_nvtAttrib("Setup", 0xFF0000FF);
	unsigned char* deviceSrcData;
	float* deviceDestData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;
	//int srcN = 3 * srcRows * srcCols;
	int datasize1 = 3 * sizeof(unsigned char);
	int datasize2 = sizeof(float);

	int chunk_size = MEM_CAP / (datasize1 + datasize2);
	int rounds = ceil((srcRows * srcCols) / (float)chunk_size);
	//cudaMalloc(&deviceSrcData, srcN*sizeof(float));
	//cudaMemcpy(deviceSrcData, input, srcN*sizeof(float), cudaMemcpyHostToDevice);

	chunkRows = IMG_CHUNK / srcCols;
	if (chunkRows == 0){
		chunkRows = srcRows;
	}
	//int rounds = ceil(srcRows / (double)chunkRows);
	//int rounds = steps;
	nvtxRangePop();//Setup

	get_nvtAttrib("K Blocks " + std::to_string(rounds), 0xFF008800);
	for (int step = 0; step < rounds; step++){
		get_nvtAttrib("Pre-K", 0xFFFF0000);
		get_nvtAttrib("Vars", 0xFF880000);
		int remainder = fmin(chunk_size, (srcRows * srcCols) - (step * chunk_size) );
		int srcN = remainder * datasize1;
		int destN = remainder * datasize2;
		if (destN <= 0)
			break;

		blocks = ((destN / datasize2) + threadsPerBlock - 1) / threadsPerBlock;
		nvtxRangePop();

		get_nvtAttrib("Malloc", 0xFF880000);
		cudaMalloc(&deviceSrcData, srcN);
		cudaMalloc(&deviceDestData, destN);
		nvtxRangePop();

		get_nvtAttrib("Memcpy", 0xFF880000);
		cudaMemcpy(deviceSrcData, input + (step * chunk_size * 3), srcN, cudaMemcpyHostToDevice);
		nvtxRangePop();
		
		nvtxRangePop();

		get_nvtAttrib("Kern", 0xFF00FF00);
		frgb2GrayKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, srcRows, srcCols, chunkRows, 0);
		cudaDeviceSynchronize();
		nvtxRangePop();

		get_nvtAttrib("Post-K", 0xFF0000FF);
		cudaMemcpy(output + offset, deviceDestData, destN, cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += remainder;
		nvtxRangePop();
	}
	nvtxRangePop();
}

void cudaReverse(unsigned char* input, unsigned char* output, int srcRows, int srcCols){
	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;
	int srcN = 3 * srcRows * srcCols;
	cudaMalloc(&deviceSrcData, srcN*sizeof(unsigned char));
	cudaMemcpy(deviceSrcData, input, srcN*sizeof(unsigned char), cudaMemcpyHostToDevice);

	chunkRows = IMG_CHUNK / srcCols;
	if (chunkRows == 0){
		chunkRows = srcRows;
	}
	int rounds = ceil(srcRows / (double)chunkRows);
	//printf("Rounds: %d \n", rounds);

	for (int step = 0; step < rounds; step++){
		int destN = fmin(chunkRows * srcCols, srcRows * srcCols - (offset));
		if (destN <= 0){
			//printf("Broken!\n");
			break;
		}

		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN * 3 * sizeof(unsigned char));

		reverseKernel <<<blocks, threadsPerBlock >>>(deviceDestData, deviceSrcData, srcN, chunkRows, offset);
		cudaMemcpy(output + (3 * offset), deviceDestData, destN * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += destN;
	}


	cudaFree(deviceSrcData);

}

void cudaGammaCorrection(unsigned char* input, unsigned char* output, double gamma, int srcRows, int srcCols){
	get_nvtAttrib("Setup Inner", 0xFF000088);
	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;
	int srcN = 3 * srcRows * srcCols;
	cudaMalloc(&deviceSrcData, srcN*sizeof(unsigned char));
	cudaMemcpy(deviceSrcData, input, srcN*sizeof(unsigned char), cudaMemcpyHostToDevice);

	chunkRows = IMG_CHUNK / srcCols;
	if (chunkRows == 0){
		chunkRows = srcRows;
	}
	int rounds = ceil(srcRows / (double)chunkRows);
	nvtxRangePop();

	get_nvtAttrib("Work Loop: " + std::to_string(srcRows * srcCols / rounds), 0xFF888888);
	for (int step = 0; step < rounds; step++){
		int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
		if (destN <= 0){
			break;
		}

		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));

		get_nvtAttrib("Kernel", 0xFF00FF00);
		gammaCorrectionKernel <<<blocks, threadsPerBlock >>> (deviceDestData, deviceSrcData, srcRows, srcCols, gamma, chunkRows, offset);
		nvtxRangePop();

		cudaMemcpy(output + offset, deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += destN;
	}
	nvtxRangePop();

	cudaFree(deviceSrcData);

}

void cudaDirectResize(unsigned char* input, unsigned char* output, int srcRows, int srcCols, int destRows, int destCols){
	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;
	int srcN = 3 * srcRows * srcCols;
	cudaMalloc(&deviceSrcData, srcN*sizeof(unsigned char));
	cudaMemcpy(deviceSrcData, input, srcN*sizeof(unsigned char), cudaMemcpyHostToDevice);

	chunkRows = IMG_CHUNK / destCols;
	if (chunkRows == 0){
		chunkRows = destRows;
	}
	int rounds = ceil(destRows / (double)chunkRows);

	for (int step = 0; step < rounds; step++){
		int destN = fmin(3 * chunkRows * destCols, 3 * destRows * destCols - offset);
		if (destN <= 0){
			break;
		}

		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));

		directResizeKernel <<<blocks, threadsPerBlock >>> (deviceDestData, deviceSrcData, srcRows, srcCols, destRows, destCols, chunkRows, offset);
		cudaMemcpy(output + offset, deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += destN;
	}
	cudaFree(deviceSrcData);
}

void cudafDirectResize(float* input, float* output, int srcRows, int srcCols, int destRows, int destCols){	//
	float* deviceSrcData;
	float* deviceDestData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;

	int datasize = sizeof(float);

	float ratio = (float)(destRows * destCols) / (float)(srcRows * srcCols);
	ratio = ((1 / ratio) + 1) * datasize;

	int remainder = destRows * destCols;
	float rRow = (float)srcRows / destRows;
	float rCol = (float)srcCols / destCols;

	int pixels = ceil(MEM_CAP / ratio);
	int sentinel = 1;

	while (remainder > 0){

		//printf("Pixels: %d\n", pixels);

		int pix_begin = (destRows * destCols) - remainder;
		int pix_end = min(destRows * destCols - 1, pix_begin + pixels - 1);
		//printf("Begin: %d, End: %d\n", pix_begin, pix_end);

		int destN = (pix_end - pix_begin + 1) * datasize;
		blocks = ((destN / datasize) + threadsPerBlock - 1) / threadsPerBlock;

		int sRow = (pix_begin / destCols) * rRow;
		//int sCol = (pix_begin % destCols) * rCol;
		int sCol = 0;
		int src_begin = (sRow * srcCols + sCol);

		sRow = ((float)pix_end / destCols) * rRow;	// (3531 / 1600) * (480 / 1200) = 0.88275
		sCol = (pix_end % destCols) * rCol;			// (3531 % 1600) * (640 / 1600) = 132.4
		int src_end = min(srcRows * srcCols - 1, ((sRow + 1) * srcCols + sCol));

		//printf("srcRows: %d, srcCols: %d, total: %d\n", srcRows, srcCols, srcRows * srcCols);

		//printf("SBegin: %d, SEnd: %d\n", src_begin, src_end);
		/*if (pix_begin / destCols <= 11 && pix_end / destCols >= 11){
			printf("Pix Begin: %d, %d\n", pix_begin / destCols, pix_begin % destCols);
			printf("  sRow: %f, sCol: %f\n", (pix_begin / destCols) * rRow, (pix_begin % destCols) * rCol);
			printf("Pix End  : %d, %d\n", pix_end / destCols, pix_end % destCols);
			printf("Src Begin: %d, %d\n", src_begin / srcCols, src_begin % srcCols);
			printf("Src End  : %d, %d\n", src_end / srcCols, src_end % srcCols);
		}*/

		int srcN = (src_end - src_begin + 1) * datasize;

		cudaMalloc(&deviceSrcData, srcN);
		cudaMalloc(&deviceDestData, destN);
		//printf("Size: %d, src_begin: %d, src_end: %d\n", srcRows * srcCols, src_begin, src_end);
		cudaMemcpy(deviceSrcData, input + src_begin, srcN, cudaMemcpyHostToDevice);

		fdirectResizeKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, srcRows, srcCols, destRows, destCols, pix_begin, src_begin);

		cudaMemcpy(output + pix_begin, deviceDestData, destN, cudaMemcpyDeviceToHost);
		
		cudaFree(deviceDestData);
		cudaFree(deviceSrcData);

		remainder -= pix_end - pix_begin + 1;
		sentinel -= 1;

	}
	return;
}

void cudaLinearResize(unsigned char* input, unsigned char* output, int srcRows, int srcCols, int destRows, int destCols){
	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;
	int srcN = 3 * srcRows * srcCols;
	cudaMalloc(&deviceSrcData, srcN*sizeof(unsigned char));
	cudaMemcpy(deviceSrcData, input, srcN*sizeof(unsigned char), cudaMemcpyHostToDevice);

	chunkRows = IMG_CHUNK / destCols;
	if (chunkRows == 0){
		chunkRows = destRows;
	}
	int rounds = ceil(destRows / (double)chunkRows);

	for (int step = 0; step < rounds; step++){
		int destN = fmin(3 * chunkRows * destCols, 3 * destRows * destCols - offset);
		if (destN <= 0){
			break;
		}

		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));

		linearResizeKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, srcRows, srcCols, destRows, destCols, chunkRows, offset);

		cudaMemcpy(output + offset, deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += destN;
	}

	cudaFree(deviceSrcData);
}

void cudaGaussianFilter(unsigned char* input, unsigned char* output, double* gKernel, int srcRows, int srcCols){
	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
	double* deviceFilter;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;
	int srcN = 3 * srcRows * srcCols;
	cudaMalloc(&deviceSrcData, srcN*sizeof(unsigned char));
	cudaMalloc(&deviceFilter, (2 * FILTER_SIZE + 1) * (2 * FILTER_SIZE + 1) * sizeof(double));
	cudaMemcpy(deviceSrcData, input, srcN*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceFilter, gKernel, (2 * FILTER_SIZE + 1)*(2 * FILTER_SIZE + 1)*sizeof(double), cudaMemcpyHostToDevice);

	chunkRows = IMG_CHUNK / srcCols;
	if (chunkRows == 0){
		chunkRows = srcRows;
	}
	int rounds = ceil(srcRows / (double)chunkRows);

	for (int step = 0; step < rounds; step++){
		int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
		if (destN <= 0){
			break;
		}

		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));

		gaussianFilterKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, deviceFilter, FILTER_SIZE, srcRows, srcCols, chunkRows, offset);

		cudaMemcpy(output + offset, deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += destN;
	}

	cudaFree(deviceSrcData);
	cudaFree(deviceFilter);
}

void cudafGaussianFilter(float* input, float* output, double* gKernel, int srcRows, int srcCols){
	get_nvtAttrib("Setup", 0xFF0000FF);
	float* deviceSrcData;
	float* deviceDestData;
	double* deviceFilter;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int datasize = sizeof(float);
	int srcN = srcRows * srcCols;

	cudaMalloc(&deviceFilter, (2 * FILTER_SIZE + 1) * (2 * FILTER_SIZE + 1) * sizeof(double));
	cudaMemcpy(deviceFilter, gKernel, (2 * FILTER_SIZE + 1) * (2 * FILTER_SIZE + 1) * sizeof(double), cudaMemcpyHostToDevice);

	int remainder = srcRows * srcCols;
	int pixels = MEM_CAP / 4.0 * datasize;

	chunkRows = MEM_CAP / srcCols;
	if (chunkRows == 0){
		chunkRows = srcRows;
	}
	int rounds = ceil(srcRows / (double)chunkRows);

	int offset = 0;
	nvtxRangePop();//Setup
	//for (int step = 0; step < rounds; step++){
	get_nvtAttrib("K Blocks " + std::to_string(rounds), 0xFF008800);
	while (remainder > 0){
		get_nvtAttrib("Pre-K", 0xFFFF0000);
		int pix_begin = (srcRows * srcCols) - remainder;
		int pix_end = min(srcRows * srcCols - 1, pix_begin + pixels - 1);

		//int destN = fmin(pixels, remainder);
		int destN = (pix_end - pix_begin + 1) * datasize;
		if (destN <= 0){
			break;
		}

		blocks = ((destN / datasize) + threadsPerBlock - 1) / threadsPerBlock;

		int src_begin = max((pix_begin - FILTER_SIZE) - (FILTER_SIZE * srcCols), 0);
		int src_end = min((pix_end + FILTER_SIZE) + (FILTER_SIZE * srcCols), srcRows * srcCols - 1);
		int srcN = (src_end - src_begin + 1) * datasize;

		get_nvtAttrib("Malloc", 0xFFFF0000);
		cudaMalloc(&deviceDestData, destN);
		cudaMalloc(&deviceSrcData, srcN);
		nvtxRangePop();//Malloc

		get_nvtAttrib("Memcpy", 0xFFFF0000);
		cudaMemcpy(deviceSrcData, input + src_begin, srcN, cudaMemcpyHostToDevice);
		nvtxRangePop();//Memcpy
		
		nvtxRangePop();//Pre-K

		get_nvtAttrib("Kern", 0xFF00FF00);
		fGaussianFilterKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, deviceFilter, FILTER_SIZE, srcRows, srcCols, src_begin, pix_begin);
		cudaDeviceSynchronize();
		nvtxRangePop();//Kern

		get_nvtAttrib("Post-K", 0xFF0000FF);
		get_nvtAttrib("Memcpy", 0xFF0000FF);
		cudaMemcpy(output + pix_begin, deviceDestData, destN, cudaMemcpyDeviceToHost);
		nvtxRangePop();//Memcpy

		get_nvtAttrib("cudaFree", 0xFF0000FF);
		cudaFree(deviceDestData);
		cudaFree(deviceSrcData);
		nvtxRangePop();//cudaFree

		offset += destN;
		remainder -= pixels;
		nvtxRangePop();//Post-K
	}
	nvtxRangePop();//K Blocks
	cudaFree(deviceFilter);
	//cudaFree(deviceSrcData);

}

void cudaMyConv2(float* temp, float* large, float* small, int tRows, int tCols, int lRows, int lCols, int sRows, int sCols){
	float* deviceTemp;
	float* deviceLarge;
	float* deviceSmall;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int datasize = sizeof(float);

	int i1, j1, i2, j2;
	int stRow, sdRow, stCol, sdCol;
	int mtRow, mdRow, mtCol, mdCol;
	int ks, km, ls, lm;
	int k, l, m, n;

	int remainder = tRows * tCols;
	int pixels = MEM_CAP / (2.1 * datasize) ;

	cudaMalloc(&deviceSmall, (sRows * sCols) * datasize);
	cudaMemcpy(deviceSmall, small, (sRows * sCols) * datasize, cudaMemcpyHostToDevice);

	int rounds = 0;

	while (true){
		if (remainder <= 0){
			break;
		}
		//printf("Remainder: %d, Pixels: %d\n", remainder, pixels);
		int pix_begin = (tRows * tCols) - remainder;
		int pix_end = min(tRows * tCols - 1, pix_begin + pixels - 1);
		i1 = pix_begin / tCols;
		j1 = pix_begin % tCols;
		i2 = pix_end / tCols;
		j2 = pix_end % tCols;

		stRow = i1 - sRows + 1;
		sdRow = max(0, stRow);
		stCol = j1 - sCols + 1;
		sdCol = max(0, stCol);
		mtRow = stRow + sRows;
		mdRow = min(lRows, stRow + sRows);
		mtCol = stCol + sCols;
		mdCol = min(lCols, stCol + sCols);
		ks = sdRow - stRow; km = sRows - (mtRow - mdRow);
		ls = sdCol - stCol; lm = sCols - (mtCol - mdCol);
		int m_begin = sdRow;
		int n_begin = sdCol;
		int lrg_begin = m_begin * lCols + n_begin;

		stRow = i2 - sRows + 1;
		sdRow = max(0, stRow);
		stCol = j2 - sCols + 1;
		sdCol = max(0, stCol);
		mtRow = stRow + sRows;
		mdRow = min(lRows, stRow + sRows);
		mtCol = stCol + sCols;
		mdCol = min(lCols, stCol + sCols);
		ks = sdRow - stRow; km = sRows - (mtRow - mdRow);
		ls = sdCol - stCol; lm = sCols - (mtCol - mdCol);
		int m_end = sdRow + (km - ks - 1);
		int n_end = sdCol + (lm - ls - 1);
		int lrg_end = m_end * lCols + n_end;

		int tmpN = (pix_end - pix_begin + 1) * datasize;
		int lrgN = (lrg_end - lrg_begin + 1) * datasize;

		cudaMalloc(&deviceTemp, tmpN);
		cudaMalloc(&deviceLarge, lrgN);

		blocks = (tmpN / datasize + threadsPerBlock - 1) / threadsPerBlock;
		if (blocks == 0) { blocks = 1; }

		/*if (lrg_begin > 300000){
			cudaFree(deviceTemp);
			cudaFree(deviceLarge);
			break;
		}*/

		//printf("    tmp_begin: %d, tmp_end: %d, tmp_limit: %d, tmpN: %d\n", pix_begin, pix_end, tRows * tCols, tmpN);
		//printf("    lrg_begin: %d, lrg_end: %d, lrg_limit: %d, lrgN: %d\n", lrg_begin, lrg_end, lRows * lCols, lrgN);
		cudaMemcpy(deviceLarge, large + lrg_begin, lrgN, cudaMemcpyHostToDevice);
		myConv2Kernel << <blocks, threadsPerBlock >> > (deviceTemp, deviceLarge, deviceSmall, tRows, tCols, lRows, lCols, sRows, sCols, pix_begin, lrg_begin);
		

		cudaMemcpy(temp + pix_begin, deviceTemp, tmpN, cudaMemcpyDeviceToHost);
		//temp[pix_begin] = 255;
		//temp[pix_end] = 255;
		remainder -= pixels;

		cudaDeviceSynchronize();
		cudaFree(deviceTemp);
		cudaFree(deviceLarge);
	}

	cudaFree(deviceSmall);
}

void cudaSobelFilter(unsigned char* input, unsigned char* output, int srcRows, int srcCols){
	get_nvtAttrib("Setup Inner", 0xFF000088);
	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
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

	cudaMalloc(&deviceSrcData, srcN*sizeof(unsigned char));
	cudaMalloc(&deviceSobel_x, 9 * sizeof(int));
	cudaMalloc(&deviceSobel_y, 9 * sizeof(int));
	cudaMalloc(&deviceRangeMin, sizeof(double));
	cudaMalloc(&deviceRangeMax, sizeof(double));

	cudaMemcpy(deviceSrcData, input, srcN*sizeof(unsigned char), cudaMemcpyHostToDevice);
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
	nvtxRangePop();

	get_nvtAttrib("Gradient", 0xFF888888);
	for (int step = 0; step < rounds; step++){
		int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
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
	nvtxRangePop();
	//printf("Works!\n");

	//cudaMemcpy(rangeMin, deviceRangeMin, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(rangeMax, deviceRangeMax, sizeof(double), cudaMemcpyDeviceToHost);

	//printf("Host temp data done");

	get_nvtAttrib("Range Find", 0xFF008800);
	for (int i = 0; i < srcRows; i++){
		for (int j = 0; j < srcCols; j++){
			for (int color = 0; color < 3; color++){
				double value = temp_data[3 * (i * srcCols + j) + color];;
				rangeMin[0] = std::fmin(value, rangeMin[0]);
				rangeMax[0] = std::fmax(value, rangeMax[0]);
			}
		}
	}
	nvtxRangePop();

	//printf("Got here!\n");
	//output = (uchar*)temp_data;
	//return;

	//printf("Range Min: %f, Range Max: %f \n", rangeMin[0], rangeMax[0]);

	//blocks = (srcN + threadsPerBlock - 1) / threadsPerBlock;
	offset = 0;

	get_nvtAttrib("Range", 0xFF888888);
	for (int step = 0; step < rounds; step++){
		int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
		if (destN <= 0){
			break;
		}
		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceTempData, destN*sizeof(short));
		cudaMemcpy(deviceTempData, temp_data + offset, destN*sizeof(short), cudaMemcpyHostToDevice);

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));
		sobelRangeKernel << <blocks, threadsPerBlock >> >(deviceDestData, deviceTempData, rangeMin[0], rangeMax[0], 20, 60, offset);

		cudaMemcpy(output + offset, deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		cudaFree(deviceDestData);
		cudaFree(deviceTempData);
		offset += destN;
	}
	nvtxRangePop();
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
	}*/


	//cudaFree(deviceTempData);
	//cudaFree(deviceDestData);

	delete[] temp_data;
}

void cudaKMeans(unsigned char* input, unsigned char* output, int srcRows, int srcCols, int k_means){
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;

	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
	float* device_k_colors;
	int* device_k_count;
	//int* device_hits;
	unsigned char* device_k_index;
	bool* device_convergence;

	float* k_colors = new float[k_means * 3];
	unsigned char* k_index = new unsigned char[srcRows * srcCols];
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
		//convergence[0] = true; //UNDO
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
			int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
			if (destN <= 0){
				break;
			}
			blocks = ((destN / 3) + threadsPerBlock - 1) / threadsPerBlock;

			cudaMalloc(&deviceSrcData, destN*sizeof(unsigned char));
			cudaMemcpy(deviceSrcData, input + offset, destN*sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMalloc(&device_k_index, destN*sizeof(unsigned char) / 3);
			cudaMemcpy(device_k_index, k_index + (offset / 3), destN*sizeof(unsigned char) / 3, cudaMemcpyHostToDevice);

			//kernel
			//kMeansCountingKernel << <blocks, threadsPerBlock >> > (deviceSrcData, device_k_index, device_k_count, device_hits, device_k_colors, device_convergence, k_means, srcRows, srcCols,count);
			kMeansCountingKernel << <blocks, threadsPerBlock >> > (deviceSrcData, device_k_index, device_k_count, device_k_colors, device_convergence, k_means, srcRows, srcCols);

			cudaMemcpy(k_index + (offset / 3), device_k_index, destN*sizeof(unsigned char) / 3, cudaMemcpyDeviceToHost);

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
			int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
			if (destN <= 0){
				break;
			}
			blocks = ((destN / 3) + threadsPerBlock - 1) / threadsPerBlock;

			cudaMalloc(&deviceSrcData, destN*sizeof(unsigned char));
			cudaMemcpy(deviceSrcData, input + offset, destN*sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMalloc(&device_k_index, destN*sizeof(unsigned char) / 3);
			cudaMemcpy(device_k_index, k_index + (offset / 3), destN*sizeof(unsigned char) / 3, cudaMemcpyHostToDevice);

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
		int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
		if (destN <= 0){
			break;
		}
		blocks = ((destN / 3) + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));
		cudaMalloc(&device_k_index, destN*sizeof(unsigned char) / 3);
		cudaMemcpy(device_k_index, k_index + (offset / 3), destN*sizeof(unsigned char) / 3, cudaMemcpyHostToDevice);

		//kernel
		kMeansOutputKernel << <blocks, threadsPerBlock >> > (deviceDestData, device_k_index, device_k_colors, srcRows, srcCols);
		cudaMemcpy(output + offset, deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);

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

}

void cudaKMeansFixed(unsigned char* input, unsigned char* output, int srcRows, int srcCols, int k_means){
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int offset = 0;

	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
	int* device_k_colors;
	int* device_k_count;
	//int* device_hits;
	unsigned char* device_k_index;
	bool* device_convergence;

	//float* k_colors = new float[k_means * 3];
	int* k_colors = new int[k_means * 3];
	unsigned char* k_index = new unsigned char[srcRows * srcCols];
	int* k_count = new int[k_means];
	//* hits = new int[k_means];

	int srcN = srcRows * srcCols * 3;

	for (int pix = 0; pix < k_means; pix++){
		int i = rand() % srcRows;
		int j = rand() % srcCols;
		for (int color = 0; color < 3; color++){
			k_colors[3 * pix + color] = input[3 * (i * srcCols + j) + color] * ONE;
		}
	}
	cudaMalloc(&device_k_colors, (3 * k_means)*sizeof(int));
	cudaMemcpy(device_k_colors, k_colors, 3 * k_means *sizeof(int), cudaMemcpyHostToDevice);

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
		//convergence[0] = true; //UNDO
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
			int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
			if (destN <= 0){
				break;
			}
			blocks = ((destN / 3) + threadsPerBlock - 1) / threadsPerBlock;

			cudaMalloc(&deviceSrcData, destN*sizeof(unsigned char));
			cudaMemcpy(deviceSrcData, input + offset, destN*sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMalloc(&device_k_index, destN*sizeof(unsigned char) / 3);
			cudaMemcpy(device_k_index, k_index + (offset / 3), destN*sizeof(unsigned char) / 3, cudaMemcpyHostToDevice);

			//kernel
			//kMeansCountingKernel << <blocks, threadsPerBlock >> > (deviceSrcData, device_k_index, device_k_count, device_hits, device_k_colors, device_convergence, k_means, srcRows, srcCols,count);
			kMeansCountingKernelFixed << <blocks, threadsPerBlock >> > (deviceSrcData, device_k_index, device_k_count, device_k_colors, device_convergence, k_means, srcRows, srcCols);

			cudaMemcpy(k_index + (offset / 3), device_k_index, destN*sizeof(unsigned char) / 3, cudaMemcpyDeviceToHost);

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
		cudaMemcpy(device_k_colors, k_colors, 3 * k_means *sizeof(int), cudaMemcpyHostToDevice);

		rounds = ceil(srcRows / (float)chunkRows);

		offset = 0;
		for (int step = 0; step < rounds; step++){
			int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
			if (destN <= 0){
				break;
			}
			blocks = ((destN / 3) + threadsPerBlock - 1) / threadsPerBlock;

			cudaMalloc(&deviceSrcData, destN*sizeof(unsigned char));
			cudaMemcpy(deviceSrcData, input + offset, destN*sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMalloc(&device_k_index, destN*sizeof(unsigned char) / 3);
			cudaMemcpy(device_k_index, k_index + (offset / 3), destN*sizeof(unsigned char) / 3, cudaMemcpyHostToDevice);

			kMeansGroupAdjustKernelFixed << <blocks, threadsPerBlock >> > (deviceSrcData, device_k_index, device_k_count, device_k_colors, k_means, srcRows, srcCols);

			cudaFree(deviceSrcData);
			cudaFree(device_k_index);

			offset += destN;
		}
		cudaMemcpy(k_colors, device_k_colors, 3 * k_means * sizeof(int), cudaMemcpyDeviceToHost);
		//kernel
	}

	int rounds = ceil(srcRows / (float)chunkRows);
	offset = 0;
	for (int step = 0; step < rounds; step++){
		int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
		if (destN <= 0){
			break;
		}
		blocks = ((destN / 3) + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));
		cudaMalloc(&device_k_index, destN*sizeof(unsigned char) / 3);
		cudaMemcpy(device_k_index, k_index + (offset / 3), destN*sizeof(unsigned char) / 3, cudaMemcpyHostToDevice);

		//kernel
		kMeansOutputKernelFixed << <blocks, threadsPerBlock >> > (deviceDestData, device_k_index, device_k_colors, srcRows, srcCols);
		cudaMemcpy(output + offset, deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);

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

}

void cudaMySiftDOG(float* current, float* next, float* dog, int curRows, int curCols){
	float* deviceCurrData;
	float* deviceNextData;
	float* deviceDogData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int datasize = sizeof(float);

	int pixels = MEM_CAP / (3 * datasize);

	int remainder = curRows * curCols;

	while (remainder > 0){

		int pix_begin = (curRows * curCols) - remainder;
		int pix_end = min(curRows * curCols - 1, pix_begin + pixels - 1);

		int destN = (pix_end - pix_begin + 1) * datasize;
		int blocks = ((destN / datasize) + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceCurrData, destN);
		cudaMalloc(&deviceNextData, destN);
		cudaMalloc(&deviceDogData, destN);
		cudaMemcpy(deviceCurrData, current + pix_begin, destN, cudaMemcpyHostToDevice);
		cudaMemcpy(deviceNextData, next + pix_begin, destN, cudaMemcpyHostToDevice);

		mySiftDOGKernel << <blocks, threadsPerBlock >> >(deviceCurrData, deviceNextData, deviceDogData);

		cudaMemcpy(dog + pix_begin, deviceDogData, destN, cudaMemcpyDeviceToHost);

		cudaFree(deviceCurrData);
		cudaFree(deviceNextData);
		cudaFree(deviceDogData);

		remainder -= pixels;
	}

}

void cudaMySiftKeypoints(float* prev_data, float* curr_data, float* next_data, char* answers, unsigned int* key_str, int curRows, int curCols, int key_str_size){
	float* devicePrevData;
	float* deviceCurrData;
	float* deviceNextData;
	char*  deviceAnswers;
	unsigned int* deviceKeyStr;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int datasize = sizeof(float);
	int keysize = sizeof(unsigned int);
	int keybits = keysize * 8;

	int pixels = (MEM_CAP) / (3.5 * datasize);
	pixels -= ceil(pixels / (float)keybits);
	//printf("Pixels: %d\n", pixels);

	int remainder = curRows * curCols;

	while (remainder > 0){

		int pix_begin = (curRows * curCols) - remainder;
		int pix_end = min(curRows * curCols - 1, pix_begin + pixels - 1);

		int block_begin = (pix_begin) / keybits;
		//int block_end = min(key_str_size - 1, (pix_end / keysize));
		int block_end = (pix_end) / keybits;

		int src_begin = max(0, pix_begin - curCols - 1);
		int src_end = min(curRows * curCols - 1, pix_end + curCols + 1);

		int pixN = (pix_end - pix_begin + 1) * datasize;
		int ansN = (pix_end - pix_begin + 1);
		int srcN = (src_end - src_begin + 1) * datasize;
		int blocks = ((pixN / datasize) + threadsPerBlock - 1) / threadsPerBlock;
		int strN = (block_end - block_begin + 1) * keysize;

		cudaMalloc(&devicePrevData, srcN);
		cudaMalloc(&deviceCurrData, srcN);
		cudaMalloc(&deviceNextData, srcN);
		cudaMalloc(&deviceAnswers, ansN);
		//cudaMalloc(&deviceKeyStr, strN);
		cudaMemcpy(devicePrevData, prev_data + src_begin, srcN, cudaMemcpyHostToDevice);
		cudaMemcpy(deviceCurrData, curr_data + src_begin, srcN, cudaMemcpyHostToDevice);
		cudaMemcpy(deviceNextData, next_data + src_begin, srcN, cudaMemcpyHostToDevice);
		//cudaMemcpy(deviceKeyStr, key_str + block_begin, strN, cudaMemcpyHostToDevice);

		mySiftKeypointsKernel << <blocks, threadsPerBlock >> >(devicePrevData, deviceCurrData, deviceNextData, deviceAnswers, curRows, curCols, pix_begin, src_begin, block_begin, keybits);

		//cudaMemcpy(key_str + block_begin, deviceKeyStr, strN, cudaMemcpyDeviceToHost);
		cudaMemcpy(answers + pix_begin, deviceAnswers, ansN, cudaMemcpyDeviceToHost);

		cudaFree(devicePrevData);
		cudaFree(deviceCurrData);
		cudaFree(deviceNextData);
		cudaFree(deviceAnswers);
		//cudaFree(deviceKeyStr);

		remainder -= pixels;
	}
}

/*void cudaMySiftOrMagGen(float* curr_data, float* or_mag, int curRows, int curCols){
	float* deviceCurrData;
	float* deviceOrMag;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int datasize = sizeof(float);

	int offset = 0;
	int remainder = curRows * curCols;

	int pixels = (MEM_CAP) / (3.2 * datasize);
	//printf("Pixels: %d\n", pixels);

	while (remainder > 0){

		int pix_begin = (curRows * curCols - remainder);
		int pix_end = min(curRows * curCols - 1, pix_begin + pixels - 1);
		int pixN = (pix_end - pix_begin + 1) * datasize;

		int src_begin = max(0, pix_begin - curCols - 1);
		int src_end = min(curRows * curCols - 1, pix_end + curCols + 1);
		//printf("src_begin: %d, src_end: %d\n", src_begin, src_end);
		int srcN = (src_end - src_begin + 1) * datasize;
		int blocks = ((pixN / datasize) + threadsPerBlock - 1) / threadsPerBlock;

		//printf("pixN: %d, srcN: %d\n", pixN, srcN);

		//cudaMalloc(&deviceCurrData, srcN);
		//cudaMalloc(&deviceOrMag, pixN);
		//printf("Test: %d\n", curr_data);
		//cudaMemcpy(deviceCurrData, curr_data + src_begin, srcN, cudaMemcpyHostToDevice);

		//printf("pix_begin: %d, pix_end: %d, pixN: %d, blocks: %d\n", pix_begin, pix_end, pixN, blocks);
		//printf("Blocks: %d, ThreadsPerBlock: %d\n", blocks, threadsPerBlock);

		//mySiftOrMagKernel << <blocks, threadsPerBlock >> >(NULL, NULL, curRows, curCols, pix_begin, pix_end, src_begin, src_end);
		testKernel << <blocks, threadsPerBlock >> >(NULL);
		
		//printf("  Threads\n");
		//cudaMemcpy(or_mag + (pix_begin * 2), deviceOrMag, pixN, cudaMemcpyDeviceToHost);
		//printf("  Threads2\n");

		//cudaFree(deviceOrMag);
		//cudaFree(deviceCurrData);

		offset += pixels;
		remainder -= pixels;
	}
}*/

void cudaTest(int curRows, int curCols){
	//cudaDeviceSynchronize();
	//float* deviceCurrData;
	//float* deviceOrMag;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int datasize = sizeof(float);

	int remainder = curRows * curCols;

	int pixels = MEM_CAP / (3.5 * datasize);
	int blocks = 1;
	testKernel <<<20, 256 >>>();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err || true){
		printf("Fish %s\n", cudaGetErrorString(err));
	}
	gpuErrchk(cudaDeviceSynchronize());

	remainder -= pixels;
}

void cudaMySiftOrMagGen(float* curr_data, float* or_mag, int curRows, int curCols){
	float* deviceCurrData;
	float* deviceOrMag;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int datasize = sizeof(float);

	int offset = 0;
	int remainder = curRows * curCols;

	int pixels = (MEM_CAP) / (3.2 * datasize);

	while (remainder > 0){

		int pix_begin = (curRows * curCols - remainder);
		int pix_end = min(curRows * curCols - 1, pix_begin + pixels - 1);
		int pixN = (pix_end - pix_begin + 1) * 2 * datasize;

		int src_begin = max(0, pix_begin - curCols - 1);
		int src_end = min(curRows * curCols - 1, pix_end + curCols + 1);
		int srcN = (src_end - src_begin + 1) * datasize;
		int blocks = ((pixN / (2 * datasize)) + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceCurrData, srcN);
		cudaMalloc(&deviceOrMag, pixN);
		cudaMemcpy(deviceCurrData, curr_data + src_begin, srcN, cudaMemcpyHostToDevice);

		mySiftOrMagKernel << <blocks, threadsPerBlock >> >(deviceCurrData, deviceOrMag, curRows, curCols, pix_begin, pix_end, src_begin, src_end);
		//printf("  Fish\n");
		cudaMemcpy(or_mag + pix_begin * 2, deviceOrMag, pixN, cudaMemcpyDeviceToHost);

		cudaFree(deviceCurrData);
		cudaFree(deviceOrMag);

		offset += pixels;
		remainder -= pixels;
	}
}

void cudaMySiftCountSort(unsigned int* data, unsigned int* index, int d, int exp){
	//printf("Start Count Sort\n");
	get_nvtAttrib("Count Sort", 0xFF000088);
	unsigned int* deviceData;
	unsigned int* deviceOutData;
	unsigned int* deviceIndex;
	unsigned int* deviceOutIndex;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int srcN = d * sizeof(unsigned int);
	int blocks = (d + threadsPerBlock - 1) / threadsPerBlock;
	//int exp_cur = 1;

	int* count = new int[10];
	for (int i = 0; i < 10; i++) count[i] = 0;
	int* deviceCount;
	cudaMalloc((void**)&deviceData, srcN);
	cudaMalloc((void**)&deviceOutData, srcN);
	cudaMalloc((void**)&deviceIndex, srcN);
	cudaMalloc((void**)&deviceOutIndex, srcN);
	cudaMalloc((void**)&deviceCount, 10 * sizeof(int));
	//printf("Test\n");

	//gpuErrchk(cudaMemcpy(deviceData, data, srcN, cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(deviceIndex, index, srcN, cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(deviceCount, count, 10 * sizeof(int), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaPeekAtLastError());

	

	cudaMemcpy(deviceData, data, srcN, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceIndex, index, srcN, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceCount, count, 10 * sizeof(int), cudaMemcpyHostToDevice);

	//printf("Setup\n");

	int exp_curr = 1;
	while (exp / exp_curr > 0){
		//printf("Loop: %d\n", exp_curr);
		//printf("Memcpy In: %d\n", exp_curr);

		mySiftCountingKernel << <blocks, threadsPerBlock >> > (deviceData, deviceCount, exp_curr, d);
		//gpuErrchk(cudaPeekAtLastError());
		cudaDeviceSynchronize();
		
		cudaMemcpy(count, deviceCount, 10 * sizeof(int), cudaMemcpyDeviceToHost);
		//printf("Memcpy Out: %d\n", exp_curr);

		for (int i = 1; i < 10; i++){
			count[i] += count[i - 1];
		}
		cudaMemcpy(deviceCount, count, 10 * sizeof(int), cudaMemcpyHostToDevice);

		mySiftCountSortKernel << <blocks, threadsPerBlock >> > (deviceData, deviceOutData, deviceIndex, deviceOutIndex, deviceCount, exp_curr, d);
		cudaDeviceSynchronize();
		mySiftCountSortSwitchKernel << <blocks, threadsPerBlock >> > (deviceData, deviceOutData, deviceIndex, deviceOutIndex, deviceCount, exp_curr, d);

		//printf("Kernels: %d\n", exp_curr);

		
		for (int i = 0; i < 10; i++) count[i] = 0;
		exp_curr *= 10;
	}

	cudaMemcpy(data, deviceOutData, srcN, cudaMemcpyDeviceToHost);
	cudaMemcpy(index, deviceOutIndex, srcN, cudaMemcpyDeviceToHost);

	cudaFree(deviceData);
	cudaFree(deviceOutData);
	cudaFree(deviceIndex);
	cudaFree(deviceOutIndex);
	cudaFree(deviceCount);

	delete[] count;
	//mySiftCountSortKernel << <blocks, threadsPerBlock >> > (deviceData, deviceOutData, deviceIndex, deviceOutIndex, deviceCount, exp);
	nvtxRangePop();

}

#endif