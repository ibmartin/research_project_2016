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
#define MEM_CAP				32768 //32 KB as a power of 2
//#define MEM_CAP				16384 //32 KB as a power of 2

void cudaRgb2Gray(unsigned char* input, unsigned char* output, int srcRows, int srcCols){
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
}

void cudafRgb2Gray(unsigned char* input, float* output, int srcRows, int srcCols){
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

	for (int step = 0; step < rounds; step++){
		int remainder = fmin(chunk_size, (srcRows * srcCols) - (step * chunk_size) );
		int srcN = remainder * datasize1;
		int destN = remainder * datasize2;
		if (destN <= 0)
			break;

		blocks = ((destN / datasize2) + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceSrcData, srcN);
		cudaMemcpy(deviceSrcData, input + (step * chunk_size * 3), srcN, cudaMemcpyHostToDevice);

		cudaMalloc(&deviceDestData, destN);
		frgb2GrayKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, srcRows, srcCols, chunkRows, 0);
		cudaMemcpy(output + offset, deviceDestData, destN, cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += remainder;
	}
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
		int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - (offset * 3));
		if (destN <= 0){
			//printf("Broken!\n");
			break;
		}

		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));

		reverseKernel <<<blocks, threadsPerBlock >>>(deviceDestData, deviceSrcData, srcN, chunkRows, offset);
		cudaMemcpy(output + (3 * offset), deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += destN / 3;
	}


	cudaFree(deviceSrcData);

}

void cudaGammaCorrection(unsigned char* input, unsigned char* output, double gamma, int srcRows, int srcCols){
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

	for (int step = 0; step < rounds; step++){
		int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
		if (destN <= 0){
			break;
		}

		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(unsigned char));

		gammaCorrectionKernel <<<blocks, threadsPerBlock >>> (deviceDestData, deviceSrcData, srcRows, srcCols, gamma, chunkRows, offset);

		cudaMemcpy(output + offset, deviceDestData, destN*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);

		offset += destN;
	}

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

void cudafDirectResize(float* input, float* output, int srcRows, int srcCols, int destRows, int destCols){
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

		int sRow = ((float)pix_begin / destCols) * rRow;
		int sCol = (pix_begin % destCols) * rCol;
		int src_begin = (sRow * srcCols + sCol);

		sRow = ((float)pix_end / destCols) * rRow;	// (3531 / 1600) * (480 / 1200) = 0.88275
		sCol = (pix_end % destCols) * rCol;			// (3531 % 1600) * (640 / 1600) = 132.4
		int src_end = ((sRow + 1) * srcCols + sCol);

		//printf("srcRows: %d, srcCols: %d, total: %d\n", srcRows, srcCols, srcRows * srcCols);

		//printf("SBegin: %d, SEnd: %d\n", src_begin, src_end);

		int srcN = (src_end - src_begin + 1) * datasize;

		cudaMalloc(&deviceSrcData, srcN);
		cudaMalloc(&deviceDestData, destN);
		cudaMemcpy(deviceSrcData, input + src_begin, srcN, cudaMemcpyHostToDevice);

		fdirectResizeKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, srcRows, srcCols, destRows, destCols, chunkRows, offset);

		cudaMemcpy(output + pix_begin, deviceDestData, destN, cudaMemcpyDeviceToHost);
		
		cudaFree(deviceDestData);
		cudaFree(deviceSrcData);

		remainder -= pix_end - pix_begin + 1;
		sentinel -= 1;

	}

	


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

void fCudaGaussianFilter(float* input, float* output, double* gKernel, int srcRows, int srcCols){
	float* deviceSrcData;
	float* deviceDestData;
	double* deviceFilter;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = 0;
	int chunkRows = 0;
	int srcN = 3 * srcRows * srcCols;

	cudaMalloc(&deviceSrcData, srcN*sizeof(float));
	cudaMalloc(&deviceFilter, (2 * FILTER_SIZE + 1) * (2 * FILTER_SIZE + 1) * sizeof(double));
	cudaMemcpy(deviceSrcData, input, srcN*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceFilter, gKernel, (2 * FILTER_SIZE + 1)*(2 * FILTER_SIZE + 1)*sizeof(double), cudaMemcpyHostToDevice);

	chunkRows = IMG_CHUNK / srcCols;
	if (chunkRows == 0){
		chunkRows = srcRows;
	}
	int rounds = ceil(srcRows / (double)chunkRows);

	int offset = 0;
	for (int step = 0; step < rounds; step++){
		printf("Step\n");
		int destN = fmin(3 * chunkRows * srcCols, 3 * srcRows * srcCols - offset);
		if (destN <= 0){
			break;
		}

		blocks = (destN + threadsPerBlock - 1) / threadsPerBlock;

		cudaMalloc(&deviceDestData, destN*sizeof(float));
		//cudaMalloc(&deviceSrcData, destN*sizeof(float));

		//cudaMemcpy(deviceSrcData, input + offset, destN*sizeof(float), cudaMemcpyHostToDevice);

		//fGaussianFilterKernel << <blocks, threadsPerBlock >> > (deviceDestData, deviceSrcData, deviceFilter, FILTER_SIZE, srcRows, srcCols, chunkRows, offset);

		//cudaMemcpy(output + offset, deviceDestData, destN*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(deviceDestData);
		//cudaFree(deviceSrcData);

		offset += destN;
	}

	cudaFree(deviceFilter);
	cudaFree(deviceSrcData);

}

void cudaSobelFilter(unsigned char* input, unsigned char* output, int srcRows, int srcCols){
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
	//printf("Works!\n");

	//cudaMemcpy(rangeMin, deviceRangeMin, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(rangeMax, deviceRangeMax, sizeof(double), cudaMemcpyDeviceToHost);

	//printf("Host temp data done");

	for (int i = 0; i < srcRows; i++){
		for (int j = 0; j < srcCols; j++){
			for (int color = 0; color < 3; color++){
				double value = temp_data[3 * (i * srcCols + j) + color];;
				rangeMin[0] = std::fmin(value, rangeMin[0]);
				rangeMax[0] = std::fmax(value, rangeMax[0]);
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

void cudaKMeansOld(unsigned char* input, unsigned char* output, int srcRows, int srcCols, int k_means){
	int threadsPerBlock = 512;
	int blocks = ((srcRows * srcCols) + threadsPerBlock - 1) / threadsPerBlock;
	unsigned char* deviceSrcData;
	unsigned char* deviceDestData;
	float* device_k_colors;
	int* device_k_count;
	unsigned char* device_k_index;
	bool* device_convergence;
	//int srcN = 3 * srcRows * srcCols;
	int srcN = fmin(3 * 1920 * 1080, 3 * srcRows * srcCols);
	cudaMalloc(&deviceSrcData, srcN*sizeof(unsigned char));
	cudaMalloc(&deviceDestData, srcN*sizeof(unsigned char));
	cudaMalloc(&device_k_colors, (3 * k_means)*sizeof(float));
	cudaMalloc(&device_k_index, (srcRows * srcCols)*sizeof(unsigned char));
	cudaMalloc(&device_k_count, (k_means)*sizeof(int));
	cudaMalloc(&device_convergence, sizeof(bool));

	cudaMemcpy(deviceSrcData, input, srcN*sizeof(unsigned char), cudaMemcpyHostToDevice);

	float* k_colors = new float[3 * k_means];
	unsigned char* k_index = new unsigned char[srcRows * srcCols];
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
	cudaMemcpy(device_k_index, k_index, srcRows * srcCols *sizeof(unsigned char), cudaMemcpyHostToDevice);
	int count = 0;

	while (!convergence[0]){
		convergence[0] = true;
		cudaMemcpy(device_convergence, convergence, sizeof(bool), cudaMemcpyHostToDevice);
		for (int k = 0; k < k_means; k++){
			k_count[k] = 0;
		}
		cudaMemcpy(device_k_count, k_count, k_means * sizeof(int), cudaMemcpyHostToDevice);

		kMeansCountingKernelOld << <blocks, threadsPerBlock >> > (deviceSrcData, device_k_index, device_k_count, device_k_colors, device_convergence, k_means, srcRows, srcCols);
		cudaMemcpy(k_index, device_k_index, (srcRows*srcCols)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
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
	kMeansOutputKernel << <blocks, threadsPerBlock >> > (deviceDestData, device_k_index, device_k_colors, srcRows, srcCols);
	cudaMemcpy(output, deviceDestData, srcN*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(deviceSrcData);
	cudaFree(deviceDestData);
	cudaFree(device_k_colors);
	cudaFree(device_k_index);
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

#endif