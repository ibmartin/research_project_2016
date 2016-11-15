#ifndef _GPU_PROC_CU_
#define _GPU_PROC_CU_

#include "GpuProcHost.cu"
#include <stdio.h>
#include <iostream>
#include <fstream>

#include <opencv2\\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat rgb2Gray(Mat image){
	Mat out = Mat(image.rows, image.cols, CV_8UC1);

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaRgb2Gray(input, output, image.rows, image.cols);

	return out;
}

Mat reverse(Mat image){
	Mat out = Mat(image.rows, image.cols, image.type());

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaReverse(input, output, image.rows, image.cols);

	return out;
}

Mat gammaCorrection(Mat image, double gamma){
	Mat out = Mat(image.rows, image.cols, image.type());

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaGammaCorrection(input, output, gamma, image.rows, image.cols);

	return out;
}

Mat directResize(Mat image, int rows, int cols){
	Mat out = Mat(rows, cols, image.type());

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaDirectResize(input, output, image.rows, image.cols, rows, cols);

	return out;
}

Mat linearResize(Mat image, int rows, int cols){
	Mat out = Mat(rows, cols, image.type());

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaLinearResize(input, output, image.rows, image.cols, rows, cols);

	return out;
}

void createFilter(double gKernel[][2 * FILTER_SIZE + 1], double inputSigma){
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
}

Mat gaussianFilter(Mat image, double sigma){
	double gKernel[2 * FILTER_SIZE + 1][2 * FILTER_SIZE + 1];
	createFilter(gKernel, sigma);

	Mat out = Mat(image.rows, image.cols, image.type());
	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaGaussianFilter(input, output, gKernel, image.rows, image.cols);

	//delete[] gKernel;

	return out;
	//return image;
}

Mat sobelFilter(Mat image){
	Mat out(image.rows, image.cols, image.type());
	//Mat temp(image.rows, image.cols, image.type());
	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaSobelFilter(input, output, image.rows, image.cols);

	return out;
}

Mat kMeans(Mat image, int k_means){
	srand(6000);
	if (k_means > 256){
		printf("Error: Max number of groups exceeded (256)\n");
		exit(-1);
	}
	Mat out(image.rows, image.cols, image.type());
	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaKMeans(input, output, image.rows, image.cols, k_means);

	return out;
}

Mat gaussianPyramid(cv::Mat image, uchar levels, float scale){
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
}

#endif