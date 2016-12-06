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

Mat frgb2Gray(Mat image){
	Mat out = Mat(image.rows, image.cols, CV_32FC1);

	unsigned char* input = (unsigned char*)image.datastart;
	float* output = (float*)out.datastart;
	cudafRgb2Gray(input, output, image.rows, image.cols);

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

Mat fdirectResize(Mat image, int rows, int cols){
	Mat out = Mat::zeros(rows, cols, CV_32FC1);
	float* input = (float*)image.datastart;
	float* output = (float*)out.datastart;
	cudafDirectResize(input, output, image.rows, image.cols, rows, cols);

	return out;
}

Mat linearResize(Mat image, int rows, int cols){
	Mat out = Mat(rows, cols, image.type());

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaLinearResize(input, output, image.rows, image.cols, rows, cols);

	return out;
}

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

void createFilter(double* gKernel, double inputSigma, int filter_size){
	//standard deviation to 1.0
	double sigma = inputSigma;
	double r, s = 2.0 * sigma * sigma;
	int W = 2 * filter_size + 1;
	double mean = W / 2.0;
	double sum = 0.0;

	for (int x = -filter_size; x <= filter_size; x++){
		for (int y = -filter_size; y <= filter_size; y++){

			//Look up math for Gaussian Filtering
			r = sqrt(x * x + y * y);
			double val = (exp(-(r*r) / s)) / (M_PI * s);

			//double val = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (2 * M_PI * sigma * sigma);

			gKernel[((x + filter_size) * W) + (y + filter_size)] = val;
			sum += val;
		}
	}

	for (int i = 0; i < W; ++i){
		for (int j = 0; j < W; ++j){
			gKernel[(i * W) + j] /= sum;
		}
	}
}

Mat gaussianFilter(Mat image, double sigma){
	int W = 2 * FILTER_SIZE + 1;
	double* gKernel = new double[W * W];
	createFilter(gKernel, sigma, FILTER_SIZE);

	Mat out = Mat(image.rows, image.cols, image.type());
	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaGaussianFilter(input, output, gKernel, image.rows, image.cols);

	//delete[] gKernel;

	return out;
	//return image;
}

Mat fGaussianFilter(Mat image, double sigma){
	int W = 2 * FILTER_SIZE + 1;
	double* gKernel = new double[W * W];
	createFilter(gKernel, sigma, FILTER_SIZE);

	Mat out = Mat::zeros(image.rows, image.cols, CV_32FC1);
	float* input = (float*)image.datastart;
	float* output = (float*)out.datastart;
	fCudaGaussianFilter(input, output, gKernel, image.rows, image.cols);

	return out;

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

Mat mySift(Mat original){
	Mat out;

	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	std::string debug_Path = "D://School//Summer 2016//Research//mySift//";
	std::string img_name = "audrey";
	std::string ftype = ".png";

	int printoff = 0;	//	Debug, used to print to console the values of all keypoints found by this function
	int full_dog = 0;	//	Set this to 1 to change the output image to a full representation of the difference-of-gaussians and scale space location of each keypoint
	int mark_dim = 2;	//	Determines size of circles in the output image.  Recommend setting to 10 or higher if full_dog is set to 1

	//	The first step is to scale up the input image by a factor of 2 in both dimensions, then apply a gaussian blur with sigma = 1.6
	//	Scaling up the input provides more keypoints and approximates a blur with sigma = 1.0, assuming the original image is roughly sigma = 0.5 which is the threshold for noise
	//	Sigma = 1.6 was experimentally determined by Lowe to give the best results.  Refer to the 2004 SIFT paper, pages 9 and 10 for discussion
	float sigma = 1.6;



	return out;
}

#endif