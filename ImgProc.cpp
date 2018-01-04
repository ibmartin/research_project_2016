#include "ImgProc.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>

namespace img_proc{

	//Color channels in cv::Mat objects are arranged [B,G,R,...], in a single array of size 3 * rows * cols, row-major order.
	//The data type for colors is uchar, which is 1 Byte and ranges 0 to 255
	//Raw values of image data can be accessed using (uchar*)image.datastart

	static int FIXED = 16;
	static int ONE = 1 << FIXED;
	bool debug_statements = false;

	//	== cv::Mat rgb2Gray(cv::Mat image) ==
	//		cv::Mat image: input image
	//	Takes an RGB image and converts it to grayscale
	cv::Mat rgb2Gray(cv::Mat image){
		get_nvtAttrib("rgb2Gray CPU", 0xFF880000);
		int srcRows = image.rows;
		int srcCols = image.cols;

		//get_nvtAttrib("Setup", 0xFF88888888);
		cv::Mat output(srcRows, srcCols, CV_8UC1);
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_data = (uchar*)output.datastart;
		uchar* dest_end = (uchar*)output.dataend;
		//nvtxRangePop();

		//get_nvtAttrib("Pixels", 0xFF222222);
		while (dest_data <= dest_end){
			*(dest_data) = 0.299 * (*(src_data + 2)) + 0.587 * (*(src_data + 1)) + 0.114 * (*(src_data));	//Based on color curves I found representing contribution of each channel to grayscale sensitivity
			dest_data += 1;
			src_data += 3;
		}
		//nvtxRangePop();

		nvtxRangePop();
		return output;
	}

	cv::Mat frgb2Gray(cv::Mat image){
		get_nvtAttrib("frgb2Gray CPU", 0xFF880000);
		int srcRows = image.rows;
		int srcCols = image.cols;

		cv::Mat output = cv::Mat::zeros(srcRows, srcCols, CV_32FC1);
		uchar* src_data = (uchar*)image.datastart;
		float* dest_data = (float*)output.datastart;
		float* dest_end = (float*)output.dataend;

		while (dest_data <= dest_end){
			float temp = (0.299 * (float)(*(src_data + 2)) + 0.587 * (float)(*(src_data + 1)) + 0.114 * (float)(*(src_data)));
			//*(dest_data) = (0.299 * (*(src_data + 2)) + 0.587 * (*(src_data + 1)) + 0.114 * (*(src_data))) / 255.0;	//Based on color curves I found representing contribution of each channel to grayscale sensitivity
			*(dest_data) = temp;
			dest_data += 1;
			src_data += 3;
		}
		nvtxRangePop();
		return output;
	}

	//	== cv::Mat reverse(cv::Mat image) ==
	//		cv::Mat image: input image
	//	Outputs a negative color image of the input
	cv::Mat reverse(cv::Mat image){	
		get_nvtAttrib("reverse CPU", 0xFF880000);
		//int srcRows = image.rows;
		//int srcCols = image.cols;
		cv::Mat output(image.rows, image.cols, image.type());

		uchar* dest_data = (uchar*)output.datastart;
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_end = (uchar*)output.dataend;
		uchar* src_end = (uchar*)image.dataend;

		while (dest_data <= dest_end && src_data <= src_end){
			*(dest_data) = 255 - *(src_data);
			*(dest_data + 1) = 255 - *(src_data + 1);
			*(dest_data + 2) = 255 - *(src_data + 2);
			dest_data += 3;
			src_data += 3;
		}

		nvtxRangePop();
		return output;
	}

	//	== cv::Mat gammaCorrection(cv::Mat image, double gamma) ==
	//		cv::Mat image: input image
	//		double gamma: gamma correction value, gamma > 1 darkens the image, gamma < 1 brightens the image
	//	Applies gamma correction which brightens or darkens the image based on gamma
	cv::Mat gammaCorrection(cv::Mat image, double gamma){	
		get_nvtAttrib("gammaCorrection CPU", 0xFF880000);

		get_nvtAttrib("Setup", 0xFF000088);
		cv::Mat output(image.rows, image.cols, image.type());

		uchar* dest_data = (uchar*)output.datastart;
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_end = (uchar*)output.dataend;
		uchar* src_end = (uchar*)image.dataend;

		double gammaCorrect = 1.00 / gamma;
		nvtxRangePop();

		get_nvtAttrib("Work Loop " + std::to_string(image.rows * image.cols), 0xFF888888);
		while (dest_data <= dest_end && src_data <= src_end){
			double color = (double)(*(src_data));
			uchar val = 255 * pow((color / 255.0), gammaCorrect);
			*(dest_data) = val;
			src_data++;
			dest_data++;
		}
		nvtxRangePop();
		nvtxRangePop();
		return output;
	}

	//	== cv::Mat directResize(cv::Mat image, int rows, int cols) ==
	//		cv::Mat image: input image
	//		int rows: number of rows for the new image to have
	//		int cols: number of columns for the new image to have
	//	Resize with no interpolation, just chooses closest pixel
	cv::Mat directResize(cv::Mat image, int rows, int cols){
		get_nvtAttrib("directResize CPU", 0xFF880000);
		cv::Mat output(rows, cols, image.type());
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_data = (uchar*)output.datastart;

		int srcRows = image.rows;
		int srcCols = image.cols;
		double rRows = (double)srcRows / rows;	//Ratio of old to new size
		double rCols = (double)srcCols / cols;
		int posI, posJ;
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				posI = i * rRows;	//Estimating position of closest pixel from source image
				posJ = j * rCols;
				dest_data[3 * (cols * i + j)] = src_data[3 * (srcCols * posI + posJ)];
				dest_data[3 * (cols * i + j) + 1] = src_data[3 * (srcCols * posI + posJ) + 1];
				dest_data[3 * (cols * i + j) + 2] = src_data[3 * (srcCols * posI + posJ) + 2];
			}
		}

		nvtxRangePop();
		return output;
	}

	//	== cv::Mat fdirectResize(cv::Mat image, int rows, int cols) ==
	//		cv::Mat image: input image
	//		int rows: number of rows for the new image to have
	//		int cols: number of columns for the new image to have
	//	Resize with no interpolation, just chooses closest pixel
	cv::Mat fdirectResize(cv::Mat image, int destRows, int destCols){
		get_nvtAttrib("fdirectResize CPU", 0xFF880000);
		//cv::Mat output(rows, cols, CV_32FC1);
		int srcRows = image.rows;
		int srcCols = image.cols;
		cv::Mat out = cv::Mat::zeros(destRows, destCols, CV_32FC1);
		float* src_data = (float*)image.datastart;
		float* dest_data = (float*)out.datastart;

		float rRows = (float)srcRows / destRows;	//Ratio of old to new size
		float rCols = (float)srcCols / destCols;
		for (int i = 0; i < destRows; i++){
			for (int j = 0; j < destCols; j++){
				int idx = i * destCols + j;
				float rRow = (float)srcRows / destRows;
				float rCol = (float)srcCols / destCols;

				int sRow = ((idx) / destCols) * rRow;
				int sCol = ((idx) % destCols) * rCol;

				//dest_data[idx] = src_data[(int)(sRow * srcCols + sCol)];
				dest_data[idx] = src_data[(int)(sRow * srcCols + sCol)];
			}
		}
		nvtxRangePop();
		return out;
	}

	//	== cv::Mat linearResize(cv::Mat image, int rows, int cols) ==
	//		cv::Mat image: input image
	//		int rows: number of rows for the new image to have
	//		int cols: number of columns for the new image to have
	//	Resize with linear interpolation based on 4 nearest pixels
	cv::Mat linearResize(cv::Mat image, int rows, int cols){
		get_nvtAttrib("linearResize GPU", 0xFF880000);
		cv::Mat output(rows, cols, image.type());
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_data = (uchar*)output.datastart;

		int srcRows = image.rows;
		int srcCols = image.cols;
		double rRows = (double)srcRows / rows;
		double rCols = (double)srcCols / cols;
		int srcN = 3 * srcRows * srcCols;
		int posI, posJ;

		//printf("Rows: %d, Cols: %d\n", srcRows, srcCols);
		//uchar temp = 1 * src_data[3 * (srcCols * (1079 + 1) + (313 + 1)) + 2];

		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				posI = i * rRows;
				posJ = j * rCols;
				if ((i * rRows - posI) + (j * rCols - posJ) < 0.0000000001){
					dest_data[3 * (cols * i + j)] = src_data[3 * (srcCols * posI + posJ)];
					dest_data[3 * (cols * i + j) + 1] = src_data[3 * (srcCols * posI + posJ) + 1];
					dest_data[3 * (cols * i + j) + 2] = src_data[3 * (srcCols * posI + posJ) + 2];
				}
				else if (posI < srcRows - 1 && posJ < srcCols - 1){
					double deltaI = i * rRows - posI;
					double deltaJ = j * rCols - posJ;
					double area1 = (1 - deltaI) * (1 - deltaJ);	//Relative contributions of each of the neighbor pixels to the new color
					double area2 = deltaI * (1 - deltaJ);
					double area3 = deltaI * deltaJ;
					double area4 = (1 - deltaI) * deltaJ;

					//printf("PosI: %d, PosJ: %d \n", posI, posJ);
					if (3 * (posI * srcCols + posJ) + 2 > srcN){
						printf("Size Error\n");
					}

					//Acuumulating values
					uchar val = 0;
					val += area1 * src_data[3 * (srcCols * posI + posJ)];
					val += area2 * src_data[3 * (srcCols * (posI + 1) + posJ)];
					val += area3 * src_data[3 * (srcCols * (posI + 1) + (posJ + 1))];
					val += area4 * src_data[3 * (srcCols * posI + (posJ + 1))];

					dest_data[3 * (cols * i + j)] = val;

					val = 0;
					val += area1 * src_data[3 * (srcCols * posI + posJ) + 1];
					val += area2 * src_data[3 * (srcCols * (posI + 1) + posJ) + 1];
					val += area3 * src_data[3 * (srcCols * (posI + 1) + (posJ + 1)) + 1];
					val += area4 * src_data[3 * (srcCols * posI + (posJ + 1)) + 1];

					dest_data[3 * (cols * i + j) + 1] = val;

					val = 0;
					val += area1 * src_data[3 * (srcCols * posI + posJ) + 2];
					val += area2 * src_data[3 * (srcCols * (posI + 1) + posJ) + 2];
					val += area3 * src_data[3 * (srcCols * (posI + 1) + (posJ + 1)) + 2];
					val += area4 * src_data[3 * (srcCols * posI + (posJ + 1)) + 2];

					dest_data[3 * (cols * i + j) + 2] = val;
				}
			}
		}
		//uchar temp = 1 * src_data[3 * (srcCols * (1079 + 1) + (313 + 1)) + 2];
		nvtxRangePop();
		return output;
	}

	cv::Mat flinearResize(cv::Mat image, int rows, int cols){
		cv::Mat output = cv::Mat::zeros(rows, cols, CV_32FC1);
		//cv::Mat output(rows, cols, image.type());
		float* src_data = (float*)image.datastart;
		float* dest_data = (float*)output.datastart;

		int srcRows = image.rows;
		int srcCols = image.cols;
		float rRows = (float)srcRows / rows;
		float rCols = (float)srcCols / cols;
		int srcN = srcRows * srcCols;
		int posI, posJ;

		//printf("Rows: %d, Cols: %d\n", srcRows, srcCols);
		//uchar temp = 1 * src_data[3 * (srcCols * (1079 + 1) + (313 + 1)) + 2];

		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				posI = i * rRows;
				posJ = j * rCols;
				if ((i * rRows - posI) + (j * rCols - posJ) < 0.0000000001){
					dest_data[cols * i + j] = src_data[srcCols * posI + posJ];
				}
				else if (posI < srcRows - 1 && posJ < srcCols - 1){
					float deltaI = i * rRows - posI;
					float deltaJ = j * rCols - posJ;
					float area1 = (1 - deltaI) * (1 - deltaJ);	//Relative contributions of each of the neighbor pixels to the new color
					float area2 = deltaI * (1 - deltaJ);
					float area3 = deltaI * deltaJ;
					float area4 = (1 - deltaI) * deltaJ;

					//printf("PosI: %d, PosJ: %d \n", posI, posJ);
					if (posI * srcCols + posJ > srcN){
						printf("Size Error\n");
					}

					//Acuumulating values
					float val = 0;
					val += area1 * src_data[srcCols * posI + posJ];
					val += area2 * src_data[srcCols * (posI + 1) + posJ];
					val += area3 * src_data[srcCols * (posI + 1) + (posJ + 1)];
					val += area4 * src_data[srcCols * posI + (posJ + 1)];

					dest_data[cols * i + j] = val;
				}
			}
		}
		//uchar temp = 1 * src_data[3 * (srcCols * (1079 + 1) + (313 + 1)) + 2];
		return output;
	}



	//	== void createFilter(double gKernel[][2 * FILTER_SIZE + 1], double inputSigma) ==
	//		double gKernel[][]: array that will contain the descreet Gaussian operand values, hardcoded to be of size (2 * FILTER_SIZE + 1)^2.  FILTER_SIZE is a global found in ImgProc.hpp
	//		double inputSigma: value of sigma to use to generate Gaussian operand values
	//	Helper function to create a discreet Gaussian operand to use for convolution
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
		//getchar();
	}

	cv::Mat makeFilter(float sigma){
		/*cv::Mat testIn(2, 2, CV_32FC1);
		testIn.at<float>(0, 0) = -1;
		testIn.at<float>(0, 1) = -1;
		testIn.at<float>(1, 0) = 1;
		testIn.at<float>(1, 1) = 1;

		cv::Mat testIn2(5, 5, CV_32FC1);
		for (int i = 1; i <= 25; i++){
			testIn2.at<float>(i - 1) = i;
		}
		cv::Mat testOut;
		//cv::filter2D(testIn, testOut, -1, testIn,cv::Point(-1,-1),0,cv::BORDER_ISOLATED);
		testOut = myConv2(testIn, testIn, 1);

		printf("\n");
		for (int i = 0; i < testOut.rows*testOut.cols; i++){
			printf(" %f ", testOut.at<float>(i));
			if ((i + 1) % testOut.cols == 0) printf("\n");
		}
		printf("\n");*/

		int dim = floor(6 * sigma);
		cv::Mat outFilter(1, dim, CV_32FC1);

		float bound = -(dim - 1.0) / 2.0;
		float sumh = 0.0;

		for (int i = 0; i < outFilter.cols; i++){
			float val = exp(-(bound * bound) / (2 * sigma * sigma));
			outFilter.at<float>(0, i) = val;
			sumh += val;
			bound += 1.0;
		}
		

		if (sumh != 0){
			for (int i = 0; i < outFilter.cols; i++){
				outFilter.at<float>(0, i) = outFilter.at<float>(0, i) / sumh;
			}
		}

		//printf("[");
		for (int i = 0; i < outFilter.cols; i++){
			//printf(" %f ", outFilter.at<float>(0, i));
		}
		//printf("] \n");

		return outFilter;
	}

	//	== cv::Mat gaussianFilter(cv::Mat image, double sigma) ==
	//		cv::Mat image: input image
	//		double sigma: scale (or intensity) of gaussian blur
	//	Applies a Gaussian blur to the input image of strength determine by sigma
	cv::Mat gaussianFilter(cv::Mat image, double sigma){
		//double gKernel[2 * FILTER_SIZE + 1][2 * FILTER_SIZE + 1];
		int W = 2 * FILTER_SIZE + 1;
		double* gKernel = new double[W * W];
		createFilter(gKernel, sigma, FILTER_SIZE);

		int srcRows = image.rows;
		int srcCols = image.cols;

		cv::Mat output(image.rows, image.cols, image.type());

		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_data = (uchar*)output.datastart;

		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				for (int color = 0; color < 3; color++){
					
					double tmp = 0.0;
					//Determining bounds of image to avoid indexing error
					//mink & maxk, minl & maxl specify range of convolution possible at current location in image
					int maxk = std::min(FILTER_SIZE, srcRows - i - 1);
					int mink = std::min(FILTER_SIZE, i);
					int maxl = std::min(FILTER_SIZE, srcCols - j - 1);
					int minl = std::min(FILTER_SIZE, j);
					for (int k = -mink; k <= maxk; k++){
						for (int l = -minl; l <= maxl; l++){
							if (i + k >= srcRows){
								printf("Error: i+k = %d \n", i + k);
							}
							tmp += gKernel[(l + FILTER_SIZE) * W + (k + FILTER_SIZE)] * src_data[3 * ((i + k) * srcCols + (j + l)) + color];
						}
					}
					dest_data[3 * (i * srcCols + j) + color] = (uchar)tmp;
					//}
				}
			}
		}

		return output;
	}

	cv::Mat fgaussianFilter(cv::Mat image, double sigma){
		//double gKernel[2 * FILTER_SIZE + 1][2 * FILTER_SIZE + 1];
		get_nvtAttrib("fgaussianFilter CPU", 0xFF880000);
		int W = 2 * FILTER_SIZE + 1;
		int center = W / 2;
		double* gKernel = new double[W * W];
		createFilter(gKernel, sigma, FILTER_SIZE);

		int srcRows = image.rows;
		int srcCols = image.cols;

		//cv::Mat output(image.rows, image.cols, image.type());
		get_nvtAttrib("out Mat", 0xFF008800);
		cv::Mat output = cv::Mat::zeros(image.rows,image.cols,CV_32FC1);
		nvtxRangePop();//out Mat

		float* src_data = (float*)image.datastart;
		float* dest_data = (float*)output.datastart;

		get_nvtAttrib("Work loop main", 0xFFBB0000);
		for (int i = 0; i < srcRows; i++){
			get_nvtAttrib("Work loop inner", 0xFFFF0000);
			for (int j = 0; j < srcCols; j++){
				float tmp = 0.0;
				for (int m = 0; m < W; m++){
					int mm = W - 1 - m;
					for (int n = 0; n < W; n++){
						int nn = W - 1 - n;
						int ii = i + (m - center);
						int jj = j + (n - center);
						if (ii < 0)	{ ii = srcRows + ii; }
						else if (ii >= srcRows)	{ ii = ii - srcRows; }
						if (jj < 0)	{ jj = srcCols + jj; }
						else if (jj >= srcCols)	{ jj = jj - srcCols; }
						if (ii >= 0 && ii < srcRows && jj >= 0 && jj < srcCols)
							tmp += src_data[ii * srcCols + jj] * gKernel[mm * W + nn];
					}
				}
				dest_data[i * srcCols + j] = tmp;
			}
			nvtxRangePop();//Work loop inner
		}
		nvtxRangePop();//Work loop main

		nvtxRangePop();//fgaussianFilter CPU
		return output;
	}

	cv::Mat fGaussianFilterSep(cv::Mat image, float sigma){
		int srcRows = image.rows, srcCols = image.cols;

		cv::Mat output = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
		cv::Mat temp = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);

		cv::Mat filter = makeFilter(sigma);

		temp = myConv2(image, filter, 0);
		output = myConv2(temp, myTranspose(filter), 0);

		return output;
	}

	cv::Mat myConv2(cv::Mat large, cv::Mat small, int mode = 0){
		//mode = 0 is 'same'
		//mode = 1 is 'full'
		if (small.rows > large.rows || small.cols > large.cols){
			return large;
		}

		int rows = large.rows + small.rows - 1;
		int cols = large.cols + small.cols - 1;
		int stRow, sdRow, stCol, sdCol;
		int mtRow, mdRow, mtCol, mdCol;
		int ks, km, ls, lm;
		int k, l, m, n;
		cv::Mat temp = cv::Mat::zeros(rows,cols,CV_32FC1);

		for (int i = 0; i < temp.rows; i++){
			for (int j = 0; j < temp.cols; j++){
				float val = 0;
				int value = 0;

				stRow = i - small.rows + 1;
				sdRow = std::max(0, stRow);
				stCol = j - small.cols + 1;
				sdCol = std::max(0, stCol);
				mtRow = stRow + small.rows;
				mdRow = std::min(large.rows, stRow + small.rows);
				mtCol = stCol + small.cols;
				mdCol = std::min(large.cols, stCol + small.cols);
				ks = sdRow - stRow; km = small.rows - (mtRow - mdRow);
				ls = sdCol - stCol; lm = small.cols - (mtCol - mdCol);

				int m_end = sdRow + (km - ks - 1);
				int n_end = sdCol + (lm - ls - 1);
				int lrg_end = m_end * large.cols + n_end;

				for (k = ks, m = sdRow; k < km; k++, m++){
					for (l = ls, n = sdCol; l < lm; l++, n++){
						//float largeval = large.at<float>(m, n);
						//float smallval = small.at<float>(k, l);
						//val += large.at<float>(m,n) * small.at<float>(small.rows - k - 1, small.cols - l - 1);
						value += (large.at<float>(m, n) * small.at<float>(small.rows - k - 1, small.cols - l - 1)) * ONE;
					}
				}


				//temp.at<float>(i, j) = val;
				temp.at<float>(i, j) = value / ONE;
			}
		}

		if (mode == 0){
			int top = small.rows / 2, bottom = (small.rows - 1) / 2, left = small.cols / 2, right = (small.cols - 1) / 2;
			return myTrim(temp, top, bottom, left, right);
		}
		return temp;

	}

	cv::Mat myTranspose(cv::Mat input){
		cv::Mat output(input.cols, input.rows, input.type());

		for (int i = 0; i < output.rows; i++){
			for (int j = 0; j < output.cols; j++){
				output.at<float>(i, j) = input.at<float>(j, i);
			}
		}
		return output;
	}

	cv::Mat myTrim(cv::Mat input, int top, int bottom, int left, int right){
		int rows = input.rows - top - bottom;
		int cols = input.cols - left - right;
		cv::Mat output(rows, cols, input.type());

		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				output.at<float>(i, j) = input.at<float>(i + top, j + left);
			}
		}
		return output;
	}

	cv::Mat myElemScalar(cv::Mat input, float scalar, int mode = 0){
		//mode == 0 is multiply
		//mode = -1 is divide
		//mode = 1 is add
		cv::Mat output(input.rows, input.cols, input.type());
		for (int i = 0; i < input.rows; i++){
			for (int j = 0; j < input.cols; j++){

				if (mode == 0){
					output.at<float>(i, j) = scalar * input.at<float>(i, j);
				}
				else if (mode == -1){
					output.at<float>(i, j) = scalar / input.at<float>(i, j);
				}
				else if (mode == 1){
					output.at<float>(i, j) = scalar + input.at<float>(i, j);
				}

			}
		}
		return output;
	}

	cv::Mat myElemMat(cv::Mat mat1, cv::Mat mat2, int mode = 0){
		//mode == 0 is multiply
		//mode == -1 is divide
		//mode == 1 is add
		//mode == 2 is subtract
		if (mat1.rows != mat2.rows || mat1.cols != mat2.cols){
			printf("Error: myElemMat\n");
			getchar();
			return mat1;
		}
		cv::Mat output(mat1.rows, mat1.cols, mat1.type());

		for (int i = 0; i < output.rows; i++){
			for (int j = 0; j < output.cols; j++){
				
				if (mode == 0){
					output.at<float>(i, j) = mat1.at<float>(i, j) * mat2.at<float>(i, j);
				}
				else if (mode == -1){
					output.at<float>(i, j) = mat1.at<float>(i, j) / mat2.at<float>(i, j);
				}
				else if (mode == 1){
					output.at<float>(i, j) = mat1.at<float>(i, j) + mat2.at<float>(i, j);
				}
				else if (mode == 2){
					output.at<float>(i, j) = mat1.at<float>(i, j) - mat2.at<float>(i, j);
				}

			}
		}

		return output;

	}

	//	== float* fGaussianFilter(float* image, double sigma, int srcRows, int srcCols) ==
	//		float* image: pointer to an array of floats that contain input image data
	//		double sigma: scale (or intensity) of gaussian blur
	//		int srcRows: number of rows for the input image
	//		int srcCols: number of columns for the input image
	//	Gaussian filter which works with float arrays instead of cv::Mat (essentially uchar arrays);  Possibly broken, ignore.
	/*float* fGaussianFilter(float* image, double sigma, int srcRows, int srcCols){	//Float image is same size as original, 3 * srcRows * srcCols
		double gKernel[2 * FILTER_SIZE + 1][2 * FILTER_SIZE + 1];
		createFilter(gKernel, sigma);

		float* output = new float[3 * srcRows * srcCols];

		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				for (int color = 0; color < 3; color++){

					float tmp = 0.0;
					int maxk = std::min(FILTER_SIZE, srcRows - i - 1);
					int mink = std::min(FILTER_SIZE, i);
					int maxl = std::min(FILTER_SIZE, srcCols - j - 1);
					int minl = std::min(FILTER_SIZE, j);
					for (int k = -mink; k <= maxk; k++){
						for (int l = -minl; l <= maxl; l++){
							tmp += gKernel[k + FILTER_SIZE][l + FILTER_SIZE] * image[3 * ((i + k) * srcCols + (j + l)) + color];
						}
					}
					output[3 * (i * srcCols + j) + color] = tmp;

				}
			}
		}

		return output;
	}*/

	//	== cv::Mat sobelFilter(cv::Mat image) ==
	//		cv::Mat image: input image
	//	Applies edge detection convolution of the sobel kernel
	//	Part of Canny edge detector.  Full functionality is not yet implemented
	cv::Mat sobelFilter(cv::Mat image){
		get_nvtAttrib("sobelFilter CPU", 0xFF880000);
		get_nvtAttrib("Setup", 0xFF000088);
		cv::Mat output(image.rows, image.cols, image.type());
		double sobel_x[3][3], sobel_xn[3][3], sobel_y[3][3], sobel_yn[3][3];

		sobel_x[0][0] = -1; sobel_x[0][1] = 0; sobel_x[0][2] = 1;
		sobel_x[1][0] = -2; sobel_x[1][1] = 0; sobel_x[1][2] = 2;
		sobel_x[2][0] = -1; sobel_x[2][1] = 0; sobel_x[2][2] = 1;

		sobel_y[0][0] = -1; sobel_y[0][1] = -2; sobel_y[0][2] = -1;
		sobel_y[1][0] = 0; sobel_y[1][1] = 0; sobel_y[1][2] = 0;
		sobel_y[2][0] = 1; sobel_y[2][1] = 2; sobel_y[2][2] = 1;

		int srcRows = image.rows;
		int srcCols = image.cols;

		//Sobel Filtering

		int flag = 1;

		//DISABLED//double * angle = new double[3 * srcRows * srcCols];			<-- Part of Canny, may use later
		//uchar * angle = new uchar[3 * srcRows * srcCols];
		//printf("Angle done\n");

		double rangeMax = 0, rangeMin = 0;

		uchar* src_data = (uchar*)image.datastart;
		uchar* fin_data = (uchar*)output.datastart;
		double* dest_data = new double[3 * srcRows * srcCols];
		nvtxRangePop();

		get_nvtAttrib("Work Loop", 0xFF888888);
		//Convolution of the Sobel kernel
		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				for (int color = 0; color < 3; color++){
					//Accumulators
					double tmpx = 0;
					double tmpy = 0;
					//Specifying range boundaries of convolution, for edges of image
					int maxk = std::min(1, srcRows - i - 1);
					int mink = std::min(1, i);
					int maxl = std::min(1, srcCols - j - 1);
					int minl = std::min(1, j);

					for (int k = -mink; k <= maxk; k++){
						for (int l = -minl; l <= maxl; l++){
							if (k > 1 || k < -1 || l > 1 || l < -1)
								printf("Error in Sobel!\n");
							tmpx = tmpx + sobel_x[k + 1][l + 1] * src_data[3 * ((i + k) * srcCols + (j + l)) + color];
							tmpy = tmpy + sobel_y[k + 1][l + 1] * src_data[3 * ((i + k) * srcCols + (j + l)) + color];
						}
					}


					double value = std::sqrt((tmpx * tmpx) + (tmpy * tmpy));
					rangeMin = std::min(value, rangeMin);
					rangeMax = std::max(value, rangeMax);

					dest_data[3 * (i * srcCols + j) + color] = value;
					//DISABLED//angle[3 * (i * srcCols + j) + color] = std::atan2(tmpy,tmpx);

				}
			}
		}
		nvtxRangePop();


		//Edge Thinning
		//Thins out pixel values that are too weak to represent a strong edge in the original image
		//Also strengthens strong edges

		get_nvtAttrib("Edge Thinning", 0xFF888888);
		double diff = rangeMax - rangeMin;
		int highThresh = 60;
		int lowThresh = 20;

		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				for (int color = 0; color < 3; color++){
					//printf("Stage 5 start... ");
					double value = 255 * ((dest_data[3 * (i * srcCols + j) + color] + rangeMin) / diff);
					if (value >= highThresh){
						value = 255;
					}
					else if (value < lowThresh){
						value = 0;
					}
					fin_data[3 * (i * srcCols + j) + color] = value;
					//printf("Stage 5 end\n");
				}
			}
		}
		nvtxRangePop();
		//The rest of this is part of Canny, not used right now.
		
		/*for (int i = 0; i < srcRows; i++){
		for (int j = 0; j < srcCols; j++){
		for (int color = 0; color < 3; color++){
		uchar max = 0;
		if (angle[3 * (i * srcCols + j) + color] == 0){
		if (i > 0 && i < srcRows - 1 && j > 0 && j < srcCols - 1)
		max = std::max(dest_data[3 * (i * srcCols + j - 1) + color], dest_data[3 * (i * srcCols + j + 1) + color]);
		else if (j == 0)
		max = dest_data[3 * (i * srcCols + j + 1) + color];
		else if (j == srcCols - 1)
		max = dest_data[3 * (i * srcCols + j - 1) + color];

		}
		else if (angle[3 * (i * srcCols + j) + color] == M_PI / 2){
		if (i > 0 && i < srcRows - 1 && j > 0 && j < srcCols - 1)
		max = std::max(dest_data[3 * ((i + 1) * srcCols + j) + color], dest_data[3 * ((i - 1) * srcCols + j) + color]);
		else if (i == 0)
		max = dest_data[3 * ((i + 1) * srcCols + j) + color];
		else if (i == srcRows - 1)
		max = dest_data[3 * ((i - 1) * srcCols + j) + color];
		}
		else if (angle[3 * (i * srcCols + j) + color] == M_PI / 4){
		if (i > 0 && i < srcRows - 1 && j > 0 && j < srcCols - 1)
		max = std::max(dest_data[3 * ((i + 1) * srcCols + j - 1) + color], dest_data[3 * ((i - 1) * srcCols + j + 1) + color]);
		else if ((i == 0 && j > 0) || (i < srcRows - 1 && j == srcCols - 1))
		max = dest_data[3 * ((i + 1) * srcCols + j - 1) + color];
		else if ((i > 0 && j == 0) || (i == srcRows - 1 && j < srcCols - 1))
		max = dest_data[3 * ((i - 1) * srcCols + j + 1) + color];
		}
		else if (angle[3 * (i * srcCols + j) + color] == (3 * M_PI) / 4){
		if (i > 0 && i < srcRows - 1 && j > 0 && j < srcCols - 1)
		max = std::max(dest_data[3 * (i * srcCols + j - 1) + color], dest_data[3 * (i * srcCols + j + 1) + color]);
		else if ((i == 0 && j < srcCols -1) || (i < srcRows -1 && j == 0))
		max = dest_data[3 * ((i + 1) * srcCols + j + 1) + color];
		else if ((i == srcRows -1 && j > 0) || (i > 0 && j == srcCols - 1))
		max = dest_data[3 * ((i - 1) * srcCols + j - 1) + color];
		}

		if (dest_data[3 * (i * srcCols + j) + color] < max)
		dest_data[3 * (i * srcCols + j) + color] = 0;
		}
		}
		}*/

		//delete[] angle;
		delete[] dest_data;
		nvtxRangePop();
		return output;
	}

	//	== float color_distance(float r1, float g1, float b1, float r2, float g2, float b2) ==
	//		float r1, g1, b1: color channel values of first point
	//		float r2, g2, b2: color channel values of second point
	//	Helper for kMeans that determines "Euclidean color distance"
	float color_distance(float r1, float g1, float b1, float r2, float g2, float b2){
		float val = 0;
		val += (b2 - b1) * (b2 - b1);
		val += (g2 - g1) * (g2 - g1);
		val += (r2 - r1) * (r2 - r1);
		return std::sqrtf(val);
	}

	//	== cv::Mat kMeans(cv::Mat image, int k_means) ==
	//		cv::Mat image: input image
	//		int k_means: number of color groups to create
	//	Uses random selections and averaging to assign all pixels into a small number of groups (determined by k_means argument)
	//	that best represent all colors in the image, then returns an image with colors based on that assignment
	//	kMeans is usually random and non-deterministic.  Normally it will work until it finds an assignment which is the best
	//	fit for all pixels, and starts by randomly picking single pixels for each color group.
	//	This implementation is hard coded to run the most important calculations 200 times, and is pre-seeded at the start of every call.

	cv::Mat kMeans(cv::Mat image, int k_means){
		get_nvtAttrib("kMeans CPU", 0xFF880000);

		get_nvtAttrib("Setup", 0xFF000088);
		srand(2000);
		int srcRows = image.rows;
		int srcCols = image.cols;
		cv::Mat output(srcRows, srcCols, image.type());
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_data = (uchar*)output.datastart;

		//	For now, limited to max of 256 groups
		//	More than this would be excessive
		if (k_means > 256){
			printf("Error: Max number of groups exceeded (256)\n");
			exit(-1);
		}

		//	Data structures that hold information on color groups
		//	k_colors holds the values of color channels for each group
		//	k_index holds the group number for every pixel in the input image
		//	k_count holds the total number of pixels assigned to each color group
		float* k_colors = new float[3 * k_means];
		uchar* k_index = new uchar[srcRows * srcCols];
		int* k_count = new int[k_means];

		//	Choosing random pixels to start groups with
		for (int pix = 0; pix < k_means; pix++){
			int i = rand() % srcRows;
			int j = rand() % srcCols;

			for (int color = 0; color < 3; color++){
				k_colors[3 * pix + color] = src_data[3 * (i * srcCols + j) + color];
			}

		}

		//	When this is true at the end of the next while loop, no pixels have changed their group assignment
		//	and the algorithm is complete.
		bool convergence = false;

		//	Initializing k_index to all zeroes to start
		for (int k = 0; k < srcRows * srcCols; k++){
			k_index[k] = 0;
		}

		int count = 0;
		nvtxRangePop();

		//	Main work loop.  First, the "color distances" of each pixel to the color of each main group, and re-assigns groups if necessary.
		//	Then, if any pixels have been re-assigned, the average colors of all the groups are re-calculated based on the new assignments
		get_nvtAttrib("Convergence Loop", 0xFF888888);
		while (!convergence){
			//convergence = true;  //UNDO
			for (int k = 0; k < k_means; k++){
				k_count[k] = 0;
			}
			for (int i = 0; i < srcRows; i++){
				for (int j = 0; j < srcCols; j++){

					//	Data order of opencv Mat is BGR, backwards of RGB
					float b2 = src_data[3 * (i * srcCols + j)];
					float g2 = src_data[3 * (i * srcCols + j) + 1];
					float r2 = src_data[3 * (i * srcCols + j) + 2];
					float min_dist = std::numeric_limits<float>::max();
					uchar new_index;

					for (int group = 0; group < k_means; group++){
						float b1 = k_colors[3 * group];
						float g1 = k_colors[3 * group + 1];
						float r1 = k_colors[3 * group + 2];

						float dist = color_distance(r1, g1, b1, r2, g2, b2);	//	Distance from the color of this pixel to each group
						if (dist < min_dist){
							min_dist = dist;
							
							new_index = group;
						}
					}
					if (k_index[i * srcCols + j] != new_index){
						k_index[i * srcCols + j] = new_index;
						
						convergence = false;	//	If a pixel has changed its group, the algorithm has not converged
					}
					k_count[new_index] += 1;
				}
			}

			if (count == 200){			// <-- Temporary hard coding of 200 runs for testing purposes
				break;					// <-- Comment out these lines for original algorithm
			}							// <--
			count++;
			//if (convergence)			// <-- Uncomment these two lines for the original algorithm
			//	break;					// <--

			//	Re-setting the color values to zero, so we can accumulate averages in one pass
			for (int k = 0; k < 3 * k_means; k++){
				k_colors[k] = 0;
			}

			//	Take color values of pixels in original image (float to avoid truncation), divide by the number of
			//	pixels assigned to this pixel's designated group, add to total
			for (int i = 0; i < srcRows; i++){
				for (int j = 0; j < srcCols; j++){
					int group = k_index[i * srcCols + j];
					int group_count = k_count[group];
					for (int color = 0; color < 3; color++){
						float src_val = src_data[3 * (i * srcCols + j) + color];
						float val = src_val / group_count;
						k_colors[3 * group + color] += val;
					}

				}
			}
		}//	End of main while loop
		nvtxRangePop();
		//	Writing to output image using the last discoved group color values and the group assignment of every pixel in
		//	the input image
		get_nvtAttrib("Output Adjust", 0xFFFF0000);
		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				int group = k_index[i * srcCols + j];

				for (int color = 0; color < 3; color++){
					dest_data[3 * (i * srcCols + j) + color] = (uchar)k_colors[3 * group + color];
				}

			}
		}
		nvtxRangePop();
		
		//Avoid memory leaks
		delete[] k_colors;
		delete[] k_index;
		delete[] k_count;

		nvtxRangePop();
		return output;
	}

	cv::Mat kMeansFixed(cv::Mat image, int k_means){
		get_nvtAttrib("kMeansFixed CPU", 0xFF880000);

		get_nvtAttrib("Setup", 0xFF000088);
		srand(2000);
		int srcRows = image.rows;
		int srcCols = image.cols;
		cv::Mat output(srcRows, srcCols, image.type());
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_data = (uchar*)output.datastart;

		//	For now, limited to max of 256 groups
		//	More than this would be excessive
		if (k_means > 256){
			printf("Error: Max number of groups exceeded (256)\n");
			exit(-1);
		}

		//	Data structures that hold information on color groups
		//	k_colors holds the values of color channels for each group
		//	k_index holds the group number for every pixel in the input image
		//	k_count holds the total number of pixels assigned to each color group
		//float* k_colors = new float[3 * k_means];
		int* k_colors = new int[3 * k_means];
		uchar* k_index = new uchar[srcRows * srcCols];
		int* k_count = new int[k_means];

		//	Choosing random pixels to start groups with
		for (int pix = 0; pix < k_means; pix++){
			int i = rand() % srcRows;
			int j = rand() % srcCols;

			for (int color = 0; color < 3; color++){
				k_colors[3 * pix + color] = src_data[3 * (i * srcCols + j) + color] * ONE;
			}

		}

		//	When this is true at the end of the next while loop, no pixels have changed their group assignment
		//	and the algorithm is complete.
		bool convergence = false;

		//	Initializing k_index to all zeroes to start
		for (int k = 0; k < srcRows * srcCols; k++){
			k_index[k] = 0;
		}

		int count = 0;
		nvtxRangePop();

		//	Main work loop.  First, the "color distances" of each pixel to the color of each main group, and re-assigns groups if necessary.
		//	Then, if any pixels have been re-assigned, the average colors of all the groups are re-calculated based on the new assignments
		get_nvtAttrib("Convergence Loop", 0xFF888888);
		while (!convergence){
			//convergence = true;  //UNDO
			for (int k = 0; k < k_means; k++){
				k_count[k] = 0;
			}
			for (int i = 0; i < srcRows; i++){
				for (int j = 0; j < srcCols; j++){

					//	Data order of opencv Mat is BGR, backwards of RGB
					float b2 = src_data[3 * (i * srcCols + j)];
					float g2 = src_data[3 * (i * srcCols + j) + 1];
					float r2 = src_data[3 * (i * srcCols + j) + 2];
					float min_dist = std::numeric_limits<float>::max();
					uchar new_index;

					for (int group = 0; group < k_means; group++){
						float b1 = (float)k_colors[3 * group] / ONE;
						float g1 = (float)k_colors[3 * group + 1] / ONE;
						float r1 = (float)k_colors[3 * group + 2] / ONE;

						float dist = color_distance(r1, g1, b1, r2, g2, b2);	//	Distance from the color of this pixel to each group
						if (dist < min_dist){
							min_dist = dist;

							new_index = group;
						}
					}
					if (k_index[i * srcCols + j] != new_index){
						k_index[i * srcCols + j] = new_index;

						convergence = false;	//	If a pixel has changed its group, the algorithm has not converged
					}
					k_count[new_index] += 1;
				}
			}

			if (count == 200){			// <-- Temporary hard coding of 200 runs for testing purposes
				break;					// <-- Comment out these lines for original algorithm
			}							// <--
			count++;
			//if (convergence)			// <-- Uncomment these two lines for the original algorithm
			//	break;					// <--

			//	Re-setting the color values to zero, so we can accumulate averages in one pass
			for (int k = 0; k < 3 * k_means; k++){
				k_colors[k] = 0;
			}

			//	Take color values of pixels in original image (float to avoid truncation), divide by the number of
			//	pixels assigned to this pixel's designated group, add to total
			for (int i = 0; i < srcRows; i++){
				for (int j = 0; j < srcCols; j++){
					int group = k_index[i * srcCols + j];
					int group_count = k_count[group];
					for (int color = 0; color < 3; color++){
						int src_val = src_data[3 * (i * srcCols + j) + color] * ONE;
						int val = src_val / (group_count);
						k_colors[3 * group + color] += val;
					}

				}
			}
		}//	End of main while loop
		nvtxRangePop();
		//	Writing to output image using the last discoved group color values and the group assignment of every pixel in
		//	the input image
		get_nvtAttrib("Output Adjust", 0xFFFF0000);
		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				int group = k_index[i * srcCols + j];

				for (int color = 0; color < 3; color++){
					dest_data[3 * (i * srcCols + j) + color] = (uchar)(k_colors[3 * group + color] / ONE);
				}

			}
		}
		nvtxRangePop();

		//Avoid memory leaks
		delete[] k_colors;
		delete[] k_index;
		delete[] k_count;

		nvtxRangePop();
		return output;
	}

	//	== cv::Mat gaussianPyramid(cv::Mat image, uchar levels, float scale) ==
	//		cv::Mat image: input image
	//		uchar levels: number of levels to build
	//		float scale: factor of which each level is scaled down (must be < 0.5)
	//	Input image is first blurred using a Gaussian filter, then is subsampled using linear-interpolation resizing.
	//	The number of times this is repeated is determined by levels, and the resizing scale is detemined by scale

	cv::Mat gaussianPyramid(cv::Mat image, uchar levels, float scale){

		//	Scale is restricted to less than 0.5, so the output image only needs to be at most
		//	twice the size of the input image
		if (scale > 0.5){
			printf("Error: Scale > 0.5\n");
			exit(-1);
		}

		int srcRows = image.rows;
		int srcCols = image.cols;
		cv::Mat output(srcRows + (srcRows * scale + 1), srcCols, image.type());
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_data = (uchar*)output.datastart;

		//	Writing original image to output image
		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				for (int color = 0; color < 3; color++){
					int idx = 3 * (i * srcCols + j) + color;
					dest_data[idx] = src_data[idx];
				}
			}
		}

		//	Pointer arithmetic to ignore the upper part of the image
		dest_data += srcRows * srcCols * 3;

		int newRows = srcRows * scale;
		int newCols = srcCols * scale;
		float newScale = scale;
		int offset = 0;

		//	Main work loop, the last image used is blurred, resized, then written to the output image
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

	//	== Class keypoint ==
	//		int idx: Row coordinate
	//		int idy: Column coordinate
	//		int oct: Octave
	//		float angle: Orientation
	//		float scale: Scale of blur (not properly implemented yet)
	//	Class used to store data about keypoints in SIFT algorithm.  See SIFT papers for more information
	

	float eucDistance(float x1, float y1, float x2, float y2){
		return sqrt(pow(x2-x1,2) + pow(y2-y1,2));
	}

	//	== cv::Mat fGaussTest(cv::Mat image) ==
	//		cv::Mat image: Input image
	//	Function used to test the float version of Gaussian filter.  Ignore for now
	/*cv::Mat fGaussTest(cv::Mat image){
		int srcRows = image.rows;
		int srcCols = image.cols;
		cv::Mat output(srcRows, srcCols, image.type());

		uchar* src_data = (uchar*)image.datastart;
		uchar* fin_data = (uchar*)output.datastart;

		float* test = new float[3 * srcRows * srcCols];

		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				for (int color = 0; color < 3; color++){
					test[3 * (i * srcCols + j) + color] = (float)src_data[3 * (i * srcCols + j) + color];
				}
			}
		}

		float* dest_data = fGaussianFilter(test, 2.0, srcRows, srcCols);

		int ind = 0;
		while (fin_data != output.dataend){
			*fin_data = (uchar)dest_data[ind];
			ind += 1;
			fin_data += 1;
		}

		/*for (int i = 0; i < srcRows; i++){
		for (int j = 0; j < srcCols; j++){
		for (int color = 0; color < 3; color++){

		}
		}
		}

		return output;
	}*/

	cv::Mat depthFromStereo(cv::Mat image_l, cv::Mat image_r, float fl, float fr, float baseline){
		image_l = frgb2Gray(image_l);
		image_r = frgb2Gray(image_r);
		int window = 20;
		float bound = window / 2;
		int srcRows = image_l.rows, srcCols = image_l.cols;

		//cv::Mat output(srcRows, srcCols, image_l.type());
		cv::Mat output = cv::Mat::zeros(srcRows, srcCols, CV_32FC1);
		float* left_data = (float*)image_l.datastart;
		float* right_data = (float*)image_r.datastart;
		float* out_data = (float*)output.datastart;

		float cx_l = 0, cx_r = 0, cy_l = 0, cy_r = 0;
		fl = 4412.147; fr = 4412.147;
		baseline = 144.174;

		cx_l = 1283.781; cx_r = 1406.974;
		cy_l = 994.694; cy_r = 994.694;

		float z_min = 1000000000.0;
		float z_max = 0.0;

		float doffs = cx_r - cx_l;

		for (int li = bound; li < srcRows - bound; li++){

			for (int lj = bound; lj < srcCols - bound; lj++){

				int d = std::min(lj - (int)bound, 250), d_min = 0;
				float min_cost = 1000000000;

				for (; d > (lj - srcCols - bound); d-= 1){
					int rj = lj - d;

					float t_cost = 0.0;
					for (int u = li - bound; u < li + bound; u++){
						for (int v = lj - bound; v < lj + bound; v++){
							//t_cost += pow(image_l.at<float>(u, v) - image_r.at<float>(u, v - d), 2);
							t_cost += pow(left_data[(u * srcCols) + v] - right_data[(u * srcCols) + (v - d)], 2);
						}
					}

					if (t_cost <= min_cost){
						min_cost = t_cost;
						d_min = d;
					}
				}

				float z = (fl * baseline) / ((float)abs(d_min) + doffs);
				//output.at<float>(li, lj) = z;
				out_data[li * srcCols + lj] = z;

				if (z < z_min){
					z_min = z;
				}
				else if (z > z_max && z < 1000000000.0){
					z_max = std::min(z, (float)1000000000.0);
				}
				//printf("Z: %f\n", z);
			}

		}

		printf("Z Min: %f; Z Max: %f\n", z_min, z_max);

		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				float temp_z = out_data[i * srcCols + j];
				if (temp_z > z_max) temp_z = z_max;

				temp_z = 255.0 - ((temp_z - z_min) * 255.0 / (z_max - z_min));
				out_data[i * srcCols + j] = temp_z;
				//printf("temp_z: %f\n", temp_z);
			}
		}

		return output;
	}

	//	== cv::Mat mySift(cv::Mat image) ==
	//		cv::Mat image: Input image
	//	Applies Scale Invariant Feature Transform to the input image in order to identify features that can be used for 
	//	object recognition.  This algorithm currently builds the scale space and uses the difference-of-gaussians to identify potential
	//	keypoints.  Not yet implemented are keypoint filtering, orientation and magnitude calculations, and keypoint storage or analysis.
	cv::Mat mySift(cv::Mat original){

		nvtxEventAttributes_t eventAttrib = get_nvtAttrib("mySift CPU", 0xFF880000);

		std::vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(9);
		std::string debug_Path = "D://School//Summer 2016//Research//mySift//";
		std::string img_name = "audrey";
		std::string ftype = ".png";

		int printoff = 0;	//	Debug, used to print to console the values of all keypoints found by this function
		int full_dog = 0;	//	Set this to 1 to change the output image to a full representation of the difference-of-gaussians and scale space location of each keypoint
		int mark_dim = 1;	//	Determines size of circles in the output image.  Recommend setting to 10 or higher if full_dog is set to 1

		//	The first step is to scale up the input image by a factor of 2 in both dimensions, then apply a gaussian blur with sigma = 1.6
		//	Scaling up the input provides more keypoints and approximates a blur with sigma = 1.0, assuming the original image is roughly sigma = 0.5 which is the threshold for noise
		//	Sigma = 1.6 was experimentally determined by Lowe to give the best results.  Refer to the 2004 SIFT paper, pages 9 and 10 for discussion
		float sigma = 1.6;
		//cv::Mat original = image;

		//cv::Mat image = linearResize(original, original.rows * 2, original.cols * 2);
		cv::Mat image = frgb2Gray(original);
		//image = fgaussianFilter(image, 1.0);
		image = flinearResize(image, image.rows * 2, image.cols * 2);
		cv::Mat temp_image;
		//cv::copyMakeBorder(image, temp_image, 1, 1, 1, 1, cv::BORDER_REPLICATE);
		temp_image = image;

		//image = linearResize(image, image.rows * 2, image.cols * 2);
		//image = gaussianFilter(image, sigma);

		uchar scales = 3;	//	Lowe found 3 to be best
		uchar octaves = 1;	//	4 or 5 octaves is the standard
		int region = 4;
		int srcRows = image.rows;
		int srcCols = image.cols;

		//	In one octave, a blurred image is related to the previous image by a factor of 2^(1/scales), see SIFT 2004 page 7
		float k = pow(2.0, (1.0 / (float)scales));
		//float k = pow(2.0, 0.5); //K_VAL HERE
		int s = scales + 3;
		int slevel = s - 1;

		int destRows, destCols;

		//	Output cv::Mat is different based on variables set above
		if (full_dog == 1){
			destRows = 2 * srcRows;
			destCols = (s)* srcCols;
		}
		else {
			destRows = srcRows / 2;
			destCols = srcCols / 2;
		}

		//cv::Mat output(destRows, destCols, original.type());
		cv::Mat output = cv::Mat::zeros(destRows, destCols, original.type());

		float* src_data = (float*)image.datastart;
		uchar* dest_data = (uchar*)output.datastart;

		int curRows = srcRows;
		int curCols = srcCols;

		std::vector<keypoint> keys;
		std::vector<std::vector<cv::Mat>> dog_oct;	//	Not currently used, meant to store DoG data long term
		std::vector <std::vector<cv::Mat>> blur_oct;

		int key_count = 0;
		int scale = 1;
		int key_index = 0;
		int bogeys = 0;

		//float additive = 0.0;

		//printf("Fish\n");
		int gauss_exp = 0;

		//	Main work loop, builds each octave, but also identifies possible keypoints before moving to next octave
		for (int oct = 0; oct < octaves; oct++){
			printf("Oct %d\n", oct);
			int octa = oct + 1;
			get_nvtAttrib("Octave " + std::to_string(oct), 0xFF888888);
			std::vector<cv::Mat> blur_img;
			std::vector<cv::Mat> dog_img;
			curRows = temp_image.rows;
			curCols = temp_image.cols;

			cv::Mat current = temp_image;
			//if (oct == 0){ current = fGaussianFilterSep(temp_image, sigma); }
			blur_img.push_back(current);

			get_nvtAttrib("Scale Space", 0xFF000088);
			for (int step = 1; step < s; step++){
				printf("  Step %d: ", step);
				float temp_scale = sigma * pow(pow(sqrt(2.0), (1.0 / slevel)), oct * slevel + step);
				printf(" %f\n", temp_scale);
				//	Applies blur of strength k to previous image, until sigma is double that of the first image in octave
				current = temp_image;
				//if (oct == 0 && step == 1){ current = fGaussianFilterSep(temp_image, sigma); }
				//cv::Mat next = cv::Mat::zeros(curRows, curCols, CV_32FC1);// = fGaussianFilterSep(current, pow(k, gauss_exp) * sigma);
				cv::Mat next = fGaussianFilterSep(temp_image, temp_scale);
				cv::Mat dog = cv::Mat::zeros(curRows,curCols,CV_32FC1);
				blur_img.push_back(next);

				float* curr_data = (float*)current.data;
				float* next_data = (float*)next.data;
				float* dog_data = (float*)dog.data;
				for (int i = 0; i < curRows ; i++){
					for (int j = 0; j < curCols; j++){
						int idx = (i * curCols) + j;

						float val = next_data[idx] - curr_data[idx];// + 127;// Data is stored as gray to allow for presentable data.  Adding 127 reduces risk of wrap-around error (range is 0 - 255).
						
						dog_data[idx] = val;
					}
				}
				//cv::imwrite("D://School//Summer 2016//Research//mySift//parrot_test_" + std::to_string((oct + 1) * 10 + (step)) + "_c.png", dog, compression_params);
				dog_img.push_back(dog);
				//current = next;
				if (step == s - 1){

				}
				gauss_exp++;
			}
			nvtxRangePop();


			
			//	Keypoint calculation.  Refer to SIFT papers for more information
			get_nvtAttrib("Keypoints", 0xFF000088);
			printf("Keypoints\n");
			for (int step = 1; step < s - 2; step++){
				int temp_exp = gauss_exp - s + (step);
				//printf("temp_exp: %d\n", temp_exp);
				float temp_scale = ((pow(k, temp_exp) * sigma) - (pow(k, temp_exp - 1) * sigma)) / 2.0 + (pow(k, temp_exp - 1.0) * sigma);

				cv::Mat prev = dog_img[step - 1];
				cv::Mat curr = dog_img[step + 0];
				cv::Mat next = dog_img[step + 1];
				float* prev_data = (float*)dog_img[step - 1].datastart;
				float* curr_data = (float*)dog_img[step + 0].datastart;
				float* next_data = (float*)dog_img[step + 1].datastart;
				

				for (int i = 1; i < curRows - 1; i++){
					for (int j = 1; j < curCols - 1; j++){
						//int idx = 3 * (i * curCols + j);
						int idx = (i * curCols + j);
						float val = curr_data[idx];
						//float val = curr.ptr<float>(i)[j];
						//float val = curr.at<float>(i,j);

						float val_c = 0;
						float val_min = val;
						float val_max = val;
						int counter = 0;

						/*for (int k = -1; k <= 1; k++){
							for (int l = -1; l <= 1; l++){
								val_c = prev_data[idx + (k * curCols + l)];
								if (val_c < val_min) val_min = val_c;
								if (val_c > val_max) val_max = val_c;

								val_c = curr_data[idx + (k * curCols + l)];
								if (val_c < val_min) val_min = val_c;
								if (val_c > val_max) val_max = val_c;

								val_c = next_data[idx + (k * curCols + l)];
								if (val_c < val_min) val_min = val_c;
								if (val_c > val_max) val_max = val_c;
							}
						}

						//printf("I: %d, J: %d, V: %f, Vmin: %f, Vmax: %f, ", i, j, val, val_min, val_max);

						if (val <= val_min || val >= val_max) {
							//printf("Yes============================================================\n");
							keypoint newKey(i, j, oct, 0, step);
							//printf("I: %d, J: %d, V: %f\n", i, j, val);
							newKey.scale = temp_scale;
							newKey.posy = ((float)i) / curRows;
							newKey.posx = ((float)j) / curCols;
							keys.push_back(newKey);
							key_count++;
							counter++;
							if (counter >= 5){
								//printf("Hold+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
								//getchar();
								counter = 0;
							}
						}*/

						for (int k = -1; k <= 1; k++){
							for (int l = -1; l <= 1; l++){
								val_c = prev_data[idx + (k * curCols) + l] - val;
								if (val_c > 0) counter += 1;
								else if (val_c < 0) counter -= 1;

								val_c = curr_data[idx + (k * curCols) + l] - val;
								if (val_c > 0) counter += 1;
								else if (val_c < 0) counter -= 1;

								val_c = next_data[idx + (k * curCols) + l] - val;
								if (val_c > 0) counter += 1;
								else if (val_c < 0) counter -= 1;
							}
						}

						//printf("I: %d, J: %d, V: %f, Vmin: %f, Vmax: %f, ", i, j, val, val_min, val_max);

						if (abs(counter) == 26){
							//printf("Yes============================================================\n");
							keypoint newKey(i, j, oct, 0, step);
							//printf("I: %d, J: %d, V: %f\n", i, j, val);
							newKey.scale = temp_scale;
							newKey.posy = ((float)i) / curRows;
							newKey.posx = ((float)j) / curCols;
							keys.push_back(newKey);
							key_count++;
						}
						else{
							//printf("No\n");
						}

					}
				}
			}
			nvtxRangePop();

			int offset = 0;
			//curRows = srcRows;
			//curCols = srcCols;

			//	Build full Difference-of-Gaussian image for presentation.  Not part of true algorithm, but good for discussions.
			if (full_dog == 1){

				for (int i = 0; i < curRows; i++){
					for (int j = 0; j < curCols; j++){
						for (int color = 0; color < 1; color++){
							int idx_i = (i * curCols + j) + color;
							int idx_o = 3 * (i * destCols + (j + offset)) + 1;

							//dest_data[idx_o] = src_data[idx_i];
						}
					}
				}

				offset += curCols;

				for (int step = 0; step < s - 1; step++){

					cv::Mat dog = dog_img[step];
					cv::Mat blr = blur_img[step];
					float* dog_data = (float*)dog.datastart;
					float* blr_data = (float*)blr.datastart;

					for (int i = 0; i < curRows; i++){
						for (int j = 0; j < curCols; j++){
							int idx_i = (i * curCols + j);
							int idx_o = 3 * (i * destCols + (j + offset));
							
							for (int color = 0; color < 3; color++){
								dest_data[idx_o + color] = dog_data[idx_i] + 0;
							}
						}
					}
					offset += curCols;

				}
				dest_data += 3 * curRows * destCols;
			}
			
			printf("  tRows: %d, tCols: %d\n", temp_image.rows, temp_image.cols);
			image = current;// (cv::Rect(1, 1, current.cols - 2, current.rows - 2));
			//image = myTrim(current, 1, 1, 1, 1);
			image = fdirectResize(image, image.rows / 2, image.cols / 2);
			//cv::copyMakeBorder(image, temp_image, 1, 1, 1, 1, cv::BORDER_REPLICATE);
			temp_image = image;

			src_data = (float*)image.datastart;
			//printf("Fish3\n");
			dog_oct.push_back(dog_img);
			blur_oct.push_back(blur_img);
			gauss_exp -= 2;
			nvtxRangePop();
		}	//	End of main work loop
		



		printf("Keys Size: %d\n", key_count);
		
		//mySiftWriteKeyFile(keys);

		//return dog_oct[0][0];

		get_nvtAttrib("Or_Mag calc", 0xFF888888);
		//Pre-compute orientation and magnitude
		std::vector < std::vector<float*> > or_mag_oct;
		curRows = srcRows;
		curCols = srcCols;
		for (int oct = 0; oct < octaves; oct++){
			//std::vector<cv::Mat> current_oct = blur_oct[oct];
			std::vector<float*> or_mag_current;
			for (int step = 0; step < s; step++){
				cv::Mat& current = blur_oct[oct][step];
				//uchar* cur_data = (uchar*)current.datastart;
				float* cur_data = (float*)current.datastart;

				//printf("Mark 4\n");
				//printf("curRows: %d, curCols: %d\n",curRows,curCols);
				float* or_mag = new float[2 * curRows * curCols];
				for (int i = 0; i < curRows; i++){
					//printf("Mark 5\n");
					int pi = 1, mi = 1;
					if (i == 0) { mi = 0; }
					else if (i == curRows - 1) { pi = 0; }

					for (int j = 0; j < curCols; j++){
						
						int pj = 1, mj = 1;
						if (j == 0) { mj = 0; }
						else if (j == curCols - 1) { pj = 0; }

						float val = sqrt(pow(cur_data[((i + pi) * curCols + j)] - cur_data[((i - mi) * curCols + j)], 2) + pow(cur_data[(i * curCols + (j + pj))] - cur_data[(i * curCols + (j - mj))], 2));
						or_mag[2 * (i * curCols + j) + 1] = val;
						float val1 = cur_data[((i + pi) * curCols + j)] - cur_data[((i - mi) * curCols + j)];
						float val2 = cur_data[(i * curCols + (j + pj))] - cur_data[(i * curCols + (j - mj))];

						val = atan2(val2, val1);
						if (val < 0){
							val = (2 * M_PI) + val;
						}
						else if (val > 2 * M_PI){
							val = val - (2 * M_PI);
						}
						//printf("Or: %f\n", val);
						//if (val == 0 && val2 >= 0)
						//	val = M_PI / 2;
						//else if (val == 0 && val2 < 0)
						//	val = (3 * M_PI) / 2;
						//else
						//	val = atan2(val2 / val);
						//val = atan((cur_data[3 * (i * curCols + (j + 1))] - cur_data[3 * (i * curCols + (j - 1))]) / (cur_data[3 * ((i + 1) * curCols + j)] - cur_data[3 * ((i - 1) * curCols + j)]));
						or_mag[2 * (i * curCols + j)] = val;
					}
				}
				or_mag_current.push_back(or_mag);
			}

			or_mag_oct.push_back(or_mag_current);
			curRows = curRows / 2;
			curCols = curCols / 2;

		}
		nvtxRangePop();

		get_nvtAttrib("Key Culling " + std::to_string(keys.size()), 0xFF000088);
		if (debug_statements) printf("Keys Size: %d\n", keys.size());

		dest_data = (uchar*)output.datastart;

		mySiftKeyCull(keys);
		printf("Fish\n");

		//Edge Responses goes here
		mySiftEdgeResponsesNew(dog_oct, keys);


		//mySiftEdgeResponses(dog_oct, keys);
		printf("Edge\n");
		printf("Size before: %d\n", keys.size());
		mySiftKeyCull(keys);
		printf("Size after: %d\n", keys.size());

		int unfiltered = 0;
		std::vector<keypoint>::iterator iter;
		/*for (iter = keys.begin(); iter != keys.end();){
			if ((*iter).filtered){
				//iter = keys.erase(iter);
				iter++;
			}
			else{
				unfiltered++;
				iter++;
			}
		}*/

		key_count = keys.size();

		//mySiftKeyCull(keys);
		nvtxRangePop();

		get_nvtAttrib("Or Assign", 0xFF888888);
		printf("Orientation Assignment\n");

		//Orientation assignment
		//int region = 4;

		int W = 2 * region + 1;
		double* gKernel = new double[W * W];
		createFilter(gKernel, 1.5, region);

		key_count = keys.size();

		//int key_index = 0;

		//keypoint& key_test = keys[4586];
		//key_test;

		for (int key_index = 0; key_index < key_count; key_index++){
			//keypoint& key_now = keys[key_index];
			keypoint key_now = keys[key_index];
			if (key_now.filtered == true) continue;

			int idx = key_now.idx, idy = key_now.idy, oct = key_now.oct, kindex = (int)key_now.index;
			float scale = key_now.scale;

			int curRows = srcRows / exp2f(oct), curCols = srcCols / exp2f(oct);
			float* or_mag = or_mag_oct[oct][kindex];

			if (idx < region || idx > curRows - region - 1 || idy < region || idy > curCols - region - 1){
				//key_now.filtered = true;
				keys[key_index].filtered = true;
				continue;
			}

			float histo[36] = { 0 };
			for (int i = -region; i < region; i++){
				for (int j = -region; j < region; j++){
					//if (eucDistance(idx,idy,idx + i, idy + j) > region) continue;

					int id_or_mag = 2 * ((idx + i) * curCols + (idy + j));
					int id_gKernel = (i + region) * W + (j + region);

					int bin = (int)((180.0 / (M_PI * 10.0)) * or_mag[id_or_mag]);
					histo[bin] += gKernel[id_gKernel] * or_mag[id_or_mag + 1];
				}
			}

			int max_bin = 0;
			float max_hist = 0.0;

			for (int bin = 0; bin < 36; bin++){
				if (histo[bin] > max_hist){
					max_bin = bin;
					max_hist = histo[bin];
				}
			}

			if (!(max_hist > 0.0)){
				keys[key_index].filtered = true;
				continue;
			}

			//float peaks[36] = { 0 };

			for (int i = 0; i < 36; i++){
				int left = i - 1, right = i + 1;
				if (i == 0){
					left = 35;
				}
				else if (i == 35){
					right = 0;
				}

				if (max_hist != 0.0 && i != max_bin && histo[i] > histo[left] && histo[i] > histo[right] && histo[i] >= 0.8 * max_hist){
					//peaks[i] = histo[i];

					float orientation = mySiftVertParabola((i - 1) * 10 + 5, histo[left], i * 10 + 5, histo[i], (i + 1) * 10 + 5, histo[right]);
					//float orientation = i * ((M_PI * 10.0) / 180.0);
					keypoint newKey(idx, idy, oct, orientation, kindex);
					keys.push_back(newKey);
				}
				else{
					//peaks[i] = -1;
				}
			}

			int max_left = max_bin - 1, max_right = max_bin + 1;
			if (max_bin == 0) max_left = 35;
			else if (max_bin == 35) max_right = 0;

			//printf("Fish: left %d, bin %d, right %d\n", max_left, max_bin, max_right);
			//key_now.angle = mySiftVertParabola(max_left * 10 + 5, histo[max_left], max_bin * 10 + 5, histo[max_bin], max_right * 10 + 5, histo[max_right]);
			//printf("Stop: %f\n", key_now.angle);
			//key_now.angle = max_bin * ((M_PI * 10.0) / 180.0);
			//keys[key_index].angle = max_bin * ((M_PI * 10.0) / 180.0);
			float angle = mySiftVertParabola((max_bin - 1) * 10 + 5, histo[max_left], max_bin * 10 + 5, histo[max_bin], (max_bin + 1) * 10 + 5, histo[max_right]);
			keys[key_index].angle = angle;
			//delete[] peaks;
		}

		//getchar();
		delete[] gKernel;
		nvtxRangePop();

		if(debug_statements) printf("Desc\n");

		mySiftDescriptors(keys, blur_oct, or_mag_oct, unfiltered);

		for (int oct = 0; oct < octaves; oct++){
			for (int step = 0; step < s; step++){
				delete[] or_mag_oct[oct][step];
			}
		}
		
		//return dog_oct[0][0];
		printf("Size before: %d\n", keys.size());
		mySiftKeyCull(keys);
		printf("Size after: %d\n", keys.size());

		get_nvtAttrib("KD Tree Build", 0xFF888888);
		//kd_node fish = mySiftKDHelp(keys);
		nvtxRangePop();
		//nvtxRangePop();
		//mySiftWriteKeyFile(keys);
		//return original;

		//int unfiltered = 0;
		/*key_index = 0;
		//	Optional print out of all keypoint data
		while (key_index < key_count){
			keypoint& key_now = keys[key_index];
			if (key_now.filtered == true){
				key_index++;
				continue;
			}
			if (printoff == 1){
				printf("(%d,%d) oct: %d, scale %f\n", key_now.idx, key_now.idy, key_now.oct, key_now.index);
			}
			unfiltered += 1;
			key_index++;
		}*/

		/*std::vector<keypoint>::iterator iter;
		for (iter = keys.begin(); iter != keys.end(); ){
			if ((*iter).filtered){
				iter = keys.erase(iter);
			}
			else{
				unfiltered++;
				iter++;
			}
		}

		key_count = keys.size();

		printf("Unfiltered: %d\n", unfiltered);*/
		//printf("Distance: %d\n", std::distance(keys.begin(),keys.end()));

		//mySiftWriteKeyFile(keys);
		//kd_node fish = mySiftKDHelp(keys);


		/*printf("\n");
		std::vector<keypoint> temp_keys;
		std::string file_name = "D://School//Summer 2016//Research//mySift//keys_in.txt";

		if (!mySiftReadKeyFile(temp_keys, file_name)){
			printf("Error reading given file: %s\n", file_name);
			return output;
		}*/

		//kd_node* kd_final = mySiftKDSearch(fish,keys,temp_keys[0]);
		//kd_node* kd_final = mySiftKDIterSearch(&fish, keys, temp_keys[0]);
		
		key_count = keys.size();

		//	DoG keypoint indicator drawing
		if (full_dog == 1){
			key_index = 0;
			while (key_index < key_count){
				keypoint key_now = keys[key_index];
				int idx = key_now.idx, idy = key_now.idy;
				int oct = key_now.oct, kindex = (int)key_now.index;
				float mult = 0.5, rmult = 1.0;

				int row_off = 0;
				for (int ti = 0; ti < oct; ti++){
					row_off += rmult * srcRows;
					rmult *= mult;
				}

				int col_off = (kindex + 1) * rmult * srcCols;

				//cvCircle(dest_data, (idx + col_off, idy + row_off), mark_dim, (255, 0, 0), 2, 8, 0);
				cv::circle(output, cv::Point(idy + col_off, idx + row_off), mark_dim, cv::Scalar(0, 0, 255), 2, 8, 0);

				key_index++;
			}
		}
		//	Regular image keypoint indicator drawing
		else {
			output = original;
			key_index = 0;
			while (key_index < key_count){
				keypoint key_now = keys[key_index];
				if (key_now.filtered){
					key_index++;
					continue;
				}

				int idx = key_now.idy, idy = key_now.idx;
				int oct = key_now.oct, kindex = (int)key_now.index;
				//printf(" %d:%d ", oct, kindex);

				float mult = pow(2, oct - 1);

				//cv::circle(output, cv::Point(idx * mult, idy * mult), mark_dim, cv::Scalar(0, 0, 255), 2, 8, 0);

				//printf("Angle: %f\n",key_now.angle);
				//float size = 10 + (5 * mark_dim * ((10 * oct) * (k * (kindex -1))));
				float size = 10 + (5 * mark_dim * ((10 * oct)));
				int newi = (idx * mult) + (sin(key_now.angle) * size);
				int newj = (idy * mult) + (cos(key_now.angle) * size);

				//if (newi >= 0 && newi <= srcRows / 2 - 1 && newj >= 0 && newj <= srcCols / 2 - 1){
				//cv::arrowedLine(output, cv::Point(idx * mult, idy * mult), cv::Point(newi, newj), cv::Scalar(0, 0, 255), 1, 8, 0);

				cv::circle(output, cv::Point(idx * mult, idy * mult), mark_dim * (oct + 1), cv::Scalar(0, 0, 255), 1, 8, 0);
				//cv::circle(output, cv::Point(idx * mult, idy * mult), mark_dim, cv::Scalar(0, 0, 255), 1, 8, 0);
				//}


				key_index++;
			}
		}


		//printf("Keys: %d\n", key_count);
		nvtxRangePop();

		return output;
	}

	void mySiftEdgeResponsesNew(std::vector<std::vector<cv::Mat>>& dog_oct, std::vector<keypoint>& keys){
		std::vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(9);
		float data[4] = { -1, -1, 1, 1};
		cv::Mat der_y = cv::Mat(2, 2, CV_32FC1, data);
		//std::cout << der_y << std::endl;
		cv::Mat sder_y = myConv2(der_y, der_y, 1);
		//std::cout << sder_y << std::endl;
		printf("Edges New\n");
		float r = 10;
		
		for (int x = 0; x < dog_oct.size(); x++){
			//printf("  Oct %d\n", x);
			for (int y = 0; y < dog_oct[x].size(); y++){
				//printf("    Level %d\n", y);

				cv::Mat test = dog_oct[x][y];
				//cv::imwrite("D://School//Summer 2016//Research//mySift//parrot_test_" + std::to_string((x + 1) * 10 + (y)) + "_c.png", dog_oct[x][y], compression_params);
				
				cv::Mat temp = myElemMat(myElemScalar(myConv2(test, sder_y, 0), -1, -1), myConv2(test, der_y, 0), 0);

				dog_oct[x][y] = myElemMat(myElemScalar(myElemMat(temp, myConv2(myTranspose(test), der_y, 0), 0), 0.5, 0), test, 1);
				//dog_oct[x][y] = myElemMat( myElemScalar( myElemMat(temp, myConv2(test, myTranspose(der_y), 0), 0), 0.5, 0), test, 1);
				
				cv::imwrite("D://School//Summer 2016//Research//mySift//parrot_test_" + std::to_string((x + 1) * 10 + (y)) + "_zc.png", dog_oct[x][y], compression_params);
				int levels = dog_oct[x].size();

			}
		}

		for (int key_idx = 0; key_idx < keys.size(); key_idx++){
			keypoint& cur_key = keys[key_idx];
			int index = cur_key.index;
			int octave = cur_key.oct;
			int idx = cur_key.idx;
			int idy = cur_key.idy;

			if (abs(dog_oct[octave][index].at<float>(idx,idy)) < 7.65){
				cur_key.filtered = true;
				continue;
			}

			float Dxx = dog_oct[octave][index].at<float>(idx - 1, idy) + dog_oct[octave][index].at<float>(idx + 1, idy) - 2 * dog_oct[octave][index].at<float>(idx, idy);
			float Dyy = dog_oct[octave][index].at<float>(idx, idy - 1) + dog_oct[octave][index].at<float>(idx, idy + 1) - 2 * dog_oct[octave][index].at<float>(idx, idy);
			float Dxy = dog_oct[octave][index].at<float>(idx - 1, idy - 1) + dog_oct[octave][index].at<float>(idx + 1, idy + 1) - dog_oct[octave][index].at<float>(idx - 1, idy + 1) - dog_oct[octave][index].at<float>(idx + 1, idy - 1);
			float deter = Dxx * Dyy - Dxy * Dxy;
			if (deter <= 0){
				cur_key.filtered = true;
				continue;
			}
			//float R = (Dxx + Dyy) / deter;
			float R = ((Dxx + Dyy) * (Dxx + Dyy)) / deter;
			float R_thresh = ((r + 1) * (r + 1)) / r;
			if (R > R_thresh){
				cur_key.filtered = true;
			}

		}

		int fish;

		return;
	}

	void mySiftEdgeResponses(std::vector<std::vector<cv::Mat>>& dog_oct, std::vector<keypoint>& keys){
		nvtxEventAttributes_t eventAttrib = get_nvtAttrib("mySiftEdgeResponses", 0xFF0000FF);
		float r = 10;
		//float t = std::pow(r + 1, 2) / r;
		float t = ((r + 1) * (r + 1)) / r;

		int key_count = keys.size();

		int key_index = 0;

		while (key_index < key_count){
			
			keypoint& key_now = keys[key_index];
			//printf("Point %d (%d,%d)\n", key_index, key_now.idx, key_now.idy);
			std::vector<cv::Mat> neighbors;
			//cv::Mat& dog = dog_oct[key_now.oct][key_now.index - 1];
			//printf("Idx: %d, Idy: %d, Oct: %d, Scale: %d\n", key_now.idx, key_now.idy, key_now.oct, (int)key_now.index);
			//if (key_now.index == 0){
			//	printf("Key %d is scale 0\n", (int)key_now.index);
			//	key_index++;
			//	continue;
			//}
			
			neighbors.push_back(dog_oct[key_now.oct][(int)key_now.index - 1]);
			neighbors.push_back(dog_oct[key_now.oct][(int)key_now.index]);
			neighbors.push_back(dog_oct[key_now.oct][(int)key_now.index + 1]);
			//printf("Neighbors %d\n", key_index);
			cv::Mat foDer = mySift_foDer(neighbors, key_now.idx, key_now.idy);
			//printf("foDer %d\n", key_index);
			cv::Mat soDer = mySift_soDer(neighbors, key_now.idx, key_now.idy);
			//printf("soDer %d\n", key_index);

			const float dxx = soDer.at<float>(0, 0);
			const float dyy = soDer.at<float>(1, 1);
			const float dxy = soDer.at<float>(0, 1);
			float* tptr = (float*)foDer.datastart;
			printf(" %d:%d ", key_now.oct, key_now.index);

			if (key_now.index != 0){
				//printf("  Point %d: dx: %f; dy: %f; ds: %f  \n", key_now.index, tptr[0], tptr[1], tptr[2]);
				//getchar();
			}

			//cv::Mat neg_soDer = soDer * -1;

			float det = 0;	//Find Det(soDer)
			det += soDer.at<float>(0, 0) * ((soDer.at<float>(1, 1) * soDer.at<float>(2, 2)) - (soDer.at<float>(2, 1) * soDer.at<float>(1, 2)));
			det -= soDer.at<float>(0, 1) * ((soDer.at<float>(1, 0) * soDer.at<float>(2, 2)) - (soDer.at<float>(2, 0) * soDer.at<float>(1, 2)));
			det += soDer.at<float>(0, 2) * ((soDer.at<float>(1, 0) * soDer.at<float>(2, 1)) - (soDer.at<float>(2, 0) * soDer.at<float>(1, 1)));

			if (det == 0){
				key_now.filtered = true;
				key_index++;
				continue;
			}

			//for (int a = 0; a < 3; a++){	// Transpose
			//	for (int b = 0; b < 3; b++){
			//		neg_soDer.at<float>(a, b) = soDer.at<float>(b, a);
			//	}
			//}
			float vals[4];
			/*cv::Mat ems = cv::Mat::zeros(3, 3, CV_32F);
			for (int a = 0; a < 3; a++){
				for (int b = 0; b < 3; b++){
					int box = 0;
					for (int i = 0; i < 3; i++){
						if (i == a) continue;
						for (int j = 0; j < 3; j++){
							if (j == b) continue;
							vals[box] = -soDer.at<float>(i, j);
							box++;
						}
					}
					ems.at<float>(a, b) = ((vals[0] * vals[3]) - (vals[1] * vals[2])) * pow(-1, 3 * a + b) / det;
				}
			}*/
			cv::Mat emt = cv::Mat::zeros(3, 3, CV_32F);
			/*for (int a = 0; a < 3; a++){	// Transpose
				for (int b = 0; b < 3; b++){
					emt.at<float>(a, b) = ems.at<float>(a, b);
				}
			}*/

			for (int iter = 0; iter < soDer.rows * soDer.cols; iter++){
				soDer.at<float>(iter) = -soDer.at<float>(iter);
			}

			emt = soDer.inv();

			cv::Mat expt;

			cv::solve(emt, foDer, expt);

			//cv::Mat extreme = emt * foDer;

			//float* exptr = (float*)extreme.datastart;
			//float mult = 1.0;

			//printf("Ex %d: %f, %f, %f\n", key_index, exptr[0], exptr[1], exptr[2]);
			//if ( abs(exptr[0]) > 0.5 || abs(exptr[1]) > 0.5 || abs(exptr[2]) > 0.5 ){
			//	key_now.filtered = true;
			//	key_index++;
			//	continue;
			//}

			float expt0 = (emt.at<float>(0, 0) * foDer.at<float>(0, 0)) + (emt.at<float>(0, 1) * foDer.at<float>(1, 0)) + (emt.at<float>(0, 2) * foDer.at<float>(2, 0));
			float expt1 = (emt.at<float>(1, 0) * foDer.at<float>(0, 0)) + (emt.at<float>(1, 1) * foDer.at<float>(1, 0)) + (emt.at<float>(1, 2) * foDer.at<float>(2, 0));
			float expt2 = (emt.at<float>(2, 0) * foDer.at<float>(0, 0)) + (emt.at<float>(2, 1) * foDer.at<float>(1, 0)) + (emt.at<float>(2, 2) * foDer.at<float>(2, 0));

			//printf("%f:%f  %f:%f  %f:%f\n", expt.at<float>(0), expt0, expt.at<float>(1), expt1, expt.at<float>(2), expt2);

			if (expt.at<float>(0,0) > 0.5 || expt.at<float>(1,0) > 0.5 || expt.at<float>(2,0) > 0.5){

				/*bool repeat = false;
				if (expt0 < -0.5){
					key_now.idx += 0.5;
					repeat = true;
				}
				else if (expt0 > 0.5){
					key_now.idx -= 0.5;
					repeat = true;
				}
				else if (expt1 < -0.5){
					key_now.idy += 0.5;
					repeat = true;
				}
				else if (expt1 > 0.5){
					key_now.idy -= 0.5;
					repeat = true;
				}
				if (repeat){
					printf("expt0: %f, expt1: %f, expt2: %f\n", expt0, expt1, expt2);
					continue;
				}*/

				printf(" %f:%f:%f \n", expt.at<float>(0, 0), expt.at<float>(1, 0), expt.at<float>(2, 0));
				keys[key_index].filtered = true;
				//key_now.filtered = true;
				key_index++;
				continue;
			}

			float ex_val = 0.0;
			//for (int i = 0; i < 3; i++){
			//	ex_val += tptr[i] * exptr[i];
			//}
			//ex_val += tptr[0] * expt.at<float>(0);
			//ex_val += tptr[1] * expt.at<float>(1);
			//ex_val += tptr[2] * expt.at<float>(2);
			ex_val = expt.dot(foDer);
			//printf(" %f:%f:%f ", expt.at<float>(0), expt.at<float>(1), expt.at<float>(2));
			printf(" %f:%f:%f ", foDer.at<float>(0), foDer.at<float>(1), foDer.at<float>(2));
			

			//printf("ex_val: %f  ", ex_val);

			//ex_val *= 0.5;
			//ex_val += dog_oct[key_now.oct][(int)key_now.index].at<float>(key_now.idx,key_now.idy);
			ex_val *= 0.5 + (dog_oct[key_now.oct][key_now.index].at<float>(key_now.idx, key_now.idy) - 127.5);
			printf("ex_val: %f ", abs(ex_val));	//Fix Later
			if (abs(ex_val) < 0.03){
				
				//printf("ex_val: %f\n", abs(ex_val));	//Fix Later
				printf("\n");
				key_now.filtered = true;
				key_index++;
				continue;
			}

			float h_trace = dxx + dyy;
			float h_det = dxx * dyy - pow(dxy, 2);

			if (h_det <= 0 || pow(h_trace,2) / h_det > t){
				key_now.filtered = true;
				printf("\n");
			}
			else
				printf("passed \n");
			key_index++;
		}
		nvtxRangePop();
	}


	//void mySiftDescriptors
	void mySiftDescriptors(std::vector<keypoint>& keys, std::vector<std::vector<cv::Mat>>& blur_oct, std::vector<std::vector<float*>>& or_mag_oct, int unfiltered){
		int region = 8;
		int key_count = keys.size();
		nvtxEventAttributes_t eventAttrib = get_nvtAttrib("mySiftDescriptors [" + std::to_string(key_count) + " ]", 0xFF00FF00);
		//printf("Key Count: %d\n", key_count);
		//getchar();

		for (int key_index = 0; key_index < key_count; key_index++){

			//printf("Key: %d\n", key_index);

			keypoint& key_now = keys[key_index];
			if (key_now.filtered == true){
				continue;
			}
			
			cv::Mat& current = blur_oct[key_now.oct][key_now.index];
			float* cur_or_mag = or_mag_oct[key_now.oct][key_now.index];
			int curRows = current.rows, curCols = current.cols;

			if (key_now.idx < region || key_now.idx >= current.rows - region || 
				key_now.idy < region || key_now.idy >= current.cols - region ||
				key_now.filtered == true){

				key_now.filtered = true;
				continue;
			}

			//printf("Fish 0 %d,%d\n", key_now.idx, key_now.idy);
			//printf("Angle: %f\n", key_now.angle);

			int or_mag_size = 2 * curRows * curCols;
			int W = 2 * region + 1;
			float* orientations = new float[W * W];
			float* magnitudes = new float[W * W];
			double* gKernel = new double[W * W];
			createFilter(gKernel, 8.0, region);
			cv::Mat gauss_cur = current(cv::Rect(key_now.idy - region, key_now.idx - region, W, W));

			//printf("Fish 1 %d,%d\n", key_now.idx, key_now.idy);

			for (int i = 0; i < W; i++){
				for (int j = 0; j < W; j++){
					int om_i = key_now.idx - region + i;
					int om_j = key_now.idy - region + j;
					orientations[i * W + j] = cur_or_mag[2 * (om_i * curCols + om_j)] + key_now.angle;
					if (orientations[i * W + j] > 2 * M_PI){
						orientations[i * W + j] -= 2 * M_PI;
					}
					else if (orientations[i * W + j] < 0){
						orientations[i * W + j] += 2 * M_PI;
					}
					magnitudes[i * W + j] = cur_or_mag[2 * (om_i * curCols + om_j) + 1] * gKernel[i * W + j];
				}
			}

			delete[] gKernel;
			
			//printf("Fish 2 %d,%d\n", key_now.idx, key_now.idy);
			std::vector<float> descriptors;

			for (int x = 0; x < W - 1; x += 4){
				for (int y = 0; y < W - 1; y += 4){

					std::vector<float> bins(8, 0);
					for (int i = x; i < x + 4; i++){
						for (int j = y; j < y + 4; j++){

							//printf("3 %d,%d\n", i, j);
							float sum = magnitudes[i * W + j] * gauss_cur.at<float>(i, j);
							int bin = (orientations[i * W + j] * (180 / M_PI)) / 45.0;
							//printf("Bin: %d\n", bin);
							//getchar();
							bins[bin] += sum;
						}
					}
					//printf("Fish 4 %d,%d\n", key_now.idx, key_now.idy);
					mySiftVectorThreshold(bins);
					//printf("Fish 5\n");
					descriptors.insert(descriptors.end(), bins.begin(), bins.end());
					//printf("Fish 6\n");
				}
			}

			key_now.descriptors = descriptors;
			delete[] orientations;
			delete[] magnitudes;
			//printf("Fish 7\n");
		}
		nvtxRangePop();
	}

	std::vector<float> mySiftVectorThreshold(std::vector<float>& vec){
		mySiftNormVec(vec);
		std::vector<float> res;
		bool threshold = false;
		for (float& elem : vec){
			if (elem <= 0.2)
				res.push_back(elem);
			else{
				res.push_back(0.2);
				threshold = true;
			}
		}
		if (threshold)
			mySiftNormVec(res);
		return res;
	}

	void mySiftNormVec(std::vector<float>& vec){
		float length = 0;
		for (int i = 0; i < vec.size(); i++){
			length += vec[i];
		}
		if (length == 0) return;
		for (int i = 0; i < vec.size(); i++){
			vec[i] = vec[i] / length;
		}
	}

	int index(int idx, int idy, int cols, int type){
		if (type == 1){
			return 3 * (idx * cols + idy);
		}
		else if (type == 0){
			return (idx * cols) + idy;
		}
		return 0;
	}

	cv::Mat mySift_foDer(std::vector<cv::Mat>& neighbors, int px, int py){
		//nvtxEventAttributes_t eventAttrib1 = get_nvtAttrib("mySift_foDer", 0x0088FF00);
		cv::Mat result = cv::Mat::zeros(3, 1, CV_32FC1);
		//printf("Test: %d\n", neighbors[1].at<cv::Vec3b>(px - 1, py)[0]);
		int rows = neighbors[1].rows, cols = neighbors[1].cols;
		float* cur_ptr = (float*)neighbors[1].datastart;
		float* pre_ptr = (float*)neighbors[0].datastart;
		float* nex_ptr = (float*)neighbors[2].datastart;

		const float dx = ((float)cur_ptr[index(px, py - 1, cols, 0)] - (float)cur_ptr[index(px, py + 1, cols, 0)]) / 2.0;
		const float dy = ((float)cur_ptr[index(px - 1, py, cols, 0)] - (float)cur_ptr[index(px + 1, py, cols, 0)]) / 2.0;
		const float ds = ((float)pre_ptr[index(px, py, cols, 0)] - (float)nex_ptr[index(px, py, cols, 0)]) / 2.0;

		//printf("dx: %f, dy: %f, ds: %f\n", dx, dy, ds);

		//const float dx = (neighbors[1].at<cv::Vec3b>(px - 1, py)[0] - neighbors[1].at<cv::Vec3b>(px + 1, py)[0]) / 2.0;// printf("Line 1  ");
		//const float dy = (neighbors[1].at<cv::Vec3b>(px, py - 1)[0] - neighbors[1].at<cv::Vec3b>(px, py + 1)[0]) / 2.0;// printf("Line 2  ");
		//const float ds = (neighbors[0].at<cv::Vec3b>(px, py)[0] - neighbors[2].at<cv::Vec3b>(px, py)[0]) / 2.0;//  printf("Line 3  ");
		//printf("Key %d\n", 0);
		float* res_ptr = (float*)result.datastart;
		res_ptr[0] = dx;
		res_ptr[1] = dy;
		res_ptr[2] = ds;

		//result.at<float>(0, 0) = dx;//  printf("Line 4  ");
		//result.at<float>(1, 0) = dy;//  printf("Line 5  ");
		//result.at<float>(2, 0) = ds;//  printf("Line 6\n");
		//nvtxRangePop();
		return result;
	}

	cv::Mat mySift_soDer(std::vector<cv::Mat>& neighbors, int px, int py){
		//nvtxEventAttributes_t eventAttrib1 = get_nvtAttrib("mySift_soDer", 0xFF880000);
		cv::Mat result = cv::Mat::zeros(3, 3, CV_32FC1);

		int rows = neighbors[1].rows, cols = neighbors[1].cols;
		float* cur_ptr = (float*)neighbors[1].datastart;
		float* pre_ptr = (float*)neighbors[0].datastart;
		float* nex_ptr = (float*)neighbors[2].datastart;

		const float dxx = (float)(cur_ptr[index(px, py + 1, cols, 0)] + cur_ptr[index(px, py - 1, cols, 0)]) - (2.0 * cur_ptr[index(px, py, cols, 0)]);
		const float dyy = (float)(cur_ptr[index(px + 1, py, cols, 0)] + cur_ptr[index(px - 1, py, cols, 0)]) - (2.0 * cur_ptr[index(px, py, cols, 0)]);
		const float dss = (float)(nex_ptr[index(px, py, cols, 0)] + pre_ptr[index(px, py, cols, 0)]) - (2.0 * cur_ptr[index(px, py, cols, 0)]);
		const float dxy = (float)(cur_ptr[index(px + 1, py + 1, cols, 0)] - cur_ptr[index(px + 1, py - 1, cols, 0)] - cur_ptr[index(px - 1, py + 1, cols, 0)] + cur_ptr[index(px - 1, py - 1, cols, 0)]) / 2.0;
		const float dxs = (float)(nex_ptr[index(px, py + 1, cols, 0)] - nex_ptr[index(px, py - 1, cols, 0)] - pre_ptr[index(px, py + 1, cols, 0)] + pre_ptr[index(px, py - 1, cols, 0)]) / 2.0;
		const float dys = (float)(nex_ptr[index(px + 1, py, cols, 0)] - nex_ptr[index(px - 1, py, cols, 0)] - pre_ptr[index(px + 1, py, cols, 0)] + pre_ptr[index(px - 1, py, cols, 0)]) / 2.0;

		float* res_ptr = (float*)result.datastart;

		res_ptr[0] = dxx;
		res_ptr[1] = dxy;
		res_ptr[2] = dxs;
		res_ptr[3] = dxy;
		res_ptr[4] = dyy;
		res_ptr[5] = dys;
		res_ptr[6] = dxs;
		res_ptr[7] = dys;
		res_ptr[8] = dss;

		//nvtxRangePop();
		return result;
	}

	bool mySiftWriteKeyFile(std::vector<keypoint>& keys){
		std::ofstream key_file;
		key_file.open("D://School//Summer 2016//Research//mySift//keys_cpu.txt");

		for (keypoint& key : keys){
			if (!key.filtered){
				key_file << std::to_string(key.idx) << ":" << std::to_string(key.idy) << ":" << std::to_string(key.oct) << ":" << std::to_string(key.index) << ":";
				key_file << std::to_string(key.angle) << ":" << std::to_string(key.scale);

				for (float val : key.descriptors){
					key_file << ":" << std::to_string(val);
				}
				key_file << ":" << std::endl;
			}
		}

		key_file.close();
		return true;
	}

	bool mySiftReadKeyFile(std::vector<keypoint>& keys, std::string file_name){
		std::string line;
		std::ifstream key_file(file_name);
		if (!key_file.is_open()){
			printf("Can't find file!\n");
			return false;
		}

		std::string delimiter = ":";

		while (std::getline(key_file, line)){
			
			//keypoint read_key;
			size_t pos = line.find(delimiter);
			int idx = stoi(line.substr(0,pos));
			line.erase(0, pos + 1);

			pos = line.find(delimiter);
			int idy = stoi(line.substr(0, pos));
			line.erase(0, pos + 1);

			pos = line.find(delimiter);
			int oct = stoi(line.substr(0, pos));
			line.erase(0, pos + 1);

			pos = line.find(delimiter);
			int index = stoi(line.substr(0, pos));
			line.erase(0, pos + 1);

			pos = line.find(delimiter);
			float angle = stof(line.substr(0, pos));
			line.erase(0, pos + 1);

			pos = line.find(delimiter);
			float scale = stof(line.substr(0, pos));
			line.erase(0, pos + 1);

			keypoint read_key(idx, idy, oct, angle, index);
			read_key.scale = scale;

			float desc = 0;
			while ((pos = line.find(delimiter)) != std::string::npos){
				desc = stof(line.substr(0, pos));
				read_key.descriptors.push_back(desc);
				line.erase(0, pos + 1);
			}
			keys.push_back(read_key);
		}

		key_file.close();
		return true;
	}

	void mySiftKeyCull(std::vector<keypoint>& keys){
		get_nvtAttrib("Vector Cull " + std::to_string(keys.size()), 0xFF0000FF);
		std::vector<keypoint>::iterator wall = keys.begin();
		std::vector<keypoint>::iterator current = keys.begin();
		std::vector<keypoint>::iterator back = keys.end();

		while (current != back){
			if ((*current).filtered == false){
				std::iter_swap(current, wall);
				wall++;
			}
			current++;
		}
		keys.resize(std::distance(keys.begin(), wall));
		nvtxRangePop();
	}

	kd_node mySiftKDHelp(std::vector<keypoint>& keys){
		std::string dims = "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

		int count = 0;
		//printf("Starting the main tree\n");

		float all_var[128] = { 0 }, all_mean[128] = { 0 };
		std::vector<keypoint>::iterator front = keys.begin();
		std::vector<keypoint>::iterator back = keys.end();
		//printf("Fish 0\n");

		int dist = std::distance(front, back);
		for (std::vector<keypoint>::iterator iter = front; iter != back; iter++){
			if ((*iter).filtered == true) continue;
			for (int i = 0; i < 128; i++)
				all_mean[i] += (*iter).descriptors[i] / (float)dist;
		}
		//printf("Fish 1\n");

		for (std::vector<keypoint>::iterator iter = front; iter != back; iter++){
			for (int i = 0; i < 128; i++)
				all_var[i] += (((*iter).descriptors[i] - all_mean[i]) * ((*iter).descriptors[i] - all_mean[i])) / (float)dist;
		}
		//printf("Fish 2\n");

		kd_node* curr = mySiftKDTree(keys, keys.begin(), keys.end(), dims, all_var, count);
		curr->parent = 0;
		//printf("Count: %d\n", count);
		//delete[] all_var;
		//delete[] all_mean;

		return *(curr);
	}

	kd_node* mySiftKDTree(std::vector<keypoint>& keys, std::vector<keypoint>::iterator front, std::vector<keypoint>::iterator back, std::string dims, float* all_var, int& count){

		//printf("One Fish ");
		kd_node* curr = new kd_node;
		curr->leaf_begin = front;
		curr->leaf_end = back;

		if (std::distance(front, back) <= 1 || dims.find('0') == std::string::npos){
			curr->leaf = true;
			count += 1;
			return curr;
		}

		int dist = std::distance(front, back);

		/*float* data_mean = new float[128];
		float* data_var = new float[128];

		for (int i = 0; i < 128; i++){
			data_mean[i] = 0;
			data_var[i] = 0;
		}

		for (std::vector<keypoint>::iterator iter = front; iter != back; iter++){
			for (int i = 0; i < 128; i++){
				if (dims[i] == '1')
					continue;
				data_mean[i] += (*iter).descriptors[i];
				//data_var[i] = std::max(data_var[i], (*iter).descriptors[i]);
			}
		}

		for (int i = 0; i < 128; i++){
			if (dims[i] == '1')
				continue;
			data_mean[i] /= dist;
		}

		for (std::vector<keypoint>::iterator iter = front; iter != back; iter++){
			for (int i = 0; i < 128; i++){
				if (dims[i] == '1')
					continue;
				data_var[i] += ((*iter).descriptors[i] - data_mean[i]) * ((*iter).descriptors[i] - data_mean[i]);
			}
		}

		for (int i = 0; i < 128; i++){
			if (dims[i] == '1')
				continue;
			data_var[i] /= dist;
		}*/

		int dim = 0;
		float max_diff = 0.0;

		for (int i = 0; i < 128; i++){
			if (dims[i] == '1')
				continue;
			if (all_var[i] >= max_diff){
				dim = i;
				max_diff = all_var[i];
			}
		}

		//printf("Two Fish ");

		dims[dim] = '1';
		curr->dim = dim;
		//delete[] data_mean;
		//delete[] data_var;

		//mySiftKDQuicksort(keys, front, back, dim);
		//printf("Sorting now!\n");
		mySiftKDRadixSort(keys, front, back, dim);
		//printf("Sorting done!\n");
		int middle = std::distance(front, back) / 2;
		curr->median = (*(front + middle)).descriptors[dim];

		//printf("Red Fish ");

		//printf("Left: %d, Right: %d\n", std::distance(front, front + middle), std::distance(front + middle, back));
		//getchar();

		curr->left = mySiftKDTree(keys, front, front + middle, dims, all_var, count);
		curr->left->parent = curr;
		curr->right = mySiftKDTree(keys, front + middle, back, dims, all_var, count);
		curr->right->parent = curr;

		//printf("Blue Fish \n");

		//if (curr.left->leaf == false){
		//	printf("Test: %d\n", curr.left->dim);
		//}

		return curr;
	}

	void mySiftKDQuicksort(std::vector<keypoint>& keys, std::vector<keypoint>::iterator front, std::vector<keypoint>::iterator back, int dim){
		if (std::distance(front,back) <= 1){
			return;
		}
		else{
			//printf("Dist: %d\n", std::distance(front, back));
		}
		//std::vector<keypoint>::iterator pivot = back - 1;
		//std::vector<keypoint>::iterator wall = front;
		//std::vector<keypoint>::iterator current = front;
		std::vector<keypoint>::iterator pivot(back); pivot--;
		std::vector<keypoint>::iterator wall(front);
		std::vector<keypoint>::iterator current(front);
		while (current != pivot){
			if ((*current).descriptors[dim] <= (*pivot).descriptors[dim]){
				std::iter_swap(current, wall);
				wall++;
			}
			current++;
		}
		std::iter_swap(pivot, wall);

		printf("Dist: %d\n", std::distance(front, wall));

		if ((*wall).descriptors[dim] != 0.0) mySiftKDQuicksort(keys, front, wall, dim);
		mySiftKDQuicksort(keys, wall + 1, back, dim);
	}

	unsigned int radixGetMax(unsigned int arr[], int n){
		unsigned int max = arr[0];
		for (int i = 1; i < n; i++){
			if (arr[i] > max) max = arr[i];
		}
		return max;
	}

	void mySiftKDCountSort(unsigned int* data, unsigned int* index, int d, int exp){
		//printf("Size: %d\n", d);
		unsigned int* output = new unsigned int[d];
		unsigned int* out_id = new unsigned int[d];
		int count[10] = { 0 };
		int i;

		for (i = 0; i < d; i++){
			//printf("Test1: %f, %u\n", data[i], data[i]);
			count[(data[i] / exp) % 10] ++;
		}

		//for (i = 0; i < 10; i++){
		//	printf("%d ", count[i]);
		//}
		//printf("\n");

		for (i = 1; i < 10; i++){
			//printf("S%d: c[i] %d, c[i - 1] %d \n", i, count[i], count[i - 1]);
			count[i] += count[i - 1];
		}

		for (i = d - 1; i >= 0; i--){
			//printf("Test1: %d\n", (data[i] / exp) % 10);
			//printf("Test2: %d\n", count[(data[i] / exp) % 10] - 1);
			output[count[(data[i] / exp) % 10] - 1] = data[i];
			out_id[count[(data[i] / exp) % 10] - 1] = index[i];
			count[(data[i] / exp) % 10] -= 1;
		}

		for (i = 0; i < d; i++){
			data[i] = output[i];
			index[i] = out_id[i];
		}

		delete[] output;
		delete[] out_id;
	}

	void mySiftKDRadixSort(std::vector<keypoint>& keys, std::vector<keypoint>::iterator front, std::vector<keypoint>::iterator back, int dim){
		int d = std::distance(front, back);
		float* temp = new float[d];
		unsigned int* data;// = new unsigned int[d];
		unsigned int* index = new unsigned int[d];
		for (int i = 0; i < d; i++){
			temp[i] = (*(front + i)).descriptors[dim];
			index[i] = i;
			//printf("%f, %u \n", (*(front + i)).descriptors[dim], temp[i]);
		}
		data = (unsigned int*)temp;
		unsigned int max = radixGetMax(data, d);

		for (int exp = 1; max / exp > 0; exp *= 10){
			mySiftKDCountSort(data, index, d, exp);
		}

		std::vector<keypoint> newVec(front, back);

		for (int i = 0; i < d; i++){
			*(front + i) = newVec[index[i]];
		}

		delete[] data;
		delete[] index;
	}

	float mySiftDescDist(keypoint& key_1, keypoint& key_2){
		float sq_dist = 0.0;

		for (int i = 0; i < 128; i++){
			sq_dist += (key_1.descriptors[i] - key_2.descriptors[i]) * (key_1.descriptors[i] - key_2.descriptors[i]);
		}

		return sq_dist;
	}

	float mySiftTheoryDist(kd_node* start, keypoint& search_key){
		keypoint new_key(search_key.idx, search_key.idy, search_key.oct, search_key.angle, search_key.index);
		for (int i = 0; i < 128; i++){
			new_key.descriptors.push_back(search_key.descriptors[i]);
		}

		while (start->parent != 0){
			if (search_key.descriptors[start->dim] >= start->median && start->parent->left == start){
				new_key.descriptors[start->dim] = start->median;
			}
			else if (search_key.descriptors[start->dim] < start->median && start->parent->right == start){
				new_key.descriptors[start->dim] = start->median;
			}
			start = start->parent;
		}

		return mySiftDescDist(search_key, new_key);
	}

	kd_node* mySiftKDIterSearch(kd_node* root, std::vector<keypoint>& keys, keypoint& search_key){
		int max_search = 7;
		
		kd_node* current = root;
		std::vector<kd_node*> search_pattern;
		std::vector<kd_node*> branches;
		branches.push_back(root);

		float closest_dist = 0.0;
		kd_node* current_best = root;

		while (true){
			if (branches.size() == 0){
				break;
			}
			current = branches.back();
			branches.pop_back();
			//search_pattern.push_back(current);
			if (current->leaf == true){
				//No iterating through bins right now
				keypoint test = *(current->leaf_begin);
				closest_dist = mySiftDescDist(search_key, test);
				current_best = current;
				printf("Dist: %f\n",closest_dist);
				break;
			}

			if (search_key.descriptors[current->dim] < current->median){
				//current = current->left;
				branches.push_back(current->right);
				branches.push_back(current->left);

			}
			else{
				//current = current->right;
				branches.push_back(current->left);
				branches.push_back(current->right);
			}

		}

		
		while (true){
			if (branches.size() == 0)
				break;
			current = branches.back();
			branches.pop_back();

			if (current->leaf == true){
				keypoint test = *(current->leaf_begin);
				float cur_dist = mySiftDescDist(search_key, test);
				printf("Dist: %f\n", cur_dist);
				if (cur_dist < closest_dist){
					closest_dist = cur_dist;
					current_best = current;
				}
				if (--max_search == 0)
					break;
				continue;
			}

			if (mySiftTheoryDist(current,search_key) > closest_dist){
				continue;
			}

			if (search_key.descriptors[current->dim] < current->median){
				branches.push_back(current->right);
				branches.push_back(current->left);
			}
			else{
				branches.push_back(current->left);
				branches.push_back(current->right);
			}

		}

		printf("Max_search: %d; Closest_dist: %f\n", max_search, closest_dist);

		return current_best;
	}

	kd_node* mySiftKDSearch(kd_node root, std::vector<keypoint>& keys, keypoint& search_key){
		int max_search = 5;

		kd_node* kd_out = mySiftKDSearchHelp(&root, keys, search_key, max_search);
		printf("Came Back!\n");
		return kd_out;
	}

	kd_node* mySiftKDSearchHelp(kd_node* current, std::vector<keypoint>& keys, keypoint& search_key, int& max_search){
		if (current->leaf == true){
			max_search -= 1;
			//printf("Trouble!\n");
			return current;
		}
		//printf("Test: %d\n", std::distance(current->leaf_begin,current->leaf_end));

		int dim = current->dim;
		int median = current->median;

		kd_node kd_final;

		if (search_key.descriptors[dim] < median){
			//printf("Step Down Left!\n");
			kd_node* back_out = mySiftKDSearchHelp(current->left, keys, search_key, max_search);

			//printf("Step Up Left!\n");
			return back_out;
		}
		else{
			//printf("Step Down Right!\n");
			kd_node* back_out = mySiftKDSearchHelp(current->right, keys, search_key, max_search);
			//printf("Step Up Right!\n");
			return back_out;
		}
	}

	cv::Mat bump_map(int dim){
		cv::Mat output(dim, dim, CV_8UC3);
		uchar* dest_data = (uchar*)output.datastart;
		float mlength = 220.0;
		float temp = acosf(mlength / 255.0);
		for (int i = 0; i < dim; i++){
			for (int j = 0; j < dim; j++){
				uchar b = 255, g = 128, r = 128;
				int rel_coori = i % 16;
				int rel_coorj = j % 16;

				float dist = sqrt(pow((float)rel_coori - 7.5, 2.0) + pow((float)rel_coorj - 7.5, 2.0));

				if (dist > 4 && dist <= 6){
					float ang = atan2((float)rel_coorj - 7.5, (float)rel_coori - 7.5);
					
					g = 128 - (uchar)((mlength / 255.0) * 128) * cos(ang);
					r = 128 + (uchar)((mlength / 255.0) * 128) * sin(ang);

					b = 255 * sin(temp);

				}

				dest_data[3 * (dim * i + j)] = b;
				dest_data[3 * (dim * i + j) + 1] = g;
				dest_data[3 * (dim * i + j) + 2] = r;
			}
		}

		return output;
	}

	cv::Mat diff_count(cv::Mat image1, cv::Mat image2){
		float* data_1 = (float*)image1.datastart;
		float* data_2 = (float*)image2.datastart;

		int rows = image1.rows, cols = image1.cols;
		int dcount = 0;

		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				int idx = i * cols + j;
				if (data_1[idx] != data_2[idx]){ 
					dcount++; 
					//data_1[idx] = data_2[idx];
					data_1[idx] = 0.0;
				}
				else{
					data_1[idx] = data_2[idx];
					//data_1[idx] = 0.0;
				}
			}
		}

		printf("\n\nDiff: %d\n", dcount);
		return image1;
	}

	float mySiftVertParabola(float l_x, float l_y, float p_x, float p_y, float r_x, float r_y){
		cv::Mat_<float> mat_a = cv::Mat::zeros(3, 3, CV_32F);
		cv::Mat_<float> mat_b = cv::Mat::zeros(3, 1, CV_32F);
		//l_x = l_x * ((M_PI * 10.0) / 180.0); p_x = p_x * ((M_PI * 10.0) / 180.0); r_x = r_x * ((M_PI * 10.0) / 180.0);
		//printf("Points: l_x %f, l_y %f, p_x %f, p_y %f, r_x %f, r_y %f\n", l_x, l_y, p_x, p_y, r_x, r_y);

		mat_a.at<float>(0, 0) = l_x * l_x;
		mat_a.at<float>(1, 0) = p_x * p_x;
		mat_a.at<float>(2, 0) = r_x * r_x;

		mat_a.at<float>(0, 1) = l_x;
		mat_a.at<float>(1, 1) = p_x;
		mat_a.at<float>(2, 1) = r_x;

		mat_a.at<float>(0, 2) = 1.0;
		mat_a.at<float>(1, 2) = 1.0;
		mat_a.at<float>(2, 2) = 1.0;

		//printf("[ %10.1f, %10.1f, %10.1f ]\n", mat_a.at<float>(0, 0), mat_a.at<float>(0, 1), mat_a.at<float>(0, 2));
		//printf("[ %10.1f, %10.1f, %10.1f ]\n", mat_a.at<float>(1, 0), mat_a.at<float>(1, 1), mat_a.at<float>(1, 2));
		//printf("[ %10.1f, %10.1f, %10.1f ]\n", mat_a.at<float>(2, 0), mat_a.at<float>(2, 1), mat_a.at<float>(2, 2));
		//printf("\n");

		mat_b.at<float>(0, 0) = l_y;
		mat_b.at<float>(1, 0) = p_y;
		mat_b.at<float>(2, 0) = r_y;

		//printf("[ %10.6f ]\n", mat_b.at<float>(0, 0));
		//printf("[ %10.6f ]\n", mat_b.at<float>(1, 0));
		//printf("[ %10.6f ]\n", mat_b.at<float>(2, 0));
		//printf("\n");

		cv::Mat mat_result = cv::Mat::zeros(3, 1, CV_32F);
		solve(mat_a, mat_b, mat_result);

		float result = -mat_result.at<float>(1, 0) / (2.0 * mat_result.at<float>(0, 0));
		if (result < 0) result += 360;
		else if (result > 360) result -= 360;

		//printf("Result: %f\n", result);
		//getchar();

		result *= M_PI / 180.0;
		return result;
	}



}