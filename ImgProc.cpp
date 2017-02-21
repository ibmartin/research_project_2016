#include "ImgProc.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>

namespace img_proc{

	//Color channels in cv::Mat objects are arranged [B,G,R,...], in a single array of size 3 * rows * cols, row-major order.
	//The data type for colors is uchar, which is 1 Byte and ranges 0 to 255
	//Raw values of image data can be accessed using (uchar*)image.datastart

	//	== cv::Mat rgb2Gray(cv::Mat image) ==
	//		cv::Mat image: input image
	//	Takes an RGB image and converts it to grayscale
	cv::Mat rgb2Gray(cv::Mat image){	
		int srcRows = image.rows;
		int srcCols = image.cols;

		cv::Mat output(srcRows, srcCols, CV_8UC1);
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_data = (uchar*)output.datastart;
		uchar* dest_end = (uchar*)output.dataend;

		while (dest_data <= dest_end){
			*(dest_data) = 0.299 * (*(src_data + 2)) + 0.587 * (*(src_data + 1)) + 0.114 * (*(src_data));	//Based on color curves I found representing contribution of each channel to grayscale sensitivity
			dest_data += 1;
			src_data += 3;
		}

		return output;
	}

	cv::Mat frgb2Gray(cv::Mat image){
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
		return output;
	}

	//	== cv::Mat reverse(cv::Mat image) ==
	//		cv::Mat image: input image
	//	Outputs a negative color image of the input
	cv::Mat reverse(cv::Mat image){	
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

		return output;
	}

	//	== cv::Mat gammaCorrection(cv::Mat image, double gamma) ==
	//		cv::Mat image: input image
	//		double gamma: gamma correction value, gamma > 1 darkens the image, gamma < 1 brightens the image
	//	Applies gamma correction which brightens or darkens the image based on gamma
	cv::Mat gammaCorrection(cv::Mat image, double gamma){	

		cv::Mat output(image.rows, image.cols, image.type());

		uchar* dest_data = (uchar*)output.datastart;
		uchar* src_data = (uchar*)image.datastart;
		uchar* dest_end = (uchar*)output.dataend;
		uchar* src_end = (uchar*)image.dataend;

		double gammaCorrect = 1.00 / gamma;

		while (dest_data <= dest_end && src_data <= src_end){
			double color = (double)(*(src_data));
			uchar val = 255 * pow((color / 255.0), gammaCorrect);
			*(dest_data) = val;
			src_data++;
			dest_data++;
		}

		return output;
	}

	//	== cv::Mat directResize(cv::Mat image, int rows, int cols) ==
	//		cv::Mat image: input image
	//		int rows: number of rows for the new image to have
	//		int cols: number of columns for the new image to have
	//	Resize with no interpolation, just chooses closest pixel
	cv::Mat directResize(cv::Mat image, int rows, int cols){
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

		return output;
	}

	//	== cv::Mat fdirectResize(cv::Mat image, int rows, int cols) ==
	//		cv::Mat image: input image
	//		int rows: number of rows for the new image to have
	//		int cols: number of columns for the new image to have
	//	Resize with no interpolation, just chooses closest pixel
	cv::Mat fdirectResize(cv::Mat image, int destRows, int destCols){
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
		return out;
	}

	//	== cv::Mat linearResize(cv::Mat image, int rows, int cols) ==
	//		cv::Mat image: input image
	//		int rows: number of rows for the new image to have
	//		int cols: number of columns for the new image to have
	//	Resize with linear interpolation based on 4 nearest pixels
	cv::Mat linearResize(cv::Mat image, int rows, int cols){
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

		int W = 2 * FILTER_SIZE + 1;
		int center = W / 2;
		double* gKernel = new double[W * W];
		createFilter(gKernel, sigma, FILTER_SIZE);

		int srcRows = image.rows;
		int srcCols = image.cols;

		//cv::Mat output(image.rows, image.cols, image.type());
		cv::Mat output = cv::Mat::zeros(image.rows,image.cols,CV_32FC1);

		float* src_data = (float*)image.datastart;
		float* dest_data = (float*)output.datastart;

		for (int i = 0; i < srcRows; i++){
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



		//Edge Thinning
		//Thins out pixel values that are too weak to represent a strong edge in the original image
		//Also strengthens strong edges

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

		//	Main work loop.  First, the "color distances" of each pixel to the color of each main group, and re-assigns groups if necessary.
		//	Then, if any pixels have been re-assigned, the average colors of all the groups are re-calculated based on the new assignments
		while (!convergence){
			convergence = true;
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
					for (int color = 0; color < 3; color++){
						float src_val = src_data[3 * (i * srcCols + j) + color];
						int group_count = k_count[group];

						float val = src_val / group_count;
						k_colors[3 * group + color] += val;
					}

				}
			}
		}//	End of main while loop

		//	Writing to output image using the last discoved group color values and the group assignment of every pixel in
		//	the input image
		for (int i = 0; i < srcRows; i++){
			for (int j = 0; j < srcCols; j++){
				int group = k_index[i * srcCols + j];

				for (int color = 0; color < 3; color++){
					dest_data[3 * (i * srcCols + j) + color] = (uchar)k_colors[3 * group + color];
				}

			}
		}

		
		//Avoid memory leaks
		delete[] k_colors;
		delete[] k_index;
		delete[] k_count;

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

	//	== cv::Mat mySift(cv::Mat image) ==
	//		cv::Mat image: Input image
	//	Applies Scale Invariant Feature Transform to the input image in order to identify features that can be used for 
	//	object recognition.  This algorithm currently builds the scale space and uses the difference-of-gaussians to identify potential
	//	keypoints.  Not yet implemented are keypoint filtering, orientation and magnitude calculations, and keypoint storage or analysis.
	cv::Mat mySift(cv::Mat original){

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
		//cv::Mat original = image;

		cv::Mat image = linearResize(original, original.rows * 2, original.cols * 2);
		image = frgb2Gray(image);
		//image = flinearResize(image, image.rows * 2, image.cols * 2);
		//image = fgaussianFilter(image, sigma);

		//image = linearResize(image, image.rows * 2, image.cols * 2);
		//image = gaussianFilter(image, sigma);

		uchar scales = 3;	//	Lowe found 3 to be best
		uchar octaves = 4;	//	4 or 5 octaves is the standard
		int region = 4;
		int srcRows = image.rows;
		int srcCols = image.cols;

		//	In one octave, a blurred image is related to the previous image by a factor of 2^(1/scales), see SIFT 2004 page 7
		float k = pow(2.0, (1.0 / (float)scales));
		int s = scales + 3;

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
			std::vector<cv::Mat> blur_img;
			std::vector<cv::Mat> dog_img;

			//cv::Mat current = image;//CHANGED
			//blur_img.push_back(image);//CHANGED
			cv::Mat current = fgaussianFilter(image, pow(k, gauss_exp) * sigma);
			//printf("Gauss_exp: %f\n", (pow(k, (float)gauss_exp) * sigma) - (pow(k, (float)gauss_exp - 1) * sigma));
			blur_img.push_back(current);
			gauss_exp += 1;
			//cv::Mat origin = image.clone();
			//std::string img_num = "_" + std::to_string(oct) + "_" + std::to_string(0);
			//std::string full_img = debug_Path + img_name + img_num + ftype;
			//imwrite(full_img, current, compression_params);

			for (int step = 1; step < s; step++){
				//cv::Mat next = fgaussianFilter(image, k);	//CHANGED	Applies blur of strength k to previous image, until sigma is double that of the first image in octave
				cv::Mat next = fgaussianFilter(image, pow(k, gauss_exp) * sigma);	//	Applies blur of strength k to previous image, until sigma is double that of the first image in octave
				cv::Mat dog = cv::Mat::zeros(curRows,curCols,CV_32FC1);
				blur_img.push_back(next);
				//printf("Gauss_exp: %f\n", (pow(k, (float)gauss_exp) * sigma) - (pow(k, (float)gauss_exp - 1) * sigma));

				float* curr_data = (float*)current.data;
				float* next_data = (float*)next.data;
				float* dog_data = (float*)dog.data;
				//printf("Step: %d\n", current.step);
				//getchar();
				//printf("Cont: %d\n", dog.step1());
				for (int i = 0; i < curRows ; i++){
					for (int j = 0; j < curCols; j++){
						//int idx = 3 * (i * curCols + j);
						//getchar();
						int idx = i * curCols + j;

						//printf("Next: %f, Curr: %f\n", next_data[idx], curr_data[idx]);
						//printf("Diff: %f\n", next_data[idx] - curr_data[idx]);

						float val = 0.0 + (next_data[idx] - curr_data[idx]);// + 127;// Data is stored as gray to allow for presentable data.  Adding 127 reduces risk of wrap-around error (range is 0 - 255).
						
						dog_data[idx] = val;
						//dog_data[idx + 1] = val;
						//dog_data[idx + 2] = val;
					}
				}

				
				//printf("Bogeys: %f\n", (float)(bogeys) / (curRows * curCols));
				//getchar();
				//std::string img_num = "_" + std::to_string(oct) + "_" + std::to_string(step + 1);
				//std::string full_img = debug_Path + img_name + img_num + ftype;
				//imwrite(full_img, next, compression_params);

				dog_img.push_back(dog);
				current = next;
				gauss_exp++;
			}

			//	Keypoint calculation.  Refer to SIFT papers for more information
			for (int step = 1; step < s - 2; step++){
				int temp_exp = gauss_exp - s + (step);
				//printf("temp_exp: %d\n", temp_exp);
				float temp_scale = ((pow(k, temp_exp) * sigma) - (pow(k, temp_exp - 1) * sigma)) / 2 + (pow(k, temp_exp - 1) * sigma);

				cv::Mat& prev = dog_img[step - 1];
				cv::Mat& curr = dog_img[step];
				cv::Mat& next = dog_img[step + 1];
				float* prev_data = (float*)dog_img[step - 1].datastart;
				float* curr_data = (float*)dog_img[step].datastart;
				float* next_data = (float*)dog_img[step + 1].datastart;

				for (int i = 1; i < curRows - 2; i++){
					for (int j = 1; j < curCols - 1; j++){
						//int idx = 3 * (i * curCols + j);
						int idx = (i * curCols + j);
						float val = curr_data[idx];
						//float val = curr.ptr<float>(i)[j];
						//float val = curr.at<float>(i,j);
						float minval = std::numeric_limits<float>::max();
						float maxval = std::numeric_limits<float>::min();

						int lc = 0;
						int mc = 0;
						int ec = 0;
						int counter = 0;
						float val_c = 0;

						for (int k = -1; k <= 1; k++){
							for (int l = -1; l <= 1; l++){
								val_c = prev_data[idx + (k * curCols + l)] - val;
								if (val_c > 0) counter += 1;
								else if (val_c < 0) counter -= 1;

								val_c = curr_data[idx + (k * curCols + l)] - val;
								if (val_c > 0) counter += 1;
								else if (val_c < 0) counter -= 1;

								val_c = next_data[idx + (k * curCols + l)] - val;
								if (val_c > 0) counter += 1;
								else if (val_c < 0) counter -= 1;
							}
						}

						/*for (int ii = i - 1; ii <= i + 1; ii++){
							for (int jj = j - 1; jj <= j + 1; jj++){
								int idy = (ii * curCols + jj);
								//float tval = prev.ptr<float>(ii)[jj];
								float tval = prev.at<float>(ii,jj);
								if (tval < minval){ minval = tval; }
								else if (tval > maxval){ maxval = tval; }

								if (tval < val){ lc++; }
								else if (tval > val){ mc++; }
								else if (tval == val){ ec++; }
							}
						}
						for (int ii = i - 1; ii <= i + 1; ii++){
							for (int jj = j - 1; jj <= j + 1; jj++){
								if (ii == i && jj == j)	//	Do not include current sample point, this is important
									continue;
								int idy = (ii * curCols + jj);
								//float tval = curr.ptr<float>(ii)[jj];
								float tval = curr.at<float>(ii, jj);
								if (tval < minval){ minval = tval; }
								else if (tval > maxval){ maxval = tval; }

								if (tval < val){ lc++; }
								else if (tval > val){ mc++; }
								else if (tval == val){ ec++; }
							}
						}
						for (int ii = i - 1; ii <= i + 1; ii++){
							for (int jj = j - 1; jj <= j + 1; jj++){
								int idy = (ii * curCols + jj);
								//float tval = next.ptr<float>(ii)[jj];
								float tval = next.at<float>(ii, jj);
								if (tval < minval){ minval = tval; }
								else if (tval > maxval){ maxval = tval; }

								if (tval < val){ lc++; }
								else if (tval > val){ mc++; }
								else if (tval == val){ ec++; }
							}
						}*/

						//if (val < additive){
							//printf("Val: %f\n", val);
							//getchar();
						//}
						//if (val < minval || val > maxval){
						//if ((lc == 26 || mc == 26)){
						if (abs(counter) == 26) {
							keypoint newKey(i, j, oct, 0, step);
							newKey.scale = temp_scale;
							keys.push_back(newKey);
							key_count++;
							//printf("min: %f, val: %f, max: %f\n", minval, val, maxval);
							//printf("lc: %d, ec: %d, mc: %d\n",lc,ec,mc);
						}
						//else{
							
						//}

					}
				}
			}

			int offset = 0;
			//curRows = srcRows;
			//curCols = srcCols;

			//	Build full Difference-of-Gaussian image for presentation.  Not part of true algorithm, but good for discussions.
			/*if (full_dog == 1){
				for (int i = 0; i < curRows; i++){
					for (int j = 0; j < curCols; j++){
						for (int color = 0; color < 3; color++){
							int idx_i = 3 * (i * curCols + j) + color;
							int idx_o = 3 * (i * destCols + (j + offset)) + color;

							dest_data[idx_o] = src_data[idx_i];
						}
					}
				}

				offset += curCols;

				for (int step = 0; step < s - 1; step++){

					cv::Mat dog = dog_img[step];
					uchar* dog_data = (uchar*)dog.datastart;

					for (int i = 0; i < curRows; i++){
						for (int j = 0; j < curCols; j++){
							int idx_i = 3 * (i * curCols + j);
							int idx_o = 3 * (i * destCols + (j + offset));
							
							for (int color = 0; color < 3; color++){
								dest_data[idx_o + color] = dog_data[idx_i + color];
							}
						}
					}
					offset += curCols;

				}
				dest_data += 3 * curRows * destCols;
			}*/

			//image = blur_img[s - 2];//CHANGED
			//image = directResize(image, image.rows / 2, image.cols / 2);//CHANGED
			curRows = curRows / 2;
			curCols = curCols / 2;
			image = fdirectResize(image, image.rows / 2, image.cols / 2);
			src_data = (float*)image.datastart;

			dog_oct.push_back(dog_img);
			blur_oct.push_back(blur_img);
			gauss_exp -= 2;
		}	//	End of main work loop

		//printf("Keys Size: %d\n", key_count);
		printf("Keys Size: %d\n", keys.size());
		//mySiftWriteKeyFile(keys);

		//return dog_oct[0][0];

		//printf("Mark 1\n");
		//printf("Keypoints: %d\n", key_count);
		printf("Bogeys: %d\n", bogeys);

		dest_data = (uchar*)output.datastart;


		//Edge Responses goes here
		mySiftEdgeResponses(dog_oct, keys);

		int unfiltered = 0;
		std::vector<keypoint>::iterator iter;
		for (iter = keys.begin(); iter != keys.end();){
			if ((*iter).filtered){
				iter = keys.erase(iter);
			}
			else{
				unfiltered++;
				iter++;
			}
		}

		key_count = keys.size();

		printf("Unfiltered: %d\n", unfiltered);
		mySiftWriteKeyFile(keys);

		return dog_oct[0][0];

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

		//Orientation assignment
		int W = 2 * region + 1;
		double* gKernel = new double[W * W];
		createFilter(gKernel, 1.5, region);

		int key_count_new = key_count;
		key_index = 0;
		while (key_index < key_count){
			keypoint& key_now = keys[key_index];
			if (key_now.filtered == true){
				key_index++;
				continue;
			}

			int idx = key_now.idx, idy = key_now.idy;
			int oct = key_now.oct, kindex = (int)key_now.index;

			int curRows = srcRows / pow(2, oct);
			int curCols = srcCols / pow(2, oct);

			float* or_mag = or_mag_oct[oct][kindex];
			//cv::Mat blur_img = blur_oct[oct][scale];
			int maxk = std::min(region, curRows - idx - 1);
			int mink = std::min(region, idx);
			int maxl = std::min(region, srcCols - idy - 1);
			int minl = std::min(region, idy);

			float histo[36];
			for (int b = 0; b < 36; b++){
				histo[b] = 0;
			}

			for (int i = -mink; i < maxk; i++){
				for (int j = -minl; j < maxl; j++){
					if (eucDistance(idx, idy, idx + i, idy + j) > region ){
						//printf("Skip!\n");
						continue;
					}
					int id_or_mag = 2 * ((idx + i) * curCols + (idy + j));
					int id_gKernel = (i + region) * W + (j + region);

					//printf("Or: %f\n", or_mag[id_or_mag]);
					int bin = (int)((180.0 / (M_PI * 10.0)) * or_mag[id_or_mag]);
					//printf("Bin: %d\n", bin);
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

			for (int bin = 0; bin < 36; bin++){
				if (bin != max_bin && histo[bin] >= 0.8 * max_hist){
					keypoint newKey(idx, idy, oct, bin * ((M_PI * 10.0) / 180.0), kindex);
					keys.push_back(newKey);
					key_count_new++;
				}
			}

			
			keys[key_index].angle = max_bin * ((M_PI * 10.0) / 180.0);

			key_index++;
		}

		delete[] gKernel;

		mySiftDescriptors(keys, blur_oct, or_mag_oct);
		

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


		printf("\n");
		std::vector<keypoint> temp_keys;
		std::string file_name = "D://School//Summer 2016//Research//mySift//keys_in.txt";

		if (!mySiftReadKeyFile(temp_keys, file_name)){
			printf("Error reading given file: %s\n", file_name);
			return output;
		}

		//kd_node* kd_final = mySiftKDSearch(fish,keys,temp_keys[0]);
		//kd_node* kd_final = mySiftKDIterSearch(&fish, keys, temp_keys[0]);
		

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
				cv::circle(output, cv::Point(idx + col_off, idy + row_off), mark_dim, cv::Scalar(0, 0, 255), 2, 8, 0);

				key_index++;
			}
		}
		//	Regular image keypoint indicator drawing
		else {
			output = original;
			key_index = 0;
			while (key_index < key_count){
				keypoint& key_now = keys[key_index];
				if (key_now.filtered){
					key_index++;
					continue;
				}

				int idx = key_now.idy, idy = key_now.idx;
				int oct = key_now.oct, kindex = (int)key_now.index;

				float mult = pow(2, oct - 1);

				//cv::circle(output, cv::Point(idx * mult, idy * mult), mark_dim, cv::Scalar(0, 0, 255), 2, 8, 0);

				//printf("Angle: %f\n",key_now.angle);
				float size = 10 + (5 * mark_dim * ((2 * oct) * (k * (kindex -1))));
				int newi = (idx * mult) + (sin(key_now.angle) * size);
				int newj = (idy * mult) + (cos(key_now.angle) * size);

				//if (newi >= 0 && newi <= srcRows / 2 - 1 && newj >= 0 && newj <= srcCols / 2 - 1){
				cv::arrowedLine(output, cv::Point(idx * mult, idy * mult), cv::Point(newi, newj), cv::Scalar(0, 0, 255), 1, 8, 0);
				//}


				key_index++;
			}
		}


		//printf("Keys: %d\n", key_count);

		for (int oct = 0; oct < octaves; oct++){
			for (int step = 0; step < s; step++){
				delete[] or_mag_oct[oct][step];
			}
		}

		return output;
	}

	void mySiftEdgeResponses(std::vector<std::vector<cv::Mat>>& dog_oct, std::vector<keypoint>& keys){
		float r = 10;
		float t = std::pow(r + 1, 2) / r;

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

			//printf("Point %d: dx: %f; dy: %f; ds: %f\n", key_index, tptr[0], tptr[1], tptr[2]);

			cv::Mat neg_soDer = soDer * -1;

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
			cv::Mat ems = cv::Mat::zeros(3, 3, CV_32F);
			for (int a = 0; a < 3; a++){
				for (int b = 0; b < 3; b++){
					int box = 0;
					for (int i = 0; i < 3; i++){
						if (i == a) continue;
						for (int j = 0; j < 3; j++){
							if (j == b) continue;
							vals[box] = neg_soDer.at<float>(i, j);
							box++;
						}
					}
					ems.at<float>(a, b) = ((vals[0] * vals[3]) - (vals[1] * vals[2])) * pow(-1, 3 * a + b) / det;
				}
			}
			cv::Mat emt = cv::Mat::zeros(3, 3, CV_32F);
			for (int a = 0; a < 3; a++){	// Transpose
				for (int b = 0; b < 3; b++){
					emt.at<float>(a, b) = ems.at<float>(b, a);
				}
			}

			cv::Mat extreme = emt * foDer;

			float* exptr = (float*)extreme.datastart;
			float mult = 1.0;

			//printf("Ex %d: %f, %f, %f\n", key_index, exptr[0], exptr[1], exptr[2]);
			if ( abs(exptr[0]) > 0.5 || abs(exptr[1]) > 0.5 || abs(exptr[2]) > 0.5 ){
				key_now.filtered = true;
				key_index++;
				continue;
			}

			float ex_val = 0.0;
			for (int i = 0; i < 3; i++){
				ex_val += tptr[i] * exptr[i];
			}
			ex_val *= 0.5;
			ex_val += dog_oct[key_now.oct][(int)key_now.index].at<float>(key_now.idx,key_now.idy);
			if (abs(ex_val) < 0.03){
				//printf("ex_val: %f\n", abs(ex_val));	//Fix Later
				key_now.filtered = true;
				key_index++;
				continue;
			}

			float h_trace = dxx + dyy;
			float h_det = dxx * dyy - pow(dxy, 2);

			if (h_det <= 0 || pow(h_trace,2) / h_det > t){
				key_now.filtered = true;
			}

			key_index++;
		}
	}


	//void mySiftDescriptors
	void mySiftDescriptors(std::vector<keypoint>& keys, std::vector<std::vector<cv::Mat>>& blur_oct, std::vector<std::vector<float*>>& or_mag_oct){
		int region = 8;
		int key_count = keys.size();

		for (int key_index = 0; key_index < key_count; key_index++){

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
			return idx * cols + idy;
		}
		return 0;
	}

	cv::Mat mySift_foDer(std::vector<cv::Mat>& neighbors, int px, int py){
		cv::Mat result = cv::Mat::zeros(3, 1, CV_32FC1);
		//printf("Test: %d\n", neighbors[1].at<cv::Vec3b>(px - 1, py)[0]);
		int rows = neighbors[1].rows, cols = neighbors[1].cols;
		float* cur_ptr = (float*)neighbors[1].datastart;
		float* pre_ptr = (float*)neighbors[0].datastart;
		float* nex_ptr = (float*)neighbors[2].datastart;

		const float dx = ((float)cur_ptr[index(px - 1, py, cols, 0)] - (float)cur_ptr[index(px + 1, py, cols, 0)]) / 2.0;
		const float dy = ((float)cur_ptr[index(px, py - 1, cols, 0)] - (float)cur_ptr[index(px, py + 1, cols, 0)]) / 2.0;
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
		return result;
	}

	cv::Mat mySift_soDer(std::vector<cv::Mat>& neighbors, int px, int py){
		cv::Mat result = cv::Mat::zeros(3, 3, CV_32FC1);

		int rows = neighbors[1].rows, cols = neighbors[1].cols;
		float* cur_ptr = (float*)neighbors[1].datastart;
		float* pre_ptr = (float*)neighbors[0].datastart;
		float* nex_ptr = (float*)neighbors[2].datastart;

		const float dxx = (float)(cur_ptr[index(px + 1, py, cols, 0)] + cur_ptr[index(px - 1, py, cols, 0)]) - 2.0 * cur_ptr[index(px, py, cols, 0)];
		const float dyy = (float)(cur_ptr[index(px, py + 1, cols, 0)] + cur_ptr[index(px, py - 1, cols, 0)]) - 2.0 * cur_ptr[index(px, py, cols, 0)];
		const float dss = (float)(nex_ptr[index(px, py, cols, 0)] + pre_ptr[index(px, py, cols, 0)]) - 2.0 * cur_ptr[index(px, py, cols, 0)];
		const float dxy = (float)(cur_ptr[index(px + 1, py + 1, cols, 0)] - cur_ptr[index(px - 1, py + 1, cols, 0)] - cur_ptr[index(px + 1, py - 1, cols, 0)] + cur_ptr[index(px - 1, py - 1, cols, 0)]) / 2.0;
		const float dxs = (float)(nex_ptr[index(px + 1, py, cols, 0)] - nex_ptr[index(px - 1, py, cols, 0)] - pre_ptr[index(px + 1, py, cols, 0)] + pre_ptr[index(px - 1, py, cols, 0)]) / 2.0;
		const float dys = (float)(nex_ptr[index(px, py + 1, cols, 0)] - nex_ptr[index(px, py - 1, cols, 0)] - pre_ptr[index(px, py + 1, cols, 0)] + pre_ptr[index(px, py - 1, cols, 0)]) / 2.0;

		//const float dxx = neighbors[1].at<cv::Vec3b>(px + 1, py)[0] + neighbors[1].at<cv::Vec3b>(px - 1, py)[0] - 2.0 * neighbors[1].at<cv::Vec3b>(px, py)[0];
		//const float dyy = neighbors[1].at<cv::Vec3b>(px, py + 1)[0] + neighbors[1].at<cv::Vec3b>(px, py - 1)[0] - 2.0 * neighbors[1].at<cv::Vec3b>(px, py)[0];
		//const float dss = neighbors[2].at<cv::Vec3b>(px, py)[0] + neighbors[0].at<cv::Vec3b>(px, py)[0] - 2.0 * neighbors[1].at<cv::Vec3b>(px, py)[0];
		//const float dxy = (neighbors[1].at<cv::Vec3b>(px + 1, py + 1)[0] - neighbors[1].at<cv::Vec3b>(px - 1, py + 1)[0] - neighbors[1].at<cv::Vec3b>(px + 1, py - 1)[0] + neighbors[1].at<cv::Vec3b>(px - 1, py - 1)[0]) / 2;
		//const float dxs = (neighbors[2].at<cv::Vec3b>(px + 1, py)[0] - neighbors[2].at<cv::Vec3b>(px - 1, py)[0] - neighbors[0].at<cv::Vec3b>(px + 1, py)[0] + neighbors[0].at<cv::Vec3b>(px - 1, py)[0]) / 2;
		//const float dys = (neighbors[2].at<cv::Vec3b>(px, py + 1)[0] - neighbors[2].at<cv::Vec3b>(px, py - 1)[0] - neighbors[0].at<cv::Vec3b>(px, py + 1)[0] + neighbors[0].at<cv::Vec3b>(px, py - 1)[0]) / 2;

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

		//result.at<float>(0, 0) = dxx;
		//result.at<float>(1, 0) = dxy;
		//result.at<float>(2, 0) = dxs;
		//result.at<float>(0, 1) = dxy;
		//result.at<float>(1, 1) = dyy;
		//result.at<float>(2, 1) = dys;
		//result.at<float>(0, 2) = dxs;
		//result.at<float>(1, 2) = dys;
		//result.at<float>(2, 2) = dss;

		//printf("soDer: \n");
		//printf("[ %f, %f, %f]\n", dxx, dxy, dxs);
		//printf("[ %f, %f, %f]\n", dxy, dyy, dys);
		//printf("[ %f, %f, %f]\n", dxs, dys, dss);
		//getchar();

		return result;
	}

	bool mySiftWriteKeyFile(std::vector<keypoint>& keys){
		std::ofstream key_file;
		key_file.open("D://School//Summer 2016//Research//gray//keys_cpu.txt");

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

	kd_node mySiftKDHelp(std::vector<keypoint>& keys){
		std::string dims = "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

		int count = 0;

		kd_node* curr = mySiftKDTree(keys, keys.begin(), keys.end(), dims, count);
		curr->parent = 0;
		//printf("Count: %d\n", count);

		return *(curr);
	}

	kd_node* mySiftKDTree(std::vector<keypoint>& keys, std::vector<keypoint>::iterator front, std::vector<keypoint>::iterator back, std::string dims, int& count){
		kd_node* curr = new kd_node;
		curr->leaf_begin = front;
		curr->leaf_end = back;

		if (std::distance(front, back) <= 1 || dims.find('0') == std::string::npos){
			curr->leaf = true;
			count += 1;
			return curr;
		}

		int dist = std::distance(front, back);

		float* data_mean = new float[128];
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
		}

		int dim = 0;
		float max_diff = 0.0;

		for (int i = 0; i < 128; i++){
			if (dims[i] == '1')
				continue;
			if (data_var[i] >= max_diff){
				dim = i;
				max_diff = data_var[i];
			}
		}

		dims[dim] = '1';
		curr->dim = dim;
		delete[] data_mean;
		delete[] data_var;

		mySiftKDQuicksort(keys, front, back, dim);
		int middle = std::distance(front, back) / 2;
		curr->median = (*(front + middle)).descriptors[dim];

		//printf("Left: %d, Right: %d\n", std::distance(front, front + middle), std::distance(front + middle, back));
		//getchar();

		curr->left = mySiftKDTree(keys, front, front + middle, dims, count);
		curr->left->parent = curr;
		curr->right = mySiftKDTree(keys, front + middle, back, dims, count);
		curr->right->parent = curr;

		//if (curr.left->leaf == false){
		//	printf("Test: %d\n", curr.left->dim);
		//}

		return curr;
	}

	void mySiftKDQuicksort(std::vector<keypoint>& keys, std::vector<keypoint>::iterator front, std::vector<keypoint>::iterator back, int dim){
		if (std::distance(front,back) <= 1){
			return;
		}
		std::vector<keypoint>::iterator pivot = back - 1;
		std::vector<keypoint>::iterator wall = front;
		std::vector<keypoint>::iterator current = front;
		while (current != pivot){
			if ((*current).descriptors[dim] < (*pivot).descriptors[dim]){
				std::iter_swap(current, wall);
				wall++;
			}
			current++;
		}
		std::iter_swap(pivot, wall);

		mySiftKDQuicksort(keys, front, wall, dim);
		mySiftKDQuicksort(keys, wall + 1, back, dim);
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
					data_1[idx] = data_2[idx];
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

}