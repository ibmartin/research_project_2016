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

class keypoint
{
public:
	int idx, idy, oct, index;
	float angle, scale, mag, posx, posy;
	bool filtered = false;
	std::vector<float> descriptors;
	keypoint(int _idx, int _idy, int _oct, float _angle, int _index){	//Constuctor
		idx = _idx;
		idy = _idy;
		oct = _oct;
		angle = _angle;
		index = _index;
	}
	keypoint(){

	}
};

class kd_node
{
public:
	bool leaf = false;
	int dim = 0;
	float median = 0;

	kd_node* parent;
	kd_node* left;
	kd_node* right;
	std::vector<keypoint>::iterator leaf_begin;
	std::vector<keypoint>::iterator leaf_end;
};

float eucDistance(float x1, float y1, float x2, float y2){
	return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

Mat rgb2Gray(Mat image){
	get_nvtAttrib("rgb2Gray GPU", 0xFF880000);

	get_nvtAttrib("Setup", 0xFF222222);
	Mat out = Mat(image.rows, image.cols, CV_8UC1);

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	nvtxRangePop();

	cudaRgb2Gray(input, output, image.rows, image.cols);

	nvtxRangePop();
	return out;
}

Mat frgb2Gray(Mat image){
	get_nvtAttrib("frgb2Gray GPU", 0xFF880000);
	Mat out = Mat(image.rows, image.cols, CV_32FC1);

	unsigned char* input = (unsigned char*)image.datastart;
	float* output = (float*)out.datastart;
	cudafRgb2Gray(input, output, image.rows, image.cols);

	nvtxRangePop();
	return out;
}

Mat reverse(Mat image){
	get_nvtAttrib("reverse GPU", 0xFF880000);
	Mat out = Mat(image.rows, image.cols, image.type());

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaReverse(input, output, image.rows, image.cols);

	nvtxRangePop();
	return out;
}

Mat gammaCorrection(Mat image, double gamma){
	get_nvtAttrib("gammaCorrection GPU", 0xFF880000);
	get_nvtAttrib("Setup Outer", 0xFF000088);
	Mat out = Mat(image.rows, image.cols, image.type());

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	nvtxRangePop();
	cudaGammaCorrection(input, output, gamma, image.rows, image.cols);

	nvtxRangePop();
	return out;
}

Mat directResize(Mat image, int rows, int cols){
	get_nvtAttrib("directResize GPU", 0xFF880000);
	Mat out = Mat(rows, cols, image.type());

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaDirectResize(input, output, image.rows, image.cols, rows, cols);

	nvtxRangePop();
	return out;
}

Mat fdirectResize(Mat image, int rows, int cols){
	get_nvtAttrib("fdirectResize GPU", 0xFF880000);
	Mat out = Mat::zeros(rows, cols, CV_32FC1);
	float* input = (float*)image.datastart;
	float* output = (float*)out.datastart;
	cudafDirectResize(input, output, image.rows, image.cols, rows, cols);

	nvtxRangePop();
	return out;
}

Mat linearResize(Mat image, int rows, int cols){
	get_nvtAttrib("linearResize GPU", 0xFF880000);
	Mat out = Mat(rows, cols, image.type());

	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaLinearResize(input, output, image.rows, image.cols, rows, cols);

	nvtxRangePop();
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
	get_nvtAttrib("createFilter GPU", 0xFF880000);
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

	nvtxRangePop();
}

Mat makeFilter(float sigma){

	int dim = floor(6 * sigma);
	Mat outFilter(1, dim, CV_32FC1);

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

	return outFilter;

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

Mat gaussianFilter(Mat image, double sigma){
	get_nvtAttrib("gaussianFilter GPU", 0xFF880000);
	int W = 2 * FILTER_SIZE + 1;
	double* gKernel = new double[W * W];
	createFilter(gKernel, sigma, FILTER_SIZE);

	Mat out = Mat(image.rows, image.cols, image.type());
	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaGaussianFilter(input, output, gKernel, image.rows, image.cols);

	//delete[] gKernel;

	nvtxRangePop();
	return out;
	//return image;
}

Mat fGaussianFilter(Mat image, double sigma){
	get_nvtAttrib("fGaussianFilter GPU", 0xFF880000);
	int W = 2 * FILTER_SIZE + 1;
	double* gKernel = new double[W * W];
	createFilter(gKernel, sigma, FILTER_SIZE);

	Mat out = Mat::zeros(image.rows, image.cols, CV_32FC1);
	float* input = (float*)image.datastart;
	float* output = (float*)out.datastart;
	cudafGaussianFilter(input, output, gKernel, image.rows, image.cols);

	nvtxRangePop();
	return out;

}

Mat myConv2(Mat large, Mat small, int mode = 0){
	if (small.rows > large.rows || small.cols > large.cols){
		return large;
	}
	int rows = large.rows + small.rows - 1;
	int cols = large.cols + small.cols - 1;
	Mat temp = Mat::zeros(rows, cols, CV_32FC1);
	float* tempP = (float*)temp.datastart;
	float* largeP = (float*)large.datastart;
	float* smallP = (float*)small.datastart;
	cudaMyConv2(tempP, largeP, smallP, temp.rows, temp.cols, large.rows, large.cols, small.rows, small.cols);

	if (mode == 0){
		int top = small.rows / 2, bottom = (small.rows - 1) / 2, left = small.cols / 2, right = (small.cols - 1) / 2;
		return myTrim(temp, top, bottom, left, right);
	}

	return temp;
}

cv::Mat fGaussianFilterSep(cv::Mat image, float sigma){
	int srcRows = image.rows, srcCols = image.cols;

	cv::Mat output = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
	cv::Mat temp = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);

	cv::Mat filter = makeFilter(sigma);
	//cout << filter << endl;
	cv::Mat tfilter = myTranspose(filter);
	//cout << tfilter << endl;

	//temp = image;
	temp = myConv2(image, filter, 0);
	output = myConv2(temp, tfilter, 0);
	//output = temp;

	return output;
}

Mat sobelFilter(Mat image){
	get_nvtAttrib("sobelFilter GPU", 0xFF880000);
	get_nvtAttrib("Setup outer", 0xFF000088);
	Mat out(image.rows, image.cols, image.type());
	//Mat temp(image.rows, image.cols, image.type());
	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	nvtxRangePop();
	cudaSobelFilter(input, output, image.rows, image.cols);

	nvtxRangePop();
	return out;
}

Mat kMeans(Mat image, int k_means){
	get_nvtAttrib("kMeans GPU", 0xFF880000);
	srand(2000);
	if (k_means > 256){
		printf("Error: Max number of groups exceeded (256)\n");
		exit(-1);
	}
	Mat out(image.rows, image.cols, image.type());
	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaKMeans(input, output, image.rows, image.cols, k_means);

	nvtxRangePop();
	return out;
}

Mat kMeansFixed(Mat image, int k_means){
	get_nvtAttrib("kMeans GPU", 0xFF880000);
	srand(2000);
	if (k_means > 256){
		printf("Error: Max number of groups exceeded (256)\n");
		exit(-1);
	}
	Mat out(image.rows, image.cols, image.type());
	uchar* input = (uchar*)image.datastart;
	uchar* output = (uchar*)out.datastart;
	cudaKMeansFixed(input, output, image.rows, image.cols, k_means);

	nvtxRangePop();
	return out;
}

Mat gaussianPyramid(cv::Mat image, uchar levels, float scale){
	get_nvtAttrib("gaussianPyramid GPU", 0xFF880000);
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

	nvtxRangePop();
	return output;
}

bool mySiftWriteKeyFile(std::vector<keypoint>& keys){
	std::ofstream key_file;
	key_file.open("D://School//Summer 2016//Research//mySift//keys_gpu.txt");

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
	int rows = neighbors[1].rows, cols = neighbors[1].cols;
	float* cur_ptr = (float*)neighbors[1].datastart;
	float* pre_ptr = (float*)neighbors[0].datastart;
	float* nex_ptr = (float*)neighbors[2].datastart;

	const float dx = ((float)cur_ptr[index(px - 1, py, cols, 0)] - (float)cur_ptr[index(px + 1, py, cols, 0)]) / 2.0;
	const float dy = ((float)cur_ptr[index(px, py - 1, cols, 0)] - (float)cur_ptr[index(px, py + 1, cols, 0)]) / 2.0;
	const float ds = ((float)pre_ptr[index(px, py, cols, 0)] - (float)nex_ptr[index(px, py, cols, 0)]) / 2.0;

	float* res_ptr = (float*)result.datastart;
	res_ptr[0] = dx;
	res_ptr[1] = dy;
	res_ptr[2] = ds;

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

	return result;
}

void mySiftEdgeResponses(std::vector<std::vector<cv::Mat>>& dog_oct, std::vector<keypoint>& keys){
	float r = 10;
	float t = std::pow(r + 1, 2) / r;
	//get_nvtAttrib("mySiftEdgeResponses", 0xFF0000FF);

	int key_count = keys.size();
	int key_index = 0;

	while (key_index < key_count){
		keypoint& key_now = keys[key_index];
		std::vector<cv::Mat> neighbors;
		neighbors.push_back(dog_oct[key_now.oct][(int)key_now.index - 1]);
		neighbors.push_back(dog_oct[key_now.oct][(int)key_now.index]);
		neighbors.push_back(dog_oct[key_now.oct][(int)key_now.index + 1]);
		//printf("Key: %d, idx: %d, idy: %d, oct: %d\n", key_index, key_now.idx, key_now.idy, key_now.oct);

		cv::Mat foDer = mySift_foDer(neighbors, key_now.idx, key_now.idy);
		cv::Mat soDer = mySift_soDer(neighbors, key_now.idx, key_now.idy);

		//cv::Mat neg_soDer = soDer * -1;
		//cv::Mat neg_2Der;// = soDer * -1.0;
		//neg_2Der = soDer * -1.0;

		float det = 0;	//Find Det(soDer)
		det += soDer.at<float>(0, 0) * ((soDer.at<float>(1, 1) * soDer.at<float>(2, 2)) - (soDer.at<float>(2, 1) * soDer.at<float>(1, 2)));
		det -= soDer.at<float>(0, 1) * ((soDer.at<float>(1, 0) * soDer.at<float>(2, 2)) - (soDer.at<float>(2, 0) * soDer.at<float>(1, 2)));
		det += soDer.at<float>(0, 2) * ((soDer.at<float>(1, 0) * soDer.at<float>(2, 1)) - (soDer.at<float>(2, 0) * soDer.at<float>(1, 1)));

		if (det == 0){
			keys[key_index].filtered = true;
			//key_now.filtered = true;
			key_index++;
			continue;
		}

		const float dxx = soDer.at<float>(0, 0);
		const float dyy = soDer.at<float>(1, 1);
		const float dxy = soDer.at<float>(0, 1);
		float* tptr = (float*)foDer.datastart;

		float vals[4];
		cv::Mat ems = cv::Mat::zeros(3, 3, CV_32F);
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
		}
		cv::Mat emt = cv::Mat::zeros(3, 3, CV_32F);
		for (int a = 0; a < 3; a++){	// Transpose
			for (int b = 0; b < 3; b++){
				emt.at<float>(a, b) = ems.at<float>(b, a);
			}
		}

		float expt0 = (emt.at<float>(0, 0) * foDer.at<float>(0, 0)) + (emt.at<float>(0, 1) * foDer.at<float>(1, 0)) + (emt.at<float>(0, 2) * foDer.at<float>(2, 0));
		float expt1 = (emt.at<float>(1, 0) * foDer.at<float>(0, 0)) + (emt.at<float>(1, 1) * foDer.at<float>(1, 0)) + (emt.at<float>(1, 2) * foDer.at<float>(2, 0));
		float expt2 = (emt.at<float>(2, 0) * foDer.at<float>(0, 0)) + (emt.at<float>(2, 1) * foDer.at<float>(1, 0)) + (emt.at<float>(2, 2) * foDer.at<float>(2, 0));

		//cv::Mat extreme = emt * foDer;

		//float* exptr = (float*)extreme.datastart;

		if (abs(expt0) > 0.5 || abs(expt1) > 0.5 || abs(expt2) > 0.5){
			keys[key_index].filtered = true;
			//key_now.filtered = true;
			key_index++;
			continue;
		}

		float ex_val = 0.0;
		//for (int i = 0; i < 3; i++){
		//	ex_val += tptr[i] * exptr[i];
		//}
		ex_val += tptr[0] * expt0;
		ex_val += tptr[1] * expt1;
		ex_val += tptr[2] * expt2;

		ex_val *= 0.5;
		ex_val += dog_oct[key_now.oct][(int)key_now.index].at<float>(key_now.idx, key_now.idy);
		if (abs(ex_val) < 0.03){
			//printf("ex_val: %f\n", abs(ex_val));	//Fix Later
			keys[key_index].filtered = true;
			//key_now.filtered = true;
			key_index++;
			continue;
		}

		float h_trace = dxx + dyy;
		float h_det = dxx * dyy - pow(dxy, 2);

		if (h_det <= 0 || pow(h_trace, 2) / h_det > t){
			keys[key_index].filtered = true;
			//key_now.filtered = true;
		}

		key_index++;
	}

	//nvtxRangePop();

}

void mySiftEdgeResponsesNew(std::vector<std::vector<cv::Mat>>& dog_oct, std::vector<keypoint>& keys){
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	float data[4] = { -1, -1, 1, 1 };
	cv::Mat der_y = cv::Mat(2, 2, CV_32FC1, data);
	//cout << der_y << endl;
	cv::Mat sder_y = myConv2(der_y, der_y, 1);
	//cout << sder_y << endl;
	printf("Edges New\n");
	float r = 10;

	for (int x = 0; x < dog_oct.size(); x++){
		printf("  Oct %d\n", x);
		for (int y = 0; y < dog_oct[x].size(); y++){
			//printf("    Level %d\n", y);

			cv::Mat test = dog_oct[x][y];
			//cv::imwrite("D://School//Summer 2016//Research//mySift//parrot_test_" + std::to_string((x + 1) * 10 + (y)) + "_g.png", dog_oct[x][y], compression_params);

			cv::Mat temp = myElemMat(myElemScalar(myConv2(test, sder_y, 0), -1, -1), myConv2(test, der_y, 0), 0);

			dog_oct[x][y] = myElemMat(myElemScalar(myElemMat(temp, myConv2(myTranspose(test), der_y, 0), 0), 0.5, 0), test, 1);
			//dog_oct[x][y] = myElemMat(myElemScalar(myElemMat(temp, myConv2(test, myTranspose(der_y), 0), 0), 0.5, 0), test, 1);

			//cv::imwrite("D://School//Summer 2016//Research//mySift//parrot_test_" + std::to_string((x + 1) * 10 + (y)) + "_zg.png", dog_oct[x][y], compression_params);
			int levels = dog_oct[x].size();

		}
	}

	for (int key_idx = 0; key_idx < keys.size(); key_idx++){
		keypoint& cur_key = keys[key_idx];
		int index = cur_key.index;
		int octave = cur_key.oct;
		int idx = cur_key.idx;
		int idy = cur_key.idy;

		if (abs(dog_oct[octave][index].at<float>(idx, idy)) < 7.65){
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

void mySiftOrAssign(std::vector<keypoint>& keys, std::vector<std::vector<float*>>& or_mag_oct, int srcRows, int srcCols){
	get_nvtAttrib("Or Assign", 0xFF888888);
	int region = REGION_SIZE;

	int W = 2 * region + 1;
	double* gKernel = new double[W * W];
	createFilter(gKernel, 1.5, region);

	int key_count = keys.size();

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

void mySiftDescriptors(std::vector<keypoint>& keys, std::vector<std::vector<cv::Mat>>& blur_oct, std::vector<std::vector<float*>>& or_mag_oct){
	int region = 8;
	int key_count = keys.size();
	//nvtxEventAttributes_t eventAttrib = get_nvtAttrib("mySiftDescriptors", 0xFF00FF00);
	nvtxEventAttributes_t eventAttrib = get_nvtAttrib("mySiftDescriptors [" + std::to_string(key_count) + " ]", 0xFF00FF00);
	//printf("Key Count: %d\n", key_count);

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

		//printf("  Fish 0 %d,%d\n", key_now.idx, key_now.idy);
		//printf("    Angle: %f\n", key_now.angle);

		int or_mag_size = 2 * curRows * curCols;
		int W = 2 * region + 1;
		float* orientations = new float[W * W];
		float* magnitudes = new float[W * W];
		double* gKernel = new double[W * W];
		createFilter(gKernel, 8.0, region);
		cv::Mat gauss_cur = current(cv::Rect(key_now.idy - region, key_now.idx - region, W, W));

		//printf("  Fish 1 %d,%d\n", key_now.idx, key_now.idy);

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

		//printf("  Fish 2 %d,%d\n", key_now.idx, key_now.idy);
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

unsigned int radixGetMax(unsigned int arr[], int n){
	unsigned int max = arr[0];
	for (int i = 1; i < n; i++){
		if (arr[i] > max) max = arr[i];
	}
	return max;
}

void mySiftKDCountSort(unsigned int* data, unsigned int* index, int d, int exp){

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

	//for (int exp = 1; max / exp > 0; exp *= 10){
	//	mySiftKDCountSort(data, index, d, exp);
	//}

	if (d > 64){
		get_nvtAttrib("Count Sort", 0xFF888888);
		cudaMySiftCountSort(data, index, d, max);
		nvtxRangePop();
	}
	else{
		//get_nvtAttrib("Select Sort", 0xFF0000FF);
		float temp_val = 0.0, min_val = 0.0;
		int temp_id = 0, min_id = 0;
		for (int wall = 0; wall < d - 1; wall++){
			min_val = temp[wall];
			min_id = wall;

			for (int itr = wall; itr < d; itr++){
				if (temp[itr] < min_val){
					min_val = temp[itr];
					min_id = itr;
				}
			}
			temp_val = temp[wall];
			temp_id = index[wall];

			temp[wall] = temp[min_id];
			index[wall] = index[min_id];

			temp[min_id] = temp_val;
			index[min_id] = temp_id;

		}
		//nvtxRangePop();
	}

	std::vector<keypoint> newVec(front, back);

	for (int i = 0; i < d; i++){
		*(front + i) = newVec[index[i]];
	}

	delete[] temp;
	delete[] index;
	cudaDeviceSynchronize();
}

kd_node* mySiftKDTree(std::vector<keypoint>& keys, std::vector<keypoint>::iterator front, std::vector<keypoint>::iterator back, std::string dims, float* all_var, int& count){
	kd_node* curr = new kd_node;
	curr->leaf_begin = front;
	curr->leaf_end = back;

	int dist = std::distance(front, back);

	if (dist <= 1 || dims.find('0') == std::string::npos){
		curr->leaf = true;
		count += 1;
		return curr;
	}

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

	dims[dim] = '1';
	curr->dim = dim;

	mySiftKDRadixSort(keys, front, back, dim);

	int middle = std::distance(front, back) / 2;
	curr->median = (*(front + middle)).descriptors[dim];

	curr->left = mySiftKDTree(keys, front, front + middle, dims, all_var, count);
	curr->left->parent = curr;
	curr->right = mySiftKDTree(keys, front + middle, back, dims, all_var, count);
	curr->right->parent = curr;

	return curr;

}

kd_node mySiftKDHelp(std::vector<keypoint>& keys){
	std::string dims = "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

	int count = 0;
	//printf("Starting the main tree\n");

	float all_var[128] = { 0 }, all_mean[128] = { 0 };
	std::vector<keypoint>::iterator front = keys.begin();
	std::vector<keypoint>::iterator back = keys.end();

	int dist = std::distance(front, back);
	for (std::vector<keypoint>::iterator iter = front; iter != back; iter++){
		if ((*iter).filtered == true) continue;
		for (int i = 0; i < 128; i++)
			all_mean[i] += (*iter).descriptors[i] / (float)dist;
	}
	printf("Fish 1\n");

	for (std::vector<keypoint>::iterator iter = front; iter != back; iter++){
		if ((*iter).filtered == true) continue;
		for (int i = 0; i < 128; i++)
			all_var[i] += (((*iter).descriptors[i] - all_mean[i]) * ((*iter).descriptors[i] - all_mean[i])) / (float)dist;
	}

	printf("Fish 2\n");

	kd_node* curr = mySiftKDTree(keys, keys.begin(), keys.end(), dims, all_var, count);
	curr->parent = 0;
	return *curr;
}

Mat mySift(Mat original){
	//Mat out;
	get_nvtAttrib("mySift GPU", 0xFF880000);

	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	std::string debug_Path = "D://School//Summer 2016//Research//mySift//";
	std::string img_name = "audrey";
	std::string ftype = ".png";

	//cv::Mat image = frgb2Gray(original);
	//image = fdirectResize(image, image.rows * 2, image.cols * 2);
	cv::Mat image = linearResize(original, original.rows * 2, original.cols * 2);
	image = frgb2Gray(image);
	cv::Mat temp_image = image;

	int printoff = 0;	//	Debug, used to print to console the values of all keypoints found by this function
	int full_dog = 0;	//	Set this to 1 to change the output image to a full representation of the difference-of-gaussians and scale space location of each keypoint
	int mark_dim = 1;	//	Determines size of circles in the output image.  Recommend setting to 10 or higher if full_dog is set to 1

	//	The first step is to scale up the input image by a factor of 2 in both dimensions, then apply a gaussian blur with sigma = 1.6
	//	Scaling up the input provides more keypoints and approximates a blur with sigma = 1.0, assuming the original image is roughly sigma = 0.5 which is the threshold for noise
	//	Sigma = 1.6 was experimentally determined by Lowe to give the best results.  Refer to the 2004 SIFT paper, pages 9 and 10 for discussion
	float sigma = 1.6;

	uchar scales = 3;
	uchar octaves = 3;
	int region = 4;
	int srcRows = image.rows;
	int srcCols = image.cols;

	float k = pow(2.0, (1.0 / (float)scales));
	int s = scales + 3;
	int slevel = s - 1;

	int curRows = srcRows, curCols = srcCols;

	std::vector<keypoint> keys;
	std::vector <std::vector<cv::Mat>> dog_oct;	//	Not currently used, meant to store DoG data long term
	std::vector <std::vector<cv::Mat>> blur_oct;

	int key_count = 0;
	int scale = 1;
	int key_index = 0;
	int bogeys = 0;
	int gauss_exp = 0;

	for (int oct = 0; oct < octaves; oct++){
		printf("Oct %d\n", oct);
		get_nvtAttrib("Octave " + to_string(oct), 0xFF888888);
		std::vector<cv::Mat> blur_img;
		std::vector<cv::Mat> dog_img;
		curRows = temp_image.rows;
		curCols = temp_image.cols;

		//printf("Oct: %d\n", oct);
		//cv::Mat current = fGaussianFilter(image, pow(k, gauss_exp) * sigma);
		cv::Mat current = temp_image;
		blur_img.push_back(current);

		get_nvtAttrib("Scale Space", 0xFF000088);
		for (int step = 1; step < s; step++){
			printf("  Step %d: ", step);
			//float temp_scale = sigma * pow(pow(sqrt(2.0), (1.0 / slevel)), oct * slevel + step);
			float temp_scale = sigma * pow(pow(2.0, (1.0 / slevel)), oct * slevel + step);
			printf(" %f\n", temp_scale);

			current = temp_image;
			//cv::Mat next = fGaussianFilter(image, pow(k, gauss_exp) * sigma);
			cv::Mat next = fGaussianFilterSep(temp_image, temp_scale);
			cv::Mat dog = cv::Mat::zeros(curRows, curCols, CV_32FC1);
			blur_img.push_back(next);

			float* curr_data = (float*)current.datastart;
			float* next_data = (float*)next.datastart;
			float* dog_data = (float*)dog.datastart;

			cudaMySiftDOG(curr_data, next_data, dog_data, curRows, curCols);

			//cv::imwrite("D://School//Summer 2016//Research//mySift//parrot_test_" + std::to_string((oct + 1) * 10 + (step)) + "_g.png", dog, compression_params);
			dog_img.push_back(dog);
			//current = next;
			gauss_exp++;
		}
		nvtxRangePop();

		get_nvtAttrib("Keypoints", 0xFF000088);
		for (int step = 1; step < s - 2; step++){
			int temp_exp = gauss_exp - s + (step);

			float temp_scale = ((pow(k, temp_exp) * sigma) - (pow(k, temp_exp - 1) * sigma)) / 2 + (pow(k, temp_exp - 1) * sigma);

			float* prev_data = (float*)dog_img[step - 1].datastart;
			float* curr_data = (float*)dog_img[step + 0].datastart;
			float* next_data = (float*)dog_img[step + 1].datastart;

			int bit_str_size = ceil((curRows * curCols) / 32.0);
			//bit_str_size = 1;
			unsigned int* key_str = new unsigned int[bit_str_size];
			for (int i = 0; i < bit_str_size; i++){
				key_str[i] = 0;
			}

			char* answers = new char[curRows * curCols];
			for (int iter = 0; iter < curRows * curCols; iter++){
				answers[iter] = 0;
			}

			//call
			cudaMySiftKeypoints(prev_data, curr_data, next_data, answers, key_str, curRows, curCols, bit_str_size);

			/*for (int key_block = 0; key_block < bit_str_size; key_block++){
				for (int entry = 0; entry < 32; entry++){
					if (key_str[key_block] & (unsigned int)powf(2, entry) != 0){
					//if (key_str[key_block] & (int)exp2(entry)){
						int id = (key_block * 32 + entry);
						int idx = id / curCols;
						int idy = id % curCols;
						keypoint newKey(idx, idy, oct, 0, step);
						newKey.scale = temp_scale;
						keys.push_back(newKey);
					}
				}
			}*/

			for (int i = 0; i < curRows; i++){
				for (int j = 0; j < curCols; j++){
					if (answers[i * curCols + j] == 1){
						int idx = (i * curCols + j) / curCols;
						int idy = (i * curCols + j) % curCols;
						keypoint newKey(idx, idy, oct, 0, step);
						//printf("    New Key, idx: %d, idy: %d, oct: %d, step: %d\n", idx, idy, oct, step);
						newKey.posy = ((float)i) / curRows;
						newKey.posx = ((float)j) / curCols;
						newKey.scale = temp_scale;
						keys.push_back(newKey);
						key_count++;
					}
				}
			}

			delete[] key_str;
			delete[] answers;

		}
		nvtxRangePop();

		cudaDeviceSynchronize();
		printf("  tRows: %d, tCols: %d\n", temp_image.rows, temp_image.cols);
		image = current;
		image = fdirectResize(image, image.rows / 2, image.cols / 2);
		temp_image = image;

		//src_data = (float*)image.datastart;
		//printf("Fish3\n");
		dog_oct.push_back(dog_img);
		blur_oct.push_back(blur_img);
		gauss_exp -= 2;
		nvtxRangePop();
	}
	cudaDeviceSynchronize();

	printf("Keys Size: %d\n", key_count);

	//mySiftWriteKeyFile(keys);

	//or_mag
	std::vector<std::vector<float*>> or_mag_oct;
	curRows = srcRows;
	curCols = srcCols;

	get_nvtAttrib("Or_Mag Calc", 0xFF888888);
	for (int oct = 0; oct < octaves; oct++){
		std::vector<float*> or_mag_current;
		curRows = blur_oct[oct][0].rows;  curCols = blur_oct[oct][0].cols;
		for (int step = 0; step < s; step++){

			cv::Mat& current = blur_oct[oct][step];

			float* curr_data = (float*)current.datastart;
			//float* curr_data = (float*)blur_oct[oct][step].datastart;
			float* or_mag = new float[2 * curRows * curCols];

			//cudaDeviceSynchronize();
			cudaMySiftOrMagGen(curr_data, or_mag, curRows, curCols);
			cudaDeviceSynchronize();
			//cudaTest(curRows, curCols);
			or_mag_current.push_back(or_mag);
		}
		or_mag_oct.push_back(or_mag_current);
	}

	nvtxRangePop();

	get_nvtAttrib("Key Culling " + std::to_string(keys.size()), 0xFF000088);
	//printf("Keys: %d\n", keys.size());

	printf("OrMagTest\n");
	cudaDeviceSynchronize();
	//cudaTest(original.rows, original.cols);
	//frgb2Gray(original);
	printf("OrMagTest End\n");

	//mySiftEdgeResponses(dog_oct, keys);
	mySiftEdgeResponsesNew(dog_oct, keys);
	cudaDeviceSynchronize();
	//cudaDeviceSynchronize();
	//cudaDeviceSynchronize();
	printf("Size before: %d\n", keys.size());
	mySiftKeyCull(keys);
	printf("Size after: %d\n", keys.size());

	//cudaTest(original.rows, original.cols);

	printf("Passed!\n");
	//getchar();

	/*int unfiltered = 0;
	std::vector<keypoint>::iterator iter;
	for (iter = keys.begin(); iter != keys.end();){
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

	//printf("Unfiltered: %d\n", unfiltered);
	nvtxRangePop();
	//mySiftWriteKeyFile(keys);

	cv::Mat output;

	cudaDeviceSynchronize();

	printf("OrAssign\n");
	mySiftOrAssign(keys, or_mag_oct, srcRows, srcCols);
	printf("OrAssign Done!\n");

	printf("Descriptors\n");
	mySiftDescriptors(keys, blur_oct, or_mag_oct);
	printf("Descriptors Done!\n");

	mySiftKeyCull(keys);
	key_count = keys.size();
	printf("Unfiltered: %d\n", key_count);
	

	
	printf("KDTree\n");
	get_nvtAttrib("KDTree", 0xFF000088);
	//kd_node fish = mySiftKDHelp(keys);
	//frgb2Gray(original);
	cudaDeviceSynchronize();
	nvtxRangePop();

	for (int oct = 0; oct < octaves; oct++){
		for (int step = 0; step < s; step++){
			delete[] or_mag_oct[oct][step];
		}
	}
	//return dog_oct[0][0];
	nvtxRangePop();
	//mySiftKeyCull(keys);
	
	nvtxRangePop();
	//mySiftWriteKeyFile(keys);
	//mySiftKeyCull(keys);
	//return original;

	if (full_dog == 1){
		int destRows = 4 * original.rows, destCols = (s) * (2 * original.cols);
		output = cv::Mat::zeros(destRows, destCols, CV_32FC1);
		float* dest_data = (float*)output.datastart;

		int roff = 0, coff = 0;
		for (int oct = 0; oct < octaves; oct++){
			curRows = dog_oct[oct][0].rows, curCols = dog_oct[oct][0].cols;
			float* curr_data = (float*)blur_oct[oct][0].datastart;
			for (int i = 0; i < curRows; i++){
				for (int j = 0; j < curCols; j++){
					dest_data[(i + roff) * destCols + (j + coff)] = curr_data[i * curCols + j];
				}
			}
			coff += curCols;
			for (int step = 0; step < s - 1; step++){
				curr_data = (float*)dog_oct[oct][step].datastart;
				
				for (int i = 0; i < curRows; i++){
					for (int j = 0; j < curCols; j++){
						dest_data[(i + roff) * destCols + (j + coff)] = curr_data[i * curCols + j];
					}
				}

				coff += curCols;
			}
			coff = 0;
			roff += curRows;
		}

	}
	else if (full_dog == 2){
		int destRows = 4 * original.rows, destCols = (s)* (2 * original.cols);
		output = cv::Mat::zeros(destRows, destCols, CV_32FC1);
		float* dest_data = (float*)output.datastart;

		int roff = 0, coff = 0;
		for (int oct = 0; oct < octaves; oct++){
			curRows = blur_oct[oct][0].rows, curCols = blur_oct[oct][0].cols;
			for (int step = 0; step < s; step++){
				float* curr_data = (float*)blur_oct[oct][step].datastart;

				for (int i = 0; i < curRows; i++){
					for (int j = 0; j < curCols; j++){
						dest_data[(i + roff) * destCols + (j + coff)] = curr_data[i * curCols + j];
					}
				}

				coff += curCols;
			}
			coff = 0;
			roff += curRows;
		}
	}
	else{
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
	
	return output;
}

#endif