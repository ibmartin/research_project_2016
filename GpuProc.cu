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
	float angle, scale, mag;
	bool filtered = false;
	std::vector<float> descriptors;
	keypoint(int _idx, int _idy, int _oct, float _angle, int _index){	//Constuctor
		idx = _idx;
		idy = _idy;
		oct = _oct;
		angle = _angle;
		index = _index;
	}
};

float eucDistance(float x1, float y1, float x2, float y2){
	return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

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
	cudafGaussianFilter(input, output, gKernel, image.rows, image.cols);

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

bool mySiftWriteKeyFile(std::vector<keypoint>& keys){
	std::ofstream key_file;
	key_file.open("D://School//Summer 2016//Research//gray//keys_gpu.txt");

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

		if (abs(exptr[0]) > 0.5 || abs(exptr[1]) > 0.5 || abs(exptr[2]) > 0.5){
			key_now.filtered = true;
			key_index++;
			continue;
		}

		float ex_val = 0.0;
		for (int i = 0; i < 3; i++){
			ex_val += tptr[i] * exptr[i];
		}
		ex_val *= 0.5;
		ex_val += dog_oct[key_now.oct][(int)key_now.index].at<float>(key_now.idx, key_now.idy);
		if (abs(ex_val) < 0.03){
			//printf("ex_val: %f\n", abs(ex_val));	//Fix Later
			key_now.filtered = true;
			key_index++;
			continue;
		}

		float h_trace = dxx + dyy;
		float h_det = dxx * dyy - pow(dxy, 2);

		if (h_det <= 0 || pow(h_trace, 2) / h_det > t){
			key_now.filtered = true;
		}

		key_index++;
	}

}

float mySiftVertParabola(float l_x, float l_y, float p_x, float p_y, float r_x, float r_y){

	return 0.0;
}

void mySiftOrAssign(std::vector<keypoint>& keys, std::vector<std::vector<float*>>& or_mag_oct, int srcRows, int srcCols){
	int region = REGION_SIZE;

	int W = 2 * region + 1;
	double* gKernel = new double[W * W];
	createFilter(gKernel, 1.5, region);

	int key_count = keys.size();

	//int key_index = 0;

	for (int key_index = 0; key_index < key_count; key_index++){
		keypoint& key_now = keys[key_index];
		if (key_now.filtered == true) continue;

		int idx = key_now.idx, idy = key_now.idy, oct = key_now.oct, kindex = (int)key_now.index;
		float scale = key_now.scale;

		int curRows = srcRows / exp2f(oct), curCols = srcCols / exp2f(oct);
		float* or_mag = or_mag_oct[oct][kindex];

		if (idx < region || idx > curRows - region - 1 || idy < region || idy > curCols - region - 1){
			key_now.filtered = true;
			continue;
		}

		float histo[36] = { 0 };
		for (int i = -region; i < region; i++){
			for (int j = -region; j < region; j++){
				if (eucDistance(idx,idy,idx + i, idy + j) > region) continue;

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

		float peaks[36] = { 0 };

		for (int i = 0; i < 36; i++){
			int left = i - 1, right = i + 1;
			if (i == 0){
				left = 35;
			}
			else if (i == 35){
				right = 0;
			}

			if (i != max_bin && histo[i] > histo[left] && histo[i] > histo[right] && histo[i] >= 0.8 * max_hist){
				peaks[i] = histo[i];

				//float orientation = mySiftVertParabola(left * 10 + 5, histo[left], i * 10 + 5, histo[i], right * 10 + 5, histo[right]);
				float orientation = i * ((M_PI * 10.0) / 180.0);
				keypoint newKey(idx, idy, oct, orientation, kindex);
				keys.push_back(newKey);
			}
			else{
				peaks[i] = -1;
			}
		}

		key_now.angle = max_bin * ((M_PI * 10.0) / 180.0);
	}

	delete[] gKernel;
}

Mat mySift(Mat original){
	//Mat out;

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

	int printoff = 0;	//	Debug, used to print to console the values of all keypoints found by this function
	int full_dog = 0;	//	Set this to 1 to change the output image to a full representation of the difference-of-gaussians and scale space location of each keypoint
	int mark_dim = 2;	//	Determines size of circles in the output image.  Recommend setting to 10 or higher if full_dog is set to 1

	//	The first step is to scale up the input image by a factor of 2 in both dimensions, then apply a gaussian blur with sigma = 1.6
	//	Scaling up the input provides more keypoints and approximates a blur with sigma = 1.0, assuming the original image is roughly sigma = 0.5 which is the threshold for noise
	//	Sigma = 1.6 was experimentally determined by Lowe to give the best results.  Refer to the 2004 SIFT paper, pages 9 and 10 for discussion
	float sigma = 1.6;

	uchar scales = 3;
	uchar octaves = 4;
	int region = REGION_SIZE;
	int srcRows = image.rows;
	int srcCols = image.cols;

	float k = pow(2.0, (1.0 / (float)scales));
	int s = scales + 3;

	

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
		std::vector<cv::Mat> blur_img;
		std::vector<cv::Mat> dog_img;

		//printf("Oct: %d\n", oct);
		cv::Mat current = fGaussianFilter(image, pow(k, gauss_exp) * sigma);
		blur_img.push_back(current);
		gauss_exp += 1;

		for (int step = 1; step < s; step++){
			cv::Mat next = fGaussianFilter(image, pow(k, gauss_exp) * sigma);
			cv::Mat dog = cv::Mat::zeros(curRows, curCols, CV_32FC1);
			blur_img.push_back(next);

			float* curr_data = (float*)current.datastart;
			float* next_data = (float*)next.datastart;
			float* dog_data = (float*)dog.datastart;

			cudaMySiftDOG(curr_data, next_data, dog_data, curRows, curCols);

			dog_img.push_back(dog);
			current = next;
			gauss_exp++;
		}

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
						newKey.scale = temp_scale;
						keys.push_back(newKey);
					}
				}
			}

			delete[] key_str;
			delete[] answers;

		}

		curRows = curRows / 2;
		curCols = curCols / 2;
		image = fdirectResize(image, curRows, curCols);
		cudaDeviceSynchronize();

		dog_oct.push_back(dog_img);
		blur_oct.push_back(blur_img);
		gauss_exp -= 2;

	}
	cudaDeviceSynchronize();

	//or_mag
	std::vector<std::vector<float*>> or_mag_oct;
	curRows = srcRows;
	curCols = srcCols;

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
			//cudaTest(curRows, curCols);
			or_mag_current.push_back(or_mag);
		}
		or_mag_oct.push_back(or_mag_current);
	}

	printf("Keys: %d\n", keys.size());

	mySiftEdgeResponses(dog_oct, keys);

	int unfiltered = 0;
	std::vector<keypoint>::iterator iter;
	for (iter = keys.begin(); iter != keys.end();){
		if ((*iter).filtered){
			iter = keys.erase(iter);
			//iter++;
		}
		else{
			unfiltered++;
			iter++;
		}
	}

	key_count = keys.size();

	printf("Unfiltered: %d\n", unfiltered);
	mySiftWriteKeyFile(keys);

	cv::Mat output;

	cudaDeviceSynchronize();

	mySiftOrAssign(keys, or_mag_oct, srcRows, srcCols);

	return dog_oct[0][0];

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
		output = frgb2Gray(original);
	}
	
	return output;
}

#endif