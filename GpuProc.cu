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
	int region = 4;
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

		dog_oct.push_back(dog_img);
		blur_oct.push_back(blur_img);
		gauss_exp -= 2;

	}

	printf("Keys: %d\n", keys.size());
	mySiftWriteKeyFile(keys);

	//or_mag
	std::vector<std::vector<float*>> or_mag_oct;
	curRows = srcRows;
	curCols = srcCols;

	for (int oct = 0; oct < octaves; oct++){
		std::vector<float*> or_mag_current;
		for (int step = 0; step < s; step++){
			cv::Mat& current = blur_oct[oct][step];

			float* curr_data = (float*)current.datastart;
			float* or_mag = new float[2 * curRows * curCols];

			//cudaMySiftOrMagGen(curr_data, or_mag, curRows, curCols);
			or_mag_current.push_back(or_mag);
		}
		or_mag_oct.push_back(or_mag_current);
	}

	cv::Mat output;

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