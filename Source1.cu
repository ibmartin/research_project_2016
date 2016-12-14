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

//-- Main

int main(void){		// [MAIN]

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

	for (int pics = 0; pics <= 10; pics++){
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
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_oranges.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_oranges.txt");
			im1 = imread("D://School//Summer 2016//Research//oranges.jpg");
			name = "oranges";
		}
		else if (pics == 9){
			avg_file.open("D://School//Summer 2016//Research//Textfiles//averages_mountains.txt");
			det_file.open("D://School//Summer 2016//Research//Textfiles//details_mountains.txt");
			im1 = imread("D://School//Summer 2016//Research//mountains.jpg");
			name = "mountains";
		}
		else if (pics == 10){
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
		//Mat im1 = imread("D://School//Summer 2016//Research//valve.png");
		det_file << "GAUSSIAN FILTER (SIZE 3, SIGMA 1.0)" << endl;
		avg_file << "GAUSSIAN FILTER (SIZE 3, SIGMA 1.0)" << endl;
		for (int runs = 1; runs <= rounds; runs++){
			double cpu_time = 0, gpu_time = 0;
			auto t1 = std::chrono::high_resolution_clock::now();
			img_proc::gaussianFilter(im1, 1.0);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			gaussianFilter(im1, 1.0);
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
			img_proc::kMeans(im1,8);
			auto t2 = std::chrono::high_resolution_clock::now();
			cpu_time = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1).count();
			det_file << "CPU Run: " << fixed << cpu_time << micro << "s" << endl;
			cpu_duration += cpu_time;

			auto t3 = std::chrono::high_resolution_clock::now();
			kMeans(im1,8);
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