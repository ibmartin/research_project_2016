#ifndef _IMG_PROC_HPP_
#define _IMG_PROC_HPP_

#include <opencv2\\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "nvt_events.hpp"

namespace img_proc{
#define FILTER_SIZE 3
#define M_PI           3.14159265358979323846  /* pi */
#define SOBEL_SIZE 3
#define UPPER_BIT 31
#define LOWER_BIT 0

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

	cv::Mat rgb2Gray(cv::Mat image);
	cv::Mat frgb2Gray(cv::Mat image);
	cv::Mat reverse(cv::Mat image);
	cv::Mat gammaCorrection(cv::Mat image, double gamma);
	cv::Mat directResize(cv::Mat image, int rows, int cols);
	cv::Mat fdirectResize(cv::Mat image, int destRows, int destCols);
	cv::Mat linearResize(cv::Mat image, int rows, int cols);
	cv::Mat flinearResize(cv::Mat image, int rows, int cols);
	void createFilter(double gKernel[][2 * FILTER_SIZE + 1], double inputSigma);
	cv::Mat gaussianFilter(cv::Mat image, double sigma);
	cv::Mat fgaussianFilter(cv::Mat image, double sigma);
	cv::Mat sobelFilter(cv::Mat image);
	double color_distance(double r1, double g1, double b1, double r2, double g2, double b2);
	cv::Mat kMeans(cv::Mat image, int k_means);
	cv::Mat gaussianPyramid(cv::Mat image, uchar levels, float scale);
	cv::Mat fGaussTest(cv::Mat image);
	cv::Mat mySift(cv::Mat image);
	void mySiftEdgeResponses(std::vector<std::vector<cv::Mat>>& dog_oct, std::vector<keypoint>& keys);
	void mySiftDescriptors(std::vector<keypoint>& keys, std::vector<std::vector<cv::Mat>>& blur_oct, std::vector<std::vector<float*>>& or_mag_oct, int unfiltered);
	std::vector<float> mySiftVectorThreshold(std::vector<float>& vec);
	void mySiftNormVec(std::vector<float>& vec);
	cv::Mat mySift_foDer(std::vector<cv::Mat>& neighbors, int px, int py);
	cv::Mat mySift_soDer(std::vector<cv::Mat>& neighbors, int px, int py);
	bool mySiftWriteKeyFile(std::vector<keypoint>& keys);
	bool mySiftReadKeyFile(std::vector<keypoint>& keys, std::string file_name);
	void mySiftKeyCull(std::vector<keypoint>& keys);
	kd_node mySiftKDHelp(std::vector<keypoint>& keys);
	kd_node* mySiftKDTree(std::vector<keypoint>& keys, std::vector<keypoint>::iterator front, std::vector<keypoint>::iterator back, std::string dims, float* all_var, int& count);
	void mySiftKDQuicksort(std::vector<keypoint>& keys, std::vector<keypoint>::iterator front, std::vector<keypoint>::iterator back, int dim);
	unsigned int radixGetMax(unsigned int arr[], int n);
	void mySiftKDCountSort(unsigned int data[], unsigned int index[], int d, int exp);
	void mySiftKDRadixSort(std::vector<keypoint>& keys, std::vector<keypoint>::iterator front, std::vector<keypoint>::iterator back, int dim);
	float mySiftDescDist(keypoint& key_1, keypoint& key_2);
	float mySiftTheoryDist(kd_node* start, keypoint& search_key);
	kd_node* mySiftKDIterSearch(kd_node* root, std::vector<keypoint>& keys, keypoint& search_key);
	kd_node* mySiftKDSearch(kd_node root, std::vector<keypoint>& keys, keypoint& search_key);
	kd_node* mySiftKDSearchHelp(kd_node* current, std::vector<keypoint>& keys, keypoint& search_key, int& max_search);
	cv::Mat bump_map(int dim);
	cv::Mat diff_count(cv::Mat image1, cv::Mat image2);
	float mySiftVertParabola(float l_x, float l_y, float p_x, float p_y, float r_x, float r_y);
}

#endif