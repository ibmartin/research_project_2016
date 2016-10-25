#ifndef _IMG_PROC_HPP_
#define _IMG_PROC_HPP_

#include <opencv2\\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

namespace img_proc{
#define FILTER_SIZE 3
#define M_PI           3.14159265358979323846  /* pi */
#define SOBEL_SIZE 3

	class keypoint
	{
	public:
		int idx, idy, oct, index;
		float angle, scale, mag;
		bool filtered = false;
		keypoint(int _idx, int _idy, int _oct, float _angle, int _index){	//Constuctor
			idx = _idx;
			idy = _idy;
			oct = _oct;
			angle = _angle;
			index = _index;
		}
	};

	cv::Mat rgb2Gray(cv::Mat image);
	cv::Mat frgb2Gray(cv::Mat image);
	cv::Mat reverse(cv::Mat image);
	cv::Mat gammaCorrection(cv::Mat image, double gamma);
	cv::Mat directResize(cv::Mat image, int rows, int cols);
	cv::Mat fdirectResize(cv::Mat image, int rows, int cols);
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
	cv::Mat mySift_foDer(std::vector<cv::Mat>& neighbors, int px, int py);
	cv::Mat mySift_soDer(std::vector<cv::Mat>& neighbors, int px, int py);

}

#endif