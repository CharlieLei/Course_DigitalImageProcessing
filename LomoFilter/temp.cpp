//#include <iostream>
//
//using namespace std;
//
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//
//using namespace cv;
//
//int main()
//{
//    Mat img = imread("../lenna.png");
//    if (img.empty()) {
//        cout << "Error" << endl;
//        return -1;
//    }
//
//    float radius = img.cols > img.rows ? (img.rows / 3) : (img.cols / 3);
//
//    Mat result;
//    const double exponential_e = exp(1.0);
//    // Create Lookup table for color curve effect
//    Mat lut(1, 256, CV_8UC1);
//    for (int i = 0; i < 256; i++)
//    {
//        float x = (float)i / 256.0;
//        lut.at<uchar>(i) = cvRound(256 * (1 / (1 + pow(exponential_e, -((x - 0.5) / 0.1)))));
//    }
//
//    // Split the image channels and apply curve transform only to red channel
//    vector<Mat> bgr;
//    split(img, bgr);
//    LUT(bgr[0], lut, bgr[0]);
//    LUT(bgr[1], lut, bgr[1]);
//    LUT(bgr[2], lut, bgr[2]);
//    //merge result
//    merge(bgr, result);
//
//    // Create image for halo dark
//    Mat halo(img.rows, img.cols, CV_32FC3, Scalar(0.3, 0.3, 0.3));
//    // Create circle
//    circle(halo, Point(img.cols / 2, img.rows / 2), img.cols / 3, Scalar(1, 1, 1), -1);
//    blur(halo, halo, Size(img.cols / 3, img.cols / 3));
//    // Convert the result to float to allow multiply by 1 factor
//    Mat resultf;
//    result.convertTo(resultf, CV_32FC3);
//    // Multiply our result with halo
//    multiply(resultf, halo, resultf);
//    // convert to 8 bits
//    resultf.convertTo(result, CV_8UC3);
//    // show result
//    imshow("Lomograpy", result);
//
//    waitKey(0);
//
//    return 0;
//}