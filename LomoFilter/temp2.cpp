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
//int main() {
//    Mat img = imread("../lenna.png");
//    if (img.empty()) {
//        cout << "Error" << endl;
//        return -1;
//    }
//    cvtColor(img, img, COLOR_BGR2HSV);
//
//    vector<Mat> hsv;
//    split(img, hsv);
//
//    double saturation = 500.0 / 255.0;
//    double scale = 1;
//    hsv[1] = saturation * hsv[1];
//
//    Mat result;
//    merge(hsv, result);
//
//    cvtColor(result, result, COLOR_HSV2BGR);
//
//    // show result
//    imshow("Lomograpy", result);
//    waitKey(0);
//
//    return 0;
//}
