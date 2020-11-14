#include <iostream>

using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

void initLookUpTable(Mat &table) {
    for (int i = 0; i < 256; i++) {
        double x = i / 256.0;
        double y = 1.0 / (1.0 + exp(-(x - 0.5) / 0.1));
        table.at<int>(i) = cvRound(256 * y);
    }
}

void curveTransform(Mat &img, Mat &lookupTable) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            img.at<uchar>(i, j) = lookupTable.at<int>(img.at<uchar>(i, j));
        }
    }
}

void drawLightCircle(Mat &halo) {
    double xCenter = halo.rows / 2.0, yCenter = halo.cols / 2.0;
    double radius = min(halo.rows, halo.cols) / 3.0;
    Scalar frontColor(2.0, 2.0, 2.0);
    Scalar backgroundColor(0.3, 0.3, 0.3);
}

int main() {
    Mat img = imread("../lenna.png");
    if (img.empty()) {
        cerr << "Error" << endl;
        return -1;
    }

    // 查表法
    Mat lookupTable(1, 256, CV_32SC1);
    initLookUpTable(lookupTable);

    vector<Mat> bgr;
    split(img, bgr);
    curveTransform(bgr[2], lookupTable);
    merge(bgr, img);

    Mat halo(img.rows, img.cols, CV_32FC3);

        // Create Lookup table for color curve effect
    Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++)
    {
        float x = (float)i / 256.0;
        lut.at<uchar>(i) = cvRound(256 * (1 / (1 + pow(exp(1.0), -((x - 0.5) / 0.1)))));
    }

    // Split the image channels and apply curve transform only to red channel
    Mat res;
    vector<Mat> bgr2;
    split(img, bgr2);
    LUT(bgr2[2], lut, bgr2[2]);
    merge(bgr2, res);
    Mat halo2(img.rows, img.cols, CV_32FC3, Scalar(0.3, 0.3, 0.3));
    circle(halo2, Point(img.cols / 2, img.rows / 2), img.cols / 3, Scalar(1, 1, 1), -1);

    // show result
    imshow("my impl", img);
    imshow("impl", res);
    waitKey(0);
}