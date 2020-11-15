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

void drawCircle(Mat &dst, double radius) {
    double xCenter = dst.rows / 2.0, yCenter = dst.cols / 2.0;
    Scalar frontColor(3.0, 3.0, 3.0);
    Scalar backgroundColor(0.5, 0.5, 0.5);

    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            double sqDist = pow(i - xCenter, 2) + pow(j - yCenter, 2);
            if (sqDist - radius * radius < 0.0001) {
                dst.at<Vec3f>(i, j)[0] = frontColor[0];
                dst.at<Vec3f>(i, j)[1] = frontColor[1];
                dst.at<Vec3f>(i, j)[2] = frontColor[1];
            } else {
                dst.at<Vec3f>(i, j)[0] = backgroundColor[0];
                dst.at<Vec3f>(i, j)[1] = backgroundColor[1];
                dst.at<Vec3f>(i, j)[2] = backgroundColor[1];
            }
        }
    }
}

void boxFiltering(Mat &src, Mat &dst, int kernelWidth, int kernelHeight) {
    if (src.rows != dst.rows && src.cols != dst.cols) {
        cerr << "boxFiltering:: sizes of two Mat are not same" << endl;
        return;
    }

    Mat buf(src.rows, src.cols, src.type());

    int paddingWidth = kernelWidth / 2, paddingHeight = kernelHeight / 2;
    double kernelSize = kernelWidth * kernelHeight;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            for (int startX = i - paddingWidth; startX < i - paddingWidth + kernelWidth; startX++) {
                if (startX < 0 || startX > src.rows - 1) {
                    continue;
                } else {
                    buf.at<Vec3f>(i, j)[0] += src.at<Vec3f>(startX, j)[0];
                    buf.at<Vec3f>(i, j)[1] += src.at<Vec3f>(startX, j)[1];
                    buf.at<Vec3f>(i, j)[2] += src.at<Vec3f>(startX, j)[2];
                }
            }

        }
    }
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            for (int startY = j - paddingHeight; startY < j - paddingHeight + kernelHeight; startY++) {
                if (startY < 0 || startY > src.cols - 1) {
                    continue;
                } else {
                    dst.at<Vec3f>(i, j)[0] += buf.at<Vec3f>(i, startY)[0];
                    dst.at<Vec3f>(i, j)[1] += buf.at<Vec3f>(i, startY)[1];
                    dst.at<Vec3f>(i, j)[2] += buf.at<Vec3f>(i, startY)[2];
                }
            }

        }
    }

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            dst.at<Vec3f>(i, j)[0] /= kernelSize;
            dst.at<Vec3f>(i, j)[1] /= kernelSize;
            dst.at<Vec3f>(i, j)[2] /= kernelSize;

        }
    }
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
//    curveTransform(bgr[0], lookupTable);
//    curveTransform(bgr[1], lookupTable);
    curveTransform(bgr[2], lookupTable);
    merge(bgr, img);

    Mat lightCircle(img.rows, img.cols, CV_32FC3);
    drawCircle(lightCircle, img.cols / 3);
    Mat halo(lightCircle.rows, lightCircle.cols, CV_32FC3);
    boxFiltering(lightCircle, halo, img.cols / 2, img.cols / 2);

    cout << img.type() << endl;

    Mat temp;
    img.convertTo(temp, CV_32FC3);
    multiply(temp, halo, temp); // temp = temp * halo
    Mat result;
    temp.convertTo(result, CV_8UC3);

    // show result
    imshow("my impl", result);
//    imshow("impl", res);
    waitKey(0);
}