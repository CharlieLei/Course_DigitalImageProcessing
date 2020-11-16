#include <iostream>
#include <vector>

using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;

double distance(int x, int y, int i, int j) {
    return sqrt(pow(x - i, 2) + pow(y - j, 2));
}

double gaussian(double x, double sigma) {
    return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (sqrt(2 * CV_PI) * sigma);
}

void detectFaces(Mat &img, vector<Rect> &faces, String &classifierPath) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    CascadeClassifier classifier;
    if (!classifier.load(classifierPath)) {
        cerr << "Error:: cannot load xml: " << classifierPath << endl;
        return;
    }

    classifier.detectMultiScale(gray, faces);
}

void bilateralFiltering(Mat &src, Mat &dst, Rect &face, int kernelSize, double sigmaI, double sigmaS) {
    for (int channel = 0; channel < 3; channel++) {
        for (int originalX = face.x; originalX < face.x + face.width; originalX++) {
            for (int originalY = face.y; originalY < face.y + face.height; originalY++) {

                // 开始卷积
                double iFiltered = 0.0, wP = 0.0;
                for (int convX = 0; convX < kernelSize; convX++) {
                    for (int convY = 0; convY < kernelSize; convY++) {
                        int neighborX = originalX - (kernelSize / 2 - convX);
                        int neighborY = originalY - (kernelSize / 2 - convY);

                        // 防止越界
                        if (neighborX < 0 || neighborX > src.rows - 1 || neighborY < 0 || neighborY > src.cols - 1)
                            continue;
                        // range kernel
                        double fr = gaussian(src.at<Vec3b>(neighborX, neighborY)[channel] -
                                             src.at<Vec3b>(originalX, originalY)[channel], sigmaI);
                        // spatial kernel
                        double gs = gaussian(distance(originalX, originalY, neighborX, neighborY), sigmaS);
                        double w = fr * gs;
                        iFiltered += src.at<Vec3b>(neighborX, neighborY)[channel] * w;
                        wP += w;
                    }
                }
                iFiltered /= wP;
                dst.at<Vec3b>(originalX, originalY)[channel] = uchar(iFiltered);

            }
        }
    }
}

int main() {
    String filepath;
    String outputPath;
    String classifierPath;
    cout << "filepath:[data/lenna.png] ";
    getline(cin, filepath);
    cout << "output file path:[newimage.png] ";
    getline(cin, outputPath);

    filepath = filepath.empty() ? "data/lenna.png" : filepath;
    outputPath = outputPath.empty() ? "newimage.png" : outputPath;
    classifierPath = "haarcascade_frontalface_default.xml";

    Mat img = imread(filepath);
    if (img.empty()) {
        cerr << "Error:: image file " << filepath << " cannot open" << endl;
        return -1;
    } else {
        cout << "Success:: open image file" << endl;
    }

    vector<Rect> faces;
    detectFaces(img, faces, classifierPath);
    cout << "Success:: detect faces in image" << endl;

    Mat result = img.clone();
    for (int i = 0; i < faces.size(); i++) {
        bilateralFiltering(img, result, faces[i], 10, 25.0, 25.0);
        cout << "Success:: bilateral filter faces " << i << endl;
    }

    Mat diff = 10 * (result - img);

    for (int i = 0; i < faces.size(); i++) {
        rectangle(result, faces[i], Scalar(0, 255, 0));
    }

    imshow("original image", img);
    imshow("filtered image", result);
    imwrite(outputPath, result);
    imshow("difference", diff);
    imwrite("difference.png", diff);
    waitKey(0);
}