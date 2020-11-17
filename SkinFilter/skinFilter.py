import sys
import cv2
import numpy as np

def distance(x, y, i, j):
    return np.sqrt(pow(x - i, 2) + pow(y - j, 2))

def gaussian(x, sigma)
    return np.exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (np.sqrt(2 * np.pi) * sigma)

def detectFaces(img, classifierPath):
    gray = cvtColor(img, cv.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier(classifierPath)
    faces = classifier.detectMultiScale(gray)    
    return faces

def bilateralFiltering(src, face, kernelSize, sigmaI, sigmaS):
    dst = np.zeros(src.shape)
    for channel in range(3):
        for originalX in range(face.x, face.x + face.width):
            for originalY in range(face.y, face.y + face.height):
            
                # 开始卷积
                iFiltered = 0.0
                wP = 0.0
                for convX in range(kernelSize):
                    for convY in range(kernelSize):
                        neighborX = originalX - (kernelSize / 2 - convX);
                        neighborY = originalY - (kernelSize / 2 - convY);

                        # 防止越界
                        if (0 <= neighborX <= src.rows - 1 and 0 <= neighborY <= src.cols - 1):
                            // range kernel
                            fr = gaussian(src[neighborX][neighborY][channel] -
                                                 src[originalX][originalY][channel], sigmaI);
                            // spatial kernel
                            gs = gaussian(distance(originalX, originalY, neighborX, neighborY), sigmaS);
                            w = fr * gs;
                            iFiltered += src[neighborX][neighborY][channel] * w;
                            wP += w;
                            
                iFiltered /= wP;
                dst[originalX][originalY][channel] = iFiltered;
                
    return dst

if __name__ == "__main__":
    filepath = ''  # input('filepath:[data/lenna.png] ')
    outputPath = ''  # input('output file path:[newimage.png] ')

    filepath = 'data/lenna.png' if filepath == '' else filepath
    outputPath = 'newimage.png' if outputPath == '' else outputPath
    classifierPath = 'haarcascade_frontalface_default.xml'

    img = cv2.imread(filepath)
    if img is None:
        print('"Error:: image file ' + filepath + ' cannot open', file=sys.stderr)
        
    faces = detectFaces(img, classifierPath)
    for face in faces:
        result = bilateralFiltering(img, face, 10, 25, 25)
    
    diff = 10 * (result - img)
    
    for face in faces:
        cv2.rectangle(result, faces[i], (0, 255, 0))
    
    imshow('original image', img)
    imshow('filtered image', result)
    imshow('difference', diff)
    imwrite('difference.png', diff)
    waitkey(0)
    
