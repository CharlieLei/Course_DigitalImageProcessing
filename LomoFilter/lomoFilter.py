import sys
import cv2
import numpy as np

def distance(x, y, i, j):
    return np.sqrt(pow(x - i, 2) + pow(y - j, 2))

def gaussian(x, sigma):
    return np.exp(-(pow(x, 2)) / (2 * pow(sigma, 2)))

def initLookUpTable():
    table = np.zeros(256)
    for i in range(256):
        x = i / 256.0
        y = 1.0 / (1.0 + np.exp(-(x - 0.5) / 0.1))
        table[i] = np.round(256 * y)
    return table


def curveTransform(img, lookupTable):
    newImg = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newImg[i][j] = lookupTable[img[i][j]]

    return newImg.astype(np.uint8)


def drawCircle(img_shape, radius):
    xCenter = img_shape[0] / 2.0
    yCenter = img_shape[1] / 2.0
    frontColor = [3.0, 3.0, 3.0]
    backgroundColor = [0.5, 0.5, 0.5]

    dst = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            sqDist = pow(i - xCenter, 2) + pow(j - yCenter, 2)
            dst[i][j] = max(2 * gaussian(distance(i,j, xCenter, yCenter), 100), 0.5)
            # if sqDist - radius * radius < 0.0001:
            #     dst[i][j] = frontColor
            # else:
            #     dst[i][j] = backgroundColor

    return dst


def boxFiltering(src, kernelSize):
    buf = np.zeros(src.shape)
    dst = np.zeros(src.shape)

    for originalX in range(src.shape[0]):
        for originalY in range(src.shape[1]):

            for convX in range(kernelSize[0]):
                neighborX = int(originalX - (kernelSize[0] / 2 - convX))
                if 0 <= neighborX <= src.shape[0] - 1:
                    buf[originalX][originalY] += src[neighborX][originalY]

    for originalX in range(src.shape[0]):
        for originalY in range(src.shape[1]):

            for convY in range(kernelSize[1]):
                neighborY = int(originalY - (kernelSize[1] / 2 - convY))
                if 0 <= neighborY <= src.shape[1] - 1:
                    dst[originalX][originalY] += buf[originalX][neighborY]

    for originalX in range(src.shape[0]):
        for originalY in range(src.shape[1]):
            dst[originalX][originalY] /= (kernelSize[0] * kernelSize[1])

    return dst


if __name__ == "__main__":
    filepath = ''  # input('filepath:[data/lenna.png] ')
    outputPath = ''  # input('output file path:[newimage.png] ')

    filepath = 'data/lenna.png' if filepath == '' else filepath
    outputPath = "newimage.png" if outputPath == '' else outputPath

    img = cv2.imread(filepath)
    if img is None:
        print('"Error:: image file ' + filepath + ' cannot open', file=sys.stderr)

    # 查表法
    lookupTable = initLookUpTable()

    b, g, r = cv2.split(img)
    r = curveTransform(r, lookupTable)
    filtered_img = cv2.merge((b, g, r))

    # 绘制光晕
    lightCircle = drawCircle(filtered_img.shape, min(filtered_img.shape[0], filtered_img.shape[1]) / 3)
    cv2.imshow('circle', lightCircle)
    # kernelWidth = int(min(filtered_img.shape[0], filtered_img.shape[1]) / 2)
    # halo = boxFiltering(lightCircle, (kernelWidth, kernelWidth))
    # cv2.imshow('halo', halo)
    #
    filtered_img = filtered_img.astype(np.float64)
    result = (filtered_img * lightCircle).astype(np.uint8)
    #
    cv2.imshow('original image', img)
    cv2.imshow("Lomo Filter", result)

    cv2.waitKey(0)
