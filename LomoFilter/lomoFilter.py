import sys
import cv2
import numpy as np


def distance(pt1, pt2):
    (x, y) = pt1
    (i, j) = pt2
    return np.sqrt(pow(x - i, 2) + pow(y - j, 2))


def gaussian(x, sigma):
    return np.exp(-(pow(x, 2)) / (2 * pow(sigma, 2)))


def scale_intensity(src):
    # scale_factor = np.max(src) - np.min(src)
    # src = (src - np.min(src)) / scale_factor * 255
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            for channel in range(3):
                src[i][j][channel] = np.clip(src[i][j][channel], 0, 255)
    return src


def test(src):
    scale_factor = np.max(src) - np.min(src)
    src = (src - np.min(src)) / scale_factor * 255
    return src


def laplacian_filtering(src):
    laplacian_kernel = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0])

    dst = np.zeros(src.shape)
    for originalX in range(1, src.shape[0] - 1):
        for originalY in range(1, src.shape[1] - 1):

            # 开始卷积
            conv = np.zeros(3)
            count = 0
            for neighbor_x in range(originalX - 1, originalX + 2):
                for neighbor_y in range(originalY - 1, originalY + 2):
                    # 防止越界
                    # if 0 <= neighbor_x <= src.shape[0] - 1 and 0 <= neighbor_y <= src.shape[1] - 1:
                    for channel in range(3):
                        conv[channel] += laplacian_kernel[count] * src[neighbor_x][neighbor_y][channel]
                    count += 1

            dst[originalX][originalY] = conv

    dst = dst + src
    # 注意拉普拉斯算子会导致数值超过颜色范围
    dst = scale_intensity(dst)
    return dst.astype(np.uint8)


def contrast_stretching(src):
    # initLookUpTable
    lookup_table = np.zeros(256)
    for i in range(256):
        x = i / 256.0
        y = 1.0 / (1.0 + np.exp(-(x - 0.5) / 0.1))
        lookup_table[i] = np.round(256 * y)

    # curveTransform
    dst = np.zeros(src.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for channel in range(3):
                dst[i][j][channel] = lookup_table[src[i][j][channel]]

    return dst.astype(np.uint8)


def vignetting_filtering(src, sigma):
    center = (src.shape[0] / 2, src.shape[1] / 2)
    corner = (0, 0)

    mask = np.zeros(src.shape)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            temp = gaussian(distance((i, j), center), sigma)
            mask[i][j] = np.array([temp, temp, temp])

    dst = overlay(src / 255, mask)
    # dst = src * mask
    # dst = scale_intensity(dst)
    return dst.astype(np.uint8)


def overlay(base_layer, top_layer):
    assert base_layer.shape == top_layer.shape

    dst = np.zeros(base_layer.shape)
    for i in range(base_layer.shape[0]):
        for j in range(base_layer.shape[1]):
            for channel in range(3):
                a = base_layer[i][j][channel]
                b = top_layer[i][j][channel]
                dst[i][j][channel] = 2 * a * b if a < 0.5 else 1 - 2 * (1 - a) * (1 - b)

    dst = dst * 255
    return dst.astype(np.uint8)


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

    return dst.astype(np.uint8)


if __name__ == "__main__":
    filepath = ''  # input('filepath:[data/lenna.png] ')
    outputPath = ''  # input('output file path:[newimage.png] ')

    filepath = 'data/lenna.png' if filepath == '' else filepath
    outputPath = "newimage.png" if outputPath == '' else outputPath

    img = cv2.imread(filepath)
    if img is None:
        print('"Error:: image file ' + filepath + ' cannot open', file=sys.stderr)

    result = img
    # result = boxFiltering(result, (3, 3))
    # cv2.imshow('smooth', result)

    # result = contrast_stretching(result)
    # cv2.imshow('contrast_stretching', result)
    result = vignetting_filtering(result, result.shape[0] / 3)
    cv2.imshow('vignetting', result)
    result = contrast_stretching(result)
    cv2.imshow('contrast_stretching', result)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(result, (3, 3), 0)
    # Apply Laplacian operator in some higher datatype
    blur = laplacian_filtering(blur)
    # result = scale_intensity(result + blur)
    # result = laplacian_filtering(result)
    # result = boxFiltering(result, (3, 3))
    cv2.imshow('smooth', blur)

    # result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    # (h, s, v) = cv2.split(result)
    # s = (s * 1.5).astype(np.uint8)
    # result = cv2.merge((h, s, v))
    # 查表法
    # lookupTable = initLookUpTable()
    #
    # b, g, r = cv2.split(img)
    # r = curveTransform(r, lookupTable)
    # filtered_img = cv2.merge((b, g, r))

    # 绘制光晕
    # lightCircle = drawCircle(filtered_img.shape, min(filtered_img.shape[0], filtered_img.shape[1]) / 3)
    # cv2.imshow('circle', lightCircle)
    # # kernelWidth = int(min(filtered_img.shape[0], filtered_img.shape[1]) / 2)
    # # halo = boxFiltering(lightCircle, (kernelWidth, kernelWidth))
    # # cv2.imshow('halo', halo)
    # #
    # filtered_img = filtered_img.astype(np.float64)
    # result = (filtered_img * lightCircle).astype(np.uint8)
    #
    # cv2.imshow('original image', img)
    # cv2.imshow("Lomo Filter", result)

    cv2.waitKey()
