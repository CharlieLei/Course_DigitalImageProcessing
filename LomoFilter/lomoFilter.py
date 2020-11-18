import sys
import cv2
import numpy as np


def distance(pt1, pt2):
    (x, y) = pt1
    (i, j) = pt2
    return np.sqrt(pow(x - i, 2) + pow(y - j, 2))


def gaussian(x, sigma):
    return np.exp(-(pow(x, 2)) / (2 * pow(sigma, 2)))


def contrast_stretching(src):
    # initLookUpTable
    lookup_table = np.zeros(256)
    for i in range(256):
        x = i / 256.0
        y = 1.0 / (1.0 + np.exp(-(x - 0.5) / 0.1))
        lookup_table[i] = np.round(256 * y)

    # curveTransform
    dst = np.zeros(src.shape)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            for channel in range(3):
                dst[i][j][channel] = lookup_table[src[i][j][channel]]

    return dst.astype(np.uint8)


def vignetting_filtering(src, sigma):
    center = (src.shape[0] / 2, src.shape[1] / 2)

    mask = np.zeros(src.shape)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            temp = gaussian(distance((i, j), center), sigma)
            mask[i][j] = np.array([temp, temp, temp])

    dst = overlay(src / 255, mask)
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


def main():
    filepath = input('filepath:[data/lenna.png] ')
    output_path = input('output file path:[newimage.png] ')

    filepath = 'data/lenna.png' if filepath == '' else filepath
    output_path = "newimage.png" if output_path == '' else output_path

    img = cv2.imread(filepath)
    if img is None:
        print('"Error:: image file ' + filepath + ' cannot open', file=sys.stderr)
        return
    else:
        cv2.imshow('original image', img)
        print('Success:: open image file')

    new_img = np.copy(img)
    print('Start:: vignette image')
    vignetting = vignetting_filtering(new_img, new_img.shape[0] / 3)
    # cv2.imshow('vignetting', result)
    print('Start:: stretch image contrast')
    lomography = contrast_stretching(vignetting)
    # cv2.imshow('contrast_stretching', result)
    
    cv2.imshow('lomography', lomography)
    cv2.imwrite(output_path, lomography)
    cv2.waitKey()


if __name__ == "__main__":
    main()
