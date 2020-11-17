import sys
import cv2
import numpy as np


def distance(x, y, i, j):
    return np.sqrt(pow(x - i, 2) + pow(y - j, 2))


def gaussian(x, sigma):
    return np.exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (np.sqrt(2 * np.pi) * sigma)


def detect_faces(img, classifier_path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier(classifier_path)
    faces = classifier.detectMultiScale(gray)
    return faces


def bilateral_filtering(src, face, kernel_size, sigma_i, sigma_s):
    [x, y, width, height] = face
    dst = np.copy(src)

    for original_x in range(x, x + width):
        print('  start processing column ' + str(original_x) + '| end: ' + str(x + width - 1))
        for original_y in range(y, y + height):

            # 开始卷积
            i_filtered = np.zeros(3)
            w_p = np.zeros(3)
            for convX in range(kernel_size):
                for convY in range(kernel_size):
                    neighbor_x = int(original_x - (kernel_size / 2 - convX))
                    neighbor_y = int(original_y - (kernel_size / 2 - convY))

                    # 防止越界
                    if 0 <= neighbor_x <= src.shape[0] - 1 and 0 <= neighbor_y <= src.shape[1] - 1:
                        for channel in range(3):
                            # range kernel
                            fr = gaussian(float(src[neighbor_x][neighbor_y][channel]) -
                                          float(src[original_x][original_y][channel]),
                                          sigma_i)
                            # spatial kernel
                            gs = gaussian(distance(original_x, original_y, neighbor_x, neighbor_y), sigma_s)
                            w = fr * gs
                            i_filtered[channel] += src[neighbor_x][neighbor_y][channel] * w
                            w_p[channel] += w

            i_filtered /= w_p
            dst[original_x][original_y] = i_filtered

    return dst.astype(np.uint8)


def main():
    filepath = input('filepath:[data/lenna.png] ')
    output_path = input('output file path:[newimage.png] ')

    filepath = 'data/lenna.png' if filepath == '' else filepath
    output_path = 'newimage.png' if output_path == '' else output_path
    classifier_path = 'haarcascade_frontalface_default.xml'

    img = cv2.imread(filepath)
    if img is None:
        print('Error:: image file ' + filepath + ' cannot open', file=sys.stderr)
        return
    else:
        print('Success:: open image file')

    faces = detect_faces(img, classifier_path)

    result = img
    for face in faces:
        print('Start processing ')
        result = bilateral_filtering(result, face, 10, 25, 25)
        print('Success:: bilateral filter face')

    diff = 10 * (result - img)

    for (x, y, width, height) in faces:
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 1)

    cv2.imshow('original image', img)
    cv2.imshow('filtered image', result)
    cv2.imshow('difference', diff)
    cv2.imwrite(output_path, result)
    cv2.imwrite('difference.png', diff)

    cv2.waitKey()


if __name__ == "__main__":
    main()
