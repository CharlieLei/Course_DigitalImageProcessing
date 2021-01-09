import argparse
import os

import cv2
import dlib
import numpy as np
# from matplotlib import pyplot as plt

from img_utils import mls_affine_deformation, idw

predictor_path = 'model/shape_predictor_68_face_landmarks.dat'

# 检测人脸区域
detector = dlib.get_frontal_face_detector()
# 预测人脸68个特征点
predictor = dlib.shape_predictor(predictor_path)


def parse_args():
    desc = "FaceLift"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input', type=str, default='imgs/default.png', help='输入图像的路径')
    parser.add_argument('--output', type=str, default='result/default_face_lifted.png', help='输出图像的路径')
    parser.add_argument('--intensity', type=int, default=50, help='瘦脸参数[0,100]')
    parser.add_argument('--algo', type=str, default='mls', help='瘦脸算法：1) mls ; 2) idw')
    return parser.parse_args()


def get_landmark(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray)

    all_landmarks = []
    for i, rect in enumerate(rects):
        landmarks = [[p.x, p.y] for p in predictor(img_gray, rect).parts()]
        all_landmarks.append(landmarks)

    return rects, np.array(all_landmarks)


# def print_img(img, lifted_img, algo_name, intensity):
#     plt.figure(figsize=(8, 8))
#     plt.subplot(121)
#     plt.axis('off')
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title('Original Image')
#     plt.subplot(122)
#     plt.axis('off')
#     plt.imshow(cv2.cvtColor(lifted_img, cv2.COLOR_BGR2RGB))
#     plt.title('%s Deformation: intensity %d' % (algo_name, intensity))
#     plt.show()


# def print_landmark(img, ctrl_pts, warp_ctrl_pts, center):
#     plt.axis('off')
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     for pt in ctrl_pts:
#         plt.scatter(pt[0], pt[1], color='blue')
#     for pt in warp_ctrl_pts:
#         plt.scatter(pt[0], pt[1], color='red', s=10)
#     plt.scatter(center[0], center[1], color='green')
#     plt.show()


def main():
    args = parse_args()
    img_path = args.input
    output_path = args.output
    intensity = args.intensity
    algo_name = args.algo

    if not os.path.exists(img_path):
        print('文件不存在: ' + img_path)
        return

    if algo_name == 'mls':
        lifted_func = mls_affine_deformation
    elif algo_name == 'idw':
        lifted_func = idw
    else:
        print('选择的算法' + algo_name + '不存在')
        return

    img_src = cv2.imread(img_path)
    rects, all_landmarks = get_landmark(img_src)

    if len(all_landmarks) == 0:
        print('未检测到人脸！')
        return

    img_lifted = np.copy(img_src)
    for i, rect in enumerate(rects):
        landmarks = all_landmarks[i]
        left_lm, right_lm = landmarks[4], landmarks[12]
        center = landmarks[30]

        warp_ratio_left = 0.2 * intensity / 100.0
        warp_ratio_right = warp_ratio_left

        left_warp_lm = left_lm + warp_ratio_left * (center - left_lm)
        right_warp_lm = right_lm + warp_ratio_right * (center - right_lm)

        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        rect_pts = [[left, top], [right, top], [right, bottom], [left, bottom]]

        ctrl_pts = np.array([landmarks[1], left_lm, landmarks[8], right_lm, landmarks[15]] + rect_pts)
        warp_ctrl_pts = np.array([landmarks[1], left_warp_lm, landmarks[8], right_warp_lm, landmarks[15]] + rect_pts)

        img_lifted = lifted_func(img_lifted, ctrl_pts, warp_ctrl_pts)

        # print_landmark(img_src, ctrl_pts, warp_ctrl_pts, center)
        # print_landmark(img_lifted, ctrl_pts, warp_ctrl_pts, center)

    # print_img(img_src, img_lifted, algo_name, intensity)

    cv2.imwrite(output_path, img_lifted)
    print("完成瘦脸， 输出图像为：" + output_path)


if __name__ == '__main__':
    main()
