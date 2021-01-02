import numpy as np

# 在计算权重w_i时，分母可能为0；避免报运行时除零错
np.seterr(divide='ignore', invalid='ignore')


def mls_affine_deformation(image: np.ndarray, p: np.ndarray, q: np.ndarray,
                           alpha: float = 1.0, density: float = 1.0):
    """
    moving least squares仿射变换
    :param image: 原图像
    :param p: 原始控制点的集合 大小为[n,2]
    :param q: 变形后控制点的集合 大小为[n,2]
    :param alpha: 权重w_i中分母的指数
    :param density: 网格的密度
    :return: 变形后的图像
    """
    height, width = image.shape[0], image.shape[1]
    # 将 (x, y) 转为 (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # 交换 p 和 q
    # 这样就只需要将目标像素点转换到对应的源像素点即可
    p, q = q, p

    # 构造网格坐标的X和Y坐标
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    # 网格行数和列数
    grow, gcol = vx.shape[0], vx.shape[1]
    # 控制点个数
    ctrls = p.shape[0]

    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    # 网格坐标点
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** alpha  # [ctrls, grow, gcol]
    w[w == np.inf] = 2 ** 31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 2, 1, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    pTwp = np.sum(reshaped_phat1 * reshaped_w * reshaped_phat2, axis=0)  # [2, 2, grow, gcol]
    try:
        inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))  # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(pTwp.transpose(2, 3, 0, 1))  # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(1, 1, grow, gcol)  # [1, 1, grow, gcol]
        adjoint = pTwp[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]  # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]  # [2, 2, grow, gcol]
        inv_pTwp = (adjoint / reshaped_det).transpose(2, 3, 0, 1)  # [grow, gcol, 2, 2]
    mul_left = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)  # [grow, gcol, 1, 2]
    mul_right = reshaped_w * reshaped_phat1  # [ctrls, 2, 1, grow, gcol]
    reshaped_mul_right = mul_right.transpose(0, 3, 4, 1, 2)  # [ctrls, grow, gcol, 2, 1]
    A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right)  # [ctrls, grow, gcol, 1, 1]
    reshaped_A = A.reshape(ctrls, 1, grow, gcol)  # [ctrls, 1, grow, gcol]

    # 计算 q 相关的部分
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [ctrls, 2, grow, gcol]

    # 获得最终的变换坐标
    transformers = np.sum(reshaped_A * qhat, axis=0) + qstar  # [2, grow, gcol]

    # pTwp是奇异矩阵需要矫正
    if flag:
        blidx = det == np.inf  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # 去除变换后位置在图片之外的点
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # 使用计算好的变换坐标进行变换
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16),
                                       (np.arange(grow) / density).astype(np.int16))
    # transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    transformed_image[new_gridX, new_gridY] = image[tuple(transformers.astype(np.int16))]

    return transformed_image


def idw(image: np.ndarray, p: np.ndarray, q: np.ndarray, u: float = 1.0):
    """
    inverse distance weighted
    :param image: 原图像
    :param p: 原始控制点的集合 大小为[n,2]
    :param q: 变形后控制点的集合 大小为[n,2]
    :param u: 权重ρ_i中分母的指数
    :return: 变形后的图像
    """
    height, width = image.shape[0], image.shape[1]
    # 将 (x, y) 转为 (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # 交换 p 和 q
    # 这样就只需要将目标像素点转换到对应的源像素点即可
    p, q = q, p

    # 构造网格坐标的X和Y坐标
    coordX = np.linspace(0, width, num=int(width), endpoint=False)
    coordY = np.linspace(0, height, num=int(height), endpoint=False)
    vy, vx = np.meshgrid(coordX, coordY)
    # 网格行数和列数
    row, col = vx.shape[0], vx.shape[1]
    # 控制点个数
    ctrls = p.shape[0]

    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    # 网格坐标点
    reshaped_v = np.vstack((vx.reshape(1, row, col), vy.reshape(1, row, col)))  # [2, row, col]

    rho = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** u  # [ctrls, 2, row, col]
    rho[rho == np.inf] = 1.0
    # rho = 1.0 / rho
    w = rho / np.sum(rho, axis=0)  # [ctrls, row, col]
    w = w.reshape(ctrls, 1, row, col)  # [ctrls, 1, row, col]
    fi = reshaped_q - reshaped_p + reshaped_v  # [ctrls, 2, row, col]
    f = np.sum(w * fi, axis=0)  # [2, row, col]

    # 去除变换后位置在图片之外的点
    f[f < 0] = 0
    f[0][f[0] > height - 1] = 0
    f[1][f[1] > width - 1] = 0

    # 使用计算好的变换坐标进行变换
    transformed_image = np.ones_like(image) * 255
    new_coordY, new_coordX = np.meshgrid(np.arange(col).astype(np.int16), np.arange(row).astype(np.int16))
    transformed_image[new_coordX, new_coordY] = image[tuple(f.astype(np.int16))]

    return transformed_image

    # img_warped = np.copy(image)
    #
    # height, width = image.shape[0], image.shape[1]
    #
    # for i in range(width):
    #     for j in range(height):
    #         curr_pt = np.array([i, j])
    #
    #         rho = np.sum((curr_pt - p) ** 2 ** u, axis=1)
    #         rho[rho == 0.0] = 1.0
    #         rho = 1.0 / rho  # [ctrls]
    #         rho = np.reshape(rho, (-1, 1))  # [ctrls,1]
    #         w = rho / np.sum(rho)  # [ctrls,1]
    #         fi = q - p + curr_pt  # [ctrls, 2]
    #         f = np.sum(w * fi, axis=0)  # [2]
    #
    #         f[0] = np.clip(f[0], 0, width - 1)
    #         f[1] = np.clip(f[1], 0, height - 1)
    #         f = f.astype(np.int)
    #
    #         img_warped[f[1], f[0]] = image[curr_pt[1], curr_pt[0]]
    #
    # return img_warped
