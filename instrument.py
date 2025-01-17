import math
import os
import openpyxl
import cv2
from astropy.table import Table
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim
import numpy as np
from astropy.modeling.models import Tabular2D
from sklearn.linear_model import LinearRegression

reference_magnitude = 0  # 9.111
reference_flux = 0  # 2396.2024  4279801.068621376 25.6873668671


def turn_star_coordinate(star, r):
    new_star_coordinate = (star[0] * r + r / 2, star[1] * r + r / 2)
    return new_star_coordinate


def calculate_ae(image1, image2):
    """
    计算image1和image2的平均误差
    :param image1:
    :param image2:
    :return:
    """
    err = np.sum(image1 - image2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err


def circular_damping(size, r, value=0.0):
    """
    返回一个大小为（size，size），中心一个半径为r的区域没有阻尼，其余区域乘于阻尼值value
    :param size:整个图像大小
    :param r: 距离中心r个像素的区域没有阻尼
    :param value: 除中间圆形没有阻尼区域外，其他区域阻尼大小value
    :return:
    """
    # 圆形阻尼
    mask = np.full(size, value)
    for i in range(size[0]):
        for j in range(size[1]):
            if math.sqrt((i - int(size[0] / 2)) ** 2 + (j - int(size[1] / 2)) ** 2) < r:
                mask[i][j] = 1
    return mask


def map_to_small_image(large_image, small_image, alignment_point):
    """
    将大尺寸图像的指定对齐坐标alignment_point和小尺寸图像的中心对齐
    将大尺寸图像的部分映射到小尺寸图像上
    :param large_image:
    :param small_image:
    :param alignment_point:
    :return: large_image
    """
    # 小尺寸图像的中心
    small_center = np.array(small_image.shape) // 2

    if small_image.shape[0] % 2 == 0:
        small_center[0] -= 1
    if small_image.shape[1] % 2 == 0:
        small_center[1] -= 1

    # 偏差
    alignment_point = tuple(elem - 0.5 for elem in alignment_point)
    bias_x = alignment_point[0] - np.floor(alignment_point[0])
    bias_y = alignment_point[1] - np.floor(alignment_point[1])

    # 计算每个像素的贡献权重
    w1 = bias_x * bias_y
    w2 = bias_x * (1 - bias_y)
    w3 = (1 - bias_x) * bias_y
    w4 = (1 - bias_x) * (1 - bias_y)

    # 计算在小尺寸图像在大尺寸图像中放置的起始坐标
    start_position = (np.floor(alignment_point) - small_center).astype(int)
    # 计算终止坐标是否在大尺寸图像中
    end_position = (alignment_point + small_center) - np.array(large_image.shape) + np.array((1, 1))

    # 在大尺寸的外围加1圈0，预防边界出错
    rows, cols = large_image.shape
    image = np.zeros((rows + 2, cols + 2), dtype=np.float64)
    image[1:-1, 1:-1] = large_image
    large_image = image

    # 判断是否已经超出边界
    # if start_position[0] < 0 or start_position[1] < 0 or end_position[0] > 0 or end_position[1] > 0:
    #     return
    # else:
    for i in range(0, small_image.shape[0]):
        for j in range(0, small_image.shape[1]):
            x = i + start_position[0] + 1
            y = j + start_position[1] + 1
            v1, v2, v3, v4 = 0, 0, 0, 0
            if 0 < x < large_image.shape[0] and 0 < y < large_image.shape[1]:
                v4 = large_image[x, y]
            if 0 < x + 1 < large_image.shape[0] and 0 < y + 1 < large_image.shape[1]:
                v1 = large_image[x + 1, y + 1]
            if 0 < x + 1 < large_image.shape[0] and 0 < y < large_image.shape[1]:
                v2 = large_image[x + 1, y]
            if 0 < x < large_image.shape[0] and 0 < y + 1 < large_image.shape[1]:
                v3 = large_image[x, y + 1]

            small_image[i, j] = small_image[i, j] + (w4 * v4 + w2 * v2 + w3 * v3 + w1 * v1)
    return small_image


def map_to_large_image(large_image, small_image, alignment_point):
    """
    将小尺寸图像的中心和大尺寸图像中指定的对齐坐标(可以是浮点数)对齐，将小尺寸图像全部映射到大尺寸图像上
    有待改进：如果指定中心坐标超出A的范围，不添加。由于本项目不会出现这种情况，所以未实现
    :param large_image:
    :param small_image:
    :param alignment_point:
    :return: large_image
    """
    # 小尺寸图像的中心
    small_center = np.array(small_image.shape) // 2

    if small_image.shape[0] % 2 == 0:
        small_center[0] -= 1
    if small_image.shape[1] % 2 == 0:
        small_center[1] -= 1

    # 在小尺寸图像的外围加1圈0
    rows, cols = small_image.shape
    image = np.zeros((rows + 2, cols + 2), dtype=np.float64)
    image[1:-1, 1:-1] = small_image
    small_image = image

    # 计算相对偏差
    alignment_point = tuple(elem - 0.5 for elem in alignment_point)
    bias_x = alignment_point[0] - np.floor(alignment_point[0])
    bias_y = alignment_point[1] - np.floor(alignment_point[1])

    # 计算每个像素贡献权重
    w1 = bias_x * bias_y
    w2 = bias_x * (1 - bias_y)
    w3 = (1 - bias_x) * bias_y
    w4 = (1 - bias_x) * (1 - bias_y)

    # 计算在小尺寸图像在大尺寸图像对应的的起始坐标
    start_position = np.floor(alignment_point) - small_center
    # 将起始坐标限制在（0，large_image.shape）范围
    start_position = np.clip(start_position, 0, np.array(large_image.shape) - np.array((1, 1))).astype(int)
    # 计算在小尺寸图像在大尺寸图像对应的的终止坐标
    end_position = start_position + np.array((rows, cols))
    # 将终止坐标限制在（0，large_image.shape）范围
    end_position = np.clip(end_position, 0, np.array(large_image.shape) - np.array((1, 1))).astype(int)

    # 遍历计算大尺寸在起始坐标到终止坐标的像素值
    for i in range(start_position[0], end_position[0] + 1):
        for j in range(start_position[1], end_position[1] + 1):
            x = i - start_position[0] + 1
            y = j - start_position[1] + 1
            large_image[i, j] = large_image[i, j] + (w1 * small_image[x - 1, y - 1] + w2 * small_image[x - 1, y]
                                                     + w3 * small_image[x, y - 1] + w4 * small_image[x, y])
    # 返回大尺寸图像
    return large_image


def sum_downsample(image, stride):
    """
    将image的stride*stride个像素求和，作为下采样的1个像素的值。
    :param image:
    :param stride:
    :return:
    """
    # 获取原图像的高度和宽度
    height, width = image.shape[:2]

    # 计算下采样后的图像大小
    new_height = height // stride
    new_width = width // stride

    # 初始化下采样后的图像
    downsampled_image = np.zeros((new_height, new_width), dtype=np.float64)

    # 遍历原图像，每3x3的像素相加得到一个新的像素
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            roi = image[i:i + stride, j:j + stride]
            average_pixel_value = np.sum(roi)
            downsampled_image[i // stride, j // stride] = average_pixel_value
    return downsampled_image


def superpixelize_image(image, new_shape, r):
    """
    将图像细化r倍到new_shape尺寸大小
    :param image:
    :param new_shape:
    :param r:
    :return:
    """
    # 初始化细化后的图像
    refined_image = np.zeros(new_shape, dtype=np.float64)
    for y in range(image.shape[0]):  # 高
        for x in range(image.shape[1]):  # 宽
            pixel = image[y, x]
            refined_image[y * r:(y + 1) * r, x * r:(x + 1) * r] = pixel / (r * r)
    return refined_image


def chart(times, *PSF):
    """
    输入多个相同大小的二维数组，展示中心水平切片和对角线切片图像
    :param times: 迭代次数
    :param PSF:
    :return:
    """
    # （水平）创建图像
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in PSF:
        for name, value in p.items():
            # 沿水平方向切片
            horizontal_slice = value[int(value.shape[1] / 2), :]
            # 绘制水平切片
            # ax.plot(horizontal_slice, label=f'{name}')
            ax.plot(horizontal_slice,
                    label=f'{name}')  # (Max: {np.max(horizontal_slice):.6f}, Min: {np.min(horizontal_slice):.6f})')

    # 添加图例
    ax.legend()
    ax.set_title('Iterations : ' + str(times) + ', Horizontal Cut Comparison')
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Normalized Intensity')
    # 显示图像
    plt.show()

    # (对角)创建图像
    fig1, ax = plt.subplots(figsize=(10, 6))
    for p in PSF:
        for name, value in p.items():
            # 沿对角线（左上-右下）切片
            diagonal_slice = np.diagonal(value)
            # 绘制对角线切片
            # ax.plot(diagonal_slice, label=f'{name}')
            ax.plot(diagonal_slice,
                    label=f'{name}')  # (Max: {np.max(diagonal_slice):.6f}, Min: {np.min(diagonal_slice):.6f})')

    # 添加图例
    ax.legend()
    ax.set_title('Iterations: ' + str(times) + ', Diagonal Cut Comparison')
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Normalized Intensity')
    # 显示图像
    plt.show()
    return


def get_star_list(file_name):
    """
    读取file_name里的星源中心坐标，和星等，其中文件为QMPF文件
    :param file_name:
    :return:
    """
    # 打开文件
    with open(file_name, 'r', encoding='utf-8') as file:
        # 读取文件的所有行到一个列表中
        lines = file.readlines()
    star_list = []
    raw_star_list = []

    for line in lines:
        data = line.split()
        raw_star_list.append((float(data[6]), float(data[7])))
        if 16 > float(data[12]) > 5:
            star_list.append((float(data[6]), float(data[7])))

    return star_list, raw_star_list
