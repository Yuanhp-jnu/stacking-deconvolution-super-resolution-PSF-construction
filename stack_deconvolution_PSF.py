import time
from astropy.io import fits
from skimage import restoration
from stack import *


class StackDeconvolutionPSF:
    image = [[]]
    r = 0
    stack_psf = [[]]
    rld_iterations = 0
    ibp_iterations = 0
    multi_star_list = []
    mask = []
    f_true = [[]]
    stack_size = 0
    res_star_list = []

    def __init__(self, image_file_name, point_file, r, rld_iterations, ibp_iterations, stack_size, radius, factor=0):
        """
        创建堆叠-去卷积对象，生成堆叠PSF
        :param image_file_name: 图像路径
        :param point_file: 指向文件路径（主要是用于获取星源中心）
        :param r: 超分辨率倍数
        :param rld_iterations: RLD去卷积迭代次数
        :param ibp_iterations: IBP去卷积迭代次数
        :param stack_size: 堆栈尺寸
        :param radius: 圆形掩模半径
        :param factor: 圆形掩模半径外权重
        """
        # 检查目标文件是否存在
        if os.path.isfile(image_file_name) and os.path.isfile(point_file):
            print(f'图像路径：', image_file_name)
            # 读取fits图像文件
            self.image = fits.getdata(image_file_name)
            # 获取星源中心坐标
            star_list, raw_star_list = get_star_list(point_file)
            self.multi_star_list.append(star_list)
        self.r = r
        self.rld_iterations = rld_iterations
        self.ibp_iterations = ibp_iterations
        self.stack_psf = np.zeros((stack_size, stack_size))
        self.mask = circular_damping((stack_size, stack_size), radius, factor)
        self.stack_size = stack_size
        self.stack()

    def create_pgrid(self):
        """
        生成像素网格响应函数
        :return:
        """
        # 初始Pgrid
        grid = np.zeros_like(self.stack_psf)
        # 计算PSF中心位置
        x0 = np.ceil(self.stack_psf.shape[0] / 2) - 1
        y0 = np.ceil(self.stack_psf.shape[1] / 2) - 1
        # 计算Pgrid，在距离中心的(-r, +r)的范围就计算，其他全部设为0
        for i in range(int(x0 - self.r), int(x0 + self.r + 1)):
            for j in range(int(y0 - self.r), int(y0 + self.r + 1)):
                grid[i, j] = (self.r - np.abs(i - x0)) * (self.r - np.abs(j - y0))

        grid[grid < 0] = 0

        return grid

    def rld_lib(self):
        """
        RLD去卷积
        :return:
        """
        # 生成模糊因子
        grid = self.create_pgrid()
        # 使用模糊因子Pgrid对PSF进去RLD去卷积
        f_rld = restoration.richardson_lucy(self.stack_psf, grid, num_iter=self.rld_iterations, clip=False)
        return f_rld

    def iterative_back_project_deconvolution(self):
        """
        IBP去卷积
        :return:
        """
        # 根据每颗星源坐标对ps进行下采样
        f_s = self.stack_psf / np.sum(self.stack_psf)
        f_sa = f_s.copy()
        f_rld = np.zeros_like(f_s)
        r = self.r
        for _ in range(self.ibp_iterations):
            print(f"第{_ + 1}次迭代")
            start_time = time.time()
            # RLD去卷积
            f_rld = self.rld_lib()

            k = 0.0
            f_sn_list = np.zeros_like(f_s)
            for image_index, star_list in enumerate(self.multi_star_list):
                for star in star_list:
                    # 将prl放置在mask_size大小的区域，要比prl尺寸大2*r，且能r倍下采样
                    mask_size = ((int(f_s.shape[0] / r) + 2) * r, (int(f_s.shape[1] / r) + 2) * r)
                    # 处理PSF
                    large_mask = np.zeros(mask_size)
                    (x0, y0) = turn_star_coordinate(star, r)
                    x_offset = x0 % r
                    y_offset = y0 % r
                    x = int(int(mask_size[0] / r) / 2) * r + x_offset
                    y = int(int(mask_size[1] / r) / 2) * r + y_offset

                    large_mask = map_to_large_image(large_mask, f_rld, (x, y))
                    # 下采样
                    sub_mask = sum_downsample(large_mask, r)
                    # 上采样
                    resample_mask = superpixelize_image(sub_mask, mask_size, r)
                    star_mask = np.zeros_like(f_s)
                    star_mask = map_to_small_image(resample_mask, star_mask, (x, y))
                    add_to_stack(f_sn_list, star_mask)
                    # 星源数加1
                    k += 1

            psf_mean = compute_stacked_psf(f_sn_list, k)
            psf_mean = psf_mean / np.sum(psf_mean)
            psf_mean = psf_mean * self.mask
            err = psf_mean - f_s
            print(f"第{_ + 1}次迭代误差：{calculate_ae(psf_mean, f_s)}")
            f_sa = f_sa - err
            f_sa[f_sa < 0] = 0
            f_sa = f_sa * self.mask
            # 展示结果
            p1 = {r'$F_s$': f_s / np.sum(f_s)}
            p2 = {r'$F_S^A$': f_sa / np.sum(f_sa)}
            p3 = {r'$F_{RLD}$': f_rld / np.sum(f_rld)}
            p4 = {r'$F_{true}$': self.f_true / np.sum(self.f_true)}
            p5 = {r'$F_S^N$': psf_mean / np.sum(psf_mean)}
            chart(_ + 1, p1, p2, p3, p4, p5)

            end_time = time.time()
            print(f"{end_time - start_time}秒")
        return f_rld, f_s

    def stack(self):
        f_s, res_star_list = stack_star(self.image, self.r, self.multi_star_list[0], self.stack_size)
        self.stack_psf = f_s
        self.res_star_list = res_star_list
