from PIL import Image

from instrument import *


# Step 1: 将图像M的所有像素细化r倍
def resample_image(image, r):
    # 细化后的图像尺寸大小
    new_shape = (r * image.shape[0], r * image.shape[1])
    # 细化
    resize_img = superpixelize_image(image, new_shape, r)
    return resize_img


def cut(image, star, size):
    x0, y0 = tuple(map(int, star))  # 思考偏移是否正确
    top = x0 - int(size / 2)
    left = y0 - int(size / 2)
    bottom = x0 + int(size / 2) + 1
    right = y0 + int(size / 2) + 1
    if top > 0 and bottom < image.shape[0] and left > 0 and right < image.shape[1]:
        return image[top:bottom, left:right]
    return


# Step 2: 以点源中心为中心剪出一个正方形区域
def cut_thumbnail(resampled_image, star, mask_size=101):
    T = np.zeros((mask_size, mask_size))
    # 以star为中心在截取细化后的图像中截取T大小的像素
    T = map_to_small_image(resampled_image, T, (star[0], star[1]))
    if T is None:
        return None
    return T


# Step 3: 抑制区域的整体亮度
def suppress_offset(mask):
    padding = 5
    image = mask.copy()
    image[padding:-padding, padding:-padding] = 0
    background = image.sum() / (2 * padding * (image.shape[0] + image.shape[1] - 2 * padding))
    mask = mask - background
    mask[mask < 0] = 0
    return mask


# Step 4: 将某个点源的区域添加到堆栈中
def add_to_stack(stack, mask):
    stack += mask
    return stack


# Step 5: 对堆栈取平均值，得到所求psf
def compute_stacked_psf(stack, length):
    psf = stack / length
    return psf


def stack_star(image, r, star_list, mask_size):
    print("细化图像...")
    # 细化
    resize_img = resample_image(image, r)
    resize_img[resize_img < 0] = 0

    # 记录堆叠后合法的星源坐标，排除坐落图像边缘的星源
    new_star_list = []
    star_stack = []
    res_star_list = []

    # 初始化堆栈
    stack = np.zeros((mask_size, mask_size))
    # 切割星源、堆叠
    for index, star in enumerate(star_list, start=1):
        print(f'处理源{index}：', star, "--- ", end="")
        # 计算细化后星源的中心坐标
        (center_x, center_y) = turn_star_coordinate(star, r)

        # 以细化后的中心为区域中心剪切出一个区域
        thumbnail = cut_thumbnail(resize_img, (center_x, center_y), mask_size)

        if thumbnail is None:
            print("源范围超出边界！剔除！")
        else:
            # 将符合的星源坐标加入到列表中
            res_star_list.append(star)
            # 抑制整体亮度（减去背景），理想图像不用减，减去周围10像素的平均值
            thumbnail = suppress_offset(thumbnail)
            # 添加到堆栈
            stack = add_to_stack(stack, thumbnail)
            star_stack.append(thumbnail)
            print("源添加成功！")
    print(f"剩下源个数：{len(res_star_list)}")
    print("计算Ps...", end="")
    # 取堆栈平均，求得PS
    ps = compute_stacked_psf(stack, len(res_star_list))
    return ps, res_star_list
