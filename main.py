import cv2
import numpy as np
import math
from scipy import signal
import os


def gaussian_kernel(img_v, r, rho):
    PI = math.pi
    size = 2 * rho + 1
    g1 = np.zeros((size, size), dtype=float)
    a = 0.0
    for ii in range(-rho, rho + 1):
        for jj in range(-rho, rho + 1):
            g1[ii + rho, jj + rho] = 1. / (2 * PI * r * r) * math.exp(-((ii * ii) + (jj * jj)) / (2 * r * r))
            a += g1[ii + rho, jj + rho]
    g1 /= a
    new_arr = signal.convolve2d(img_v, g1, mode="same", boundary="symm")
    new_arr = np.clip(new_arr, 0, 255)
    return new_arr


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def multiscale_retinex(image, r_list, rho_list):
    illumination_maps = []
    epsilon = 1e-6  # 非零小值
    image = np.where(image == 0, epsilon, image)

    weights = [1.0 / 3.0] * len(r_list)  # 三个尺度的权重，每个尺度的权重都为 1/3

    for r, rho, weight in zip(r_list, rho_list, weights):
        # 应用高斯核
        blurred = gaussian_kernel(image, r, rho)

        # 计算光照分量
        illumination = np.log10(blurred+1.0)

        # 将光照分量缩放到0-255范围
        illumination = (illumination - np.min(illumination)) / (np.max(illumination) - np.min(illumination)) * 255



        illumination_maps.append(illumination)


    combined_illumination = np.mean(illumination_maps, axis=0)

    combined_illumination = (combined_illumination - np.min(combined_illumination)) / (
            np.max(combined_illumination) - np.min(combined_illumination)) * 255


    return combined_illumination

def computed_data(image_name):
    r1 = 5
    r2 = 25
    r3 = 80
    r_list = [r1, r2, r3]
    rho_list = [3 * r1, 3 * r2, 3 * r3]

    illumination_image = multiscale_retinex(image_name, r_list, rho_list)
    median_illumination = np.median(illumination_image)
    average_illmination = np.mean(illumination_image)
    return (median_illumination + average_illmination)/2


def clahe(image):
    # 创建CLAHE对象并设定参数
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image)
    return image_clahe




def msr_color(img, r_list, rho_list):
    alpha = 255
    if computed_data(img) < 120:
        equalized_image = cv2.equalizeHist(img)
        beta = computed_data(equalized_image)
    else:
        beta = computed_data(img)

    img = replaceZeroes(img)

    L1 = gaussian_kernel(img, r_list[0], rho_list[0])
    L1 = np.clip(L1, 8, 255)
    r1 = alpha * np.log10(np.true_divide(img, L1)) + beta
    r1 = np.clip(r1, 0, 255)
    print("r1计算完成")

    L2 = gaussian_kernel(img, r_list[1], rho_list[1])
    r2 = alpha * np.log10(np.true_divide(img, L2)) + beta
    r2 = np.clip(r2, 0, 255)
    print("r2计算完成")

    L3 = gaussian_kernel(img, r_list[2], rho_list[2])
    r3 = alpha * np.log10(np.true_divide(img, L3)) + beta
    r3 = np.clip(r3, 0, 255)
    print("r3计算完成")

    return r1, r2, r3


if __name__ == '__main__':
    image_name = 'girl'
    r1 = 5
    r2 = 25
    r3 = 80
    r_list = [r1, r2, r3]
    src_img = cv2.imread('image/' + image_name + '.ppm')
    result_path = './Adaptive_New_way_image_result/' + image_name + '/'
    os.makedirs(result_path, exist_ok=True)
    rho_list = [3 * r1, 3 * r2, 3 * r3]
    hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    ####################new_msr#################
    V_r1, V_r2, V_r3 = msr_color(v, r_list, rho_list)

    v_new = 1.0 / 3.0 * (V_r1 + V_r2 + V_r3)
    v_new = np.array(v_new, dtype=np.uint8)

    height, weight = v.shape[:2]
    for i in range(height):
        for j in range(weight):
            if v_new[i][j] / v[i][j] < 1:
                v_new[i][j] = v[i][j]

    hsv_msr = np.zeros(hsv.shape, dtype=np.uint8)
    hsv_msr[:, :, 0] = h  # H通道不变
    hsv_msr[:, :, 1] = s  # S通道不变
    hsv_msr[:, :, 2] = v_new
    new_msr = cv2.cvtColor(hsv_msr, cv2.COLOR_HSV2BGR)

    ##############显示图片#####################
    cv2.imshow('src_img', src_img)

    cv2.imshow('NewMSR', new_msr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

