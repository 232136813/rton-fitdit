import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图
img = cv2.imread("/Users/kevin/Desktop/abc.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("无法读取图片，请检查路径。")
else:
    img_float = np.float32(img)
    img_dct = cv2.dct(img_float)

    # 【核心改进1】：对数变换，初步放大暗部
    img_dct_log = np.log(np.abs(img_dct) + 1)

    # 【核心改进2】：强行进行线性拉伸，把 0-255 的对比度完全充满
    img_dct_visual = cv2.normalize(img_dct_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 绘图展示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')

    # 使用 'jet' 伪彩色地图，能把肉眼看不清的黑色渐变，用蓝、绿、红不同的颜色区分开
    plt.subplot(1, 2, 2)
    plt.title("Watermark Signal (Enhanced Color)")
    plt.imshow(img_dct_visual, cmap='jet')
    plt.colorbar()
    plt.show()
