import cv2
import os
import util
import os.path as osp
from PIL import Image
import numpy as np


class UnsupportedFormat(Exception):
    def __init__(self, input_type):
        self.t = input_type

    def __str__(self):
        return "不支持'{}'模式的转换，请使用为图片地址(path)、PIL.Image(pil)或OpenCV(cv2)模式".format(self.t)


class MatteMatting():
    def __init__(self, original_graph, mask_graph, input_type='path'):
        """
        将输入的图片经过蒙版转化为透明图构造函数
        :param original_graph:输入的图片地址、PIL格式、CV2格式
        :param mask_graph:蒙版的图片地址、PIL格式、CV2格式
        :param input_type:输入的类型，有path：图片地址、pil：pil类型、cv2类型
        """
        self.img1 = cv2.imread(original_graph, cv2.IMREAD_UNCHANGED)
        self.img2 = cv2.imread(mask_graph, cv2.IMREAD_UNCHANGED)
        if self.img1.shape[2] == 3:
            b_channel, g_channel, r_channel = cv2.split(self.img1)
            # creating a dummy alpha channel image.
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 0
            self.img1 = cv2.merge(
                (b_channel, g_channel, r_channel, alpha_channel))

        # print(self.img1.shape, self.img2.shape)

    @staticmethod
    def __transparent_back(img):
        """
        :param img: 传入图片地址
        :return: 返回替换白色后的透明图
        """
        img = img.convert('RGBA')
        L, H = img.size
        color_0 = (255, 255, 255, 255)  # 要替换的颜色
        for h in range(H):
            for l in range(L):
                dot = (l, h)
                color_1 = img.getpixel(dot)
                if color_1 == color_0:
                    color_1 = color_1[:-1] + (0,)
                    img.putpixel(dot, color_1)
        return img

    def save_image(self):
        """
        用于保存透明图
        :param path: 保存位置
        """
        # 先向外扩展mask边界，然后向内腐蚀平滑边缘曲线
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # self.img2 = cv2.dilate(self.img2, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.img2 = cv2.erode(self.img2, kernel)
        # self.img2 = cv2.medianBlur(self.img2, 5)
        self.img2 = cv2.medianBlur(self.img2, 9)
        image = cv2.add(self.img1, self.img2)

        return image

    @staticmethod
    def __image_to_opencv(image):
        """
        PIL.Image转换成OpenCV格式
        """
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return img


# mm = MatteMatting('3.jpg', '3_jpg_result.png')
# print(mm.save_image())
