import cv2
from scipy.spatial import distance
from scipy.ndimage import zoom
import random
import numpy as np


def enhance_image(img):
    # crop_img = img[70 : int(img.shape[0]) - 70, 50 : int(img.shape[1]) - 50]
    # gray2 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) #BGR->GRAY
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR->GRAY
    gray2 = cv2.blur(gray2, (5, 5))
    clahe = cv2.createCLAHE(clipLimit=5)
    image_enhanced = clahe.apply(gray2)
    return image_enhanced


def enhance_image_2(img):
    # 裁剪图像
    # crop_img = img[70:img.shape[0] - 70, 50:img.shape[1] - 50]
    # 对彩色图像进行平滑处理
    # blurred = cv2.blur(crop_img, (5, 5))
    blurred = cv2.blur(img, (5, 5))
    # 将平滑后的图像转换为 LAB 色彩空间
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    # 分割 LAB 色彩空间的通道
    l, a, b = cv2.split(lab)
    # 应用自适应直方图均衡化到 L 通道
    clahe = cv2.createCLAHE(clipLimit=5)
    enhanced_l = clahe.apply(l)
    # 合并增强后的 L 通道和原始 a、b 通道
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    # 转换回 BGR 色彩空间
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_img




def zoom_coordinates(img, x, y, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        y1 = (y + ((h - zh) // 2)) * zoom_factor
        x1 = (x + ((w - zw) // 2)) * zoom_factor
        # Zero-padding
        out = np.zeros_like(img)
        out[top : top + zh, left : left + zw] = zoom(img, zoom_tuple, **kwargs)
        y1 = np.int32(y1)
        x1 = np.int32(x1)

    # Zooming in
    elif zoom_factor >= 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        y1 = (y - ((h - zh) // 2)) * zoom_factor
        x1 = (x - ((w - zw) // 2)) * zoom_factor
        out = zoom(img[top : top + zh, left : left + zw], zoom_tuple, **kwargs)
        y1 = np.int32(y1)
        x1 = np.int32(x1)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        # trim_top = ((out.shape[0] - h) // 2)
        # trim_left = ((out.shape[1] - w) // 2)
        # out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return x1, y1


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        # Zero-padding
        out = np.zeros_like(img)
        out[top : top + zh, left : left + zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor >= 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top : top + zh, left : left + zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = (out.shape[0] - h) // 2
        trim_left = (out.shape[1] - w) // 2
        out = out[trim_top : trim_top + h, trim_left : trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def extract_image_keypoints(image, extractor_id):
    assert extractor_id == "SIFT"
    sift = cv2.SIFT_create(400)
    #sift = cv2.SIFT_create(400)
    keypoints, features = sift.detectAndCompute(image, None) #产生关键点位置跟特征向量
    return keypoints, features
