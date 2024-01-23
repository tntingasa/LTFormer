import numpy as np

import torch
import os
import cv2 as cv
from torch.utils.data import Dataset
from scipy.spatial import distance
from numpy import loadtxt
import glob
import os.path as osp
import tqdm


class PatchDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, root_path, patch_size):
        self.image_name = []
        self.keypoints_GT = []
        self.root_path = root_path
        self.patch_size = patch_size #64

        self.all_frames_per_sequence = []
        self.all_keypoints = []
        self.data = []
        self.load_data_path()

    def __len__(self):
        return len(self.data)

    def load_data_path(self):
        # get list of  sequences
        sequence_list = glob.glob(self.root_path + "/*")
        # print(sequence_list)

        # got through sequences
        for sequence in sequence_list:
            #  first get all gt matches for the current sequence
            # sorted 返回的是一个新的 list，而不是在原来的基础上进行的操作。
            curr_matches_list = sorted(glob.glob(sequence+"/matches/*"))
            for match_path in curr_matches_list:
                folder_name = os.path.basename(match_path)
                curr_matches = sorted(glob.glob(match_path +"/*"))
                # print(curr_matches)
                for match in curr_matches:
                    # get src and dst frame filenames
                    file_path = os.path.splitext(match)[0]
                    file_name = os.path.basename(file_path)
                    src_filename = file_name+".png"
                    dst_filename = file_name+".png"
                    src_abs_path = osp.join(sequence,"rgb",folder_name, src_filename)
                    dst_abs_path = osp.join(sequence,"NIRS",folder_name, dst_filename)
                    #print(src_abs_path)
                    # if all related data exist
                    matchpoint = loadtxt(match, dtype='float')
                    if matchpoint.size == 0:
                        print("******")
                        continue
                    if osp.exists(src_abs_path) and osp.exists(dst_abs_path):
                        # print("************")
                        self.data.append({"match": match, "src_frame": src_abs_path, "dst_frame": dst_abs_path})

    def enhance(self, img):
        #crop_img = img[70 : int(img.shape[0]) - 70, 50 : int(img.shape[1]) - 40]
        gray2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=5)
        image_enhanced = clahe.apply(gray2)
        # image_enhanced = cv.equalizeHist(gray2)
        return image_enhanced
    
    def enhance_2(self,img):
        # 将图像转换为 LAB 颜色空间
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        # 将 LAB 图像分离为 L、A、B 三个通道
        l, a, b = cv.split(lab)
        # 创建 CLAHE 对象
        clahe = cv.createCLAHE(clipLimit=5.0)
        # 对 L 通道进行直方图均衡化
        l_enhanced = clahe.apply(l)
        # 合并增强后的 L 通道和原始的 A、B 通道
        lab_enhanced = cv.merge((l_enhanced, a, b))
        # 将增强后的 LAB 图像转换回彩色图像
        image_enhanced = cv.cvtColor(lab_enhanced, cv.COLOR_LAB2BGR)
        return image_enhanced

    def __getitem__(self, idx):
        frame_src = cv.imread(self.data[idx]["src_frame"], 3) #flag=3 原深度 3通道
        Next_frame = cv.imread(self.data[idx]["dst_frame"], 3)

        ori_src = cv.cvtColor(frame_src, cv.COLOR_BGR2RGB)
        ori_next = cv.cvtColor(Next_frame, cv.COLOR_BGR2RGB)

        # frame_src = self.enhance_2(frame_src)
        # Next_frame = self.enhance_2(Next_frame)
        frame_src = self.enhance(frame_src)
        Next_frame = self.enhance(Next_frame)

        image_path = self.data[idx]["src_frame"]
        file_path = os.path.splitext(image_path)[0]
        folder_name = file_path.split('/')[-2]

        # keypoints filenames
        matches = self.data[idx]["match"]

        list_matches = loadtxt(matches, dtype='float') #match为读取地址 有2个点 四个值

        list_keypoints_src = []
        list_keypoints_next_frame = []

        for i in range(0, len(list_matches)):
            list_keypoints_src.append((list_matches[i][0], list_matches[i][1]))
            list_keypoints_next_frame.append((list_matches[i][2], list_matches[i][3]))
        # h, w, _ = frame_src.shape
        h, w = frame_src.shape

        # ---------------------------------------------------Generate_data-----------------------------------------------------------
        i = 0
        gt_key_src = []
        gt_key_next_frame = []
        patches_src = []
        patches_next_frame = []
        i = 0
        # ---------------------------------------------------Generate_data-----------------------------------------------------------

        for b in range(0, len(list_keypoints_src)):

            xa = int(list_keypoints_src[b][0])
            ya = int(list_keypoints_src[b][1])
            xp = int(list_keypoints_next_frame[b][0])
            yp = int(list_keypoints_next_frame[b][1])

            if (
                    ((ya - self.patch_size) > 0)
                    & ((xa - self.patch_size) > 0)
                    & ((ya + self.patch_size) < h)
                    & ((xa + self.patch_size) < w)
                    & ((yp - self.patch_size) > 0)
                    & ((xp - self.patch_size) > 0)
                    & ((yp + self.patch_size) < h)
                    & ((xp + self.patch_size) < w)
            ):
                crop_patches_src = frame_src[
                    ya - self.patch_size : ya + self.patch_size, xa - self.patch_size : xa + self.patch_size
                ]
                crop_patches_next_frame = Next_frame[
                    yp - self.patch_size : yp + self.patch_size, xp - self.patch_size : xp + self.patch_size
                ]

                patches_next_frame.append(crop_patches_next_frame)
                patches_src.append(crop_patches_src)
                gt_key_src.append(list_keypoints_src[i])
                gt_key_next_frame.append(list_keypoints_next_frame[i])
                i += 1
        # if not patches_src :
        #     print(self.data[idx]["src_frame"])
        return {
            #图像
            "image_src_name": frame_src,
            "image_dst_name": Next_frame,
            #patch队列
            "patch_src": patches_src,
            "patch_dst": patches_next_frame,
            #标签信息
            "keypoint_src": gt_key_src,
            "keypoint_dst": gt_key_next_frame,
            #图像名
            "image_name":folder_name,
            "ori_src": ori_src,
            "ori_dst": ori_next
        }

# if __name__ == "__main__":
#     validation_data_root = '../../data/optical/test/'
#     patch_size = 64
#     test_dataset=PatchDataset(validation_data_root, patch_size)
    #print(tqdm.tqdm(test_dataset))