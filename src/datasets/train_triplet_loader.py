import os.path

import torch
import torch.utils.data as data
import os.path as osp
import glob
import sys
from PIL import Image
import cv2 as cv
from torchvision import transforms

sys.path.append("../")
supported_image_ext = [".png", ".bmp", ".jpg", ".jpeg"]
labels = []


class TripletDataset(data.Dataset):
    def __init__(self, root, image_size, shuffle=False, use_cache=True):
        self.root = root
        self.list = self.get_file_list()
        self.nb_samples = len(self.list)
        self.phase = 0
        self.transform = True
        self.image_size = image_size

    def get_file_list(self):
        local_list = []
        folder_list = glob.glob(self.root + "/*") #返回所有匹配的文件路径列表 *匹配任意字符

        for d in folder_list:
            bOk = True
            image_list = glob.glob(d + "/*")

            # TODO what is this if statement ??
            if not len(image_list) == 3:
                continue

            for i in image_list:
                #分离文件名与拓展名
                bOk = bOk and osp.splitext(i)[1] in supported_image_ext

            # check same extension
            ext_0 = osp.splitext(image_list[0])[1]
            ext_1 = osp.splitext(image_list[1])[1]
            ext_2 = osp.splitext(image_list[2])[1]

            if not (ext_0 == ext_1 == ext_2):
                pass

            # check for a, n, and p
            folder = osp.split(image_list[0])[0] #返回文件的路径和文件名
            anchor = folder + "/a" + ext_0
            positive = folder + "/p" + ext_0
            negative = folder + "/n" + ext_0

            # Check whether the specified path exists or not 检查指定的路径是否存在
            bOk = bOk and osp.exists(anchor)
            bOk = bOk and osp.exists(positive)
            bOk = bOk and osp.exists(negative)

            if bOk:
                local_list.append({"a": anchor, "p": positive, "n": negative})

        #print(local_list)
        return local_list

    def __getitem__(self, item):
        triplet_dict = self.list[item]
        # anchor patch
        img = Image.open(triplet_dict["a"]).convert('L')
        # img_cropped = img.crop((0, 0, self.image_size, self.image_size))
        img_cropped = img.resize((self.image_size, self.image_size))
        anchor_image = transforms.ToTensor()(img_cropped)

        # positive patch
        img = Image.open(triplet_dict["p"]).convert('L')
        # img_cropped = img.crop((0, 0, self.image_size, self.image_size))
        img_cropped = img.resize((self.image_size, self.image_size))
        positive_image = transforms.ToTensor()(img_cropped)
        # print('positive_image:',positive_image.shape)

        # negative patch
        img = Image.open(triplet_dict["n"]).convert('L')
        # img_cropped = img.crop((0, 0, self.image_size, self.image_size))
        img_cropped = img.resize((self.image_size, self.image_size))
        negative_image = transforms.ToTensor()(img_cropped)

        input_tensor = torch.cat([anchor_image, positive_image, negative_image])
        # print(input_tensor.shape)
        return item, input_tensor

    def __len__(self):
        return self.nb_samples
#
# if __name__ == "__main__":
#     train_data = '../../data/optical/train/triplets_256'
#     image_size = 256
#     TripletDataset(train_data,image_size)