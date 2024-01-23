import glob
import os
import os.path as osp
import logging

from _version import __version__
import hydra
import json
from omegaconf import OmegaConf

import cv2
from scipy.spatial import distance
from scipy.ndimage import zoom
import random
import numpy as np

from src.utils.image_keypoints_extractors import extract_image_keypoints, enhance_image,enhance_image_2,zoom_coordinates,clipped_zoom

logger = logging.getLogger("Triplet-Dataset-Generator")
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config_dataset")
def main(cfg):
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)
    logger.info("Start Processing raw sequence")

    input_rgb_image_lists = []
    input_NIRS_image_lists = []


    # for file in cfg.paths.rgb_data_dirs:
    #     input_rgb_image_lists.append(sorted(glob.glob(file)))
    # for input_folder2 in cfg.paths.NIRS_data_dirs:
    #     input_NIRS_image_lists.append(sorted(glob.glob(input_folder2 + "/*.png")))

    input_rgb_image_lists = sorted(glob.glob(cfg.paths.rgb_data_dirs + "/*"))
    input_NIRS_image_lists = sorted(glob.glob(cfg.paths.NIRS_data_dirs + "/*"))

    # process sequence folder one by one
    #for input_image_list, folder_name in zip(input_rgb_image_lists, cfg.paths.rgb_data_dirs):
    assert input_rgb_image_lists
    assert input_NIRS_image_lists

    # get only the folder name and create the output folder
    # sequence_folder_name = osp.split(folder_name)[1]
    # export_folder = osp.join(cfg.paths.export_dir, sequence_folder_name)
    # if not osp.exists(export_folder):
    #     os.makedirs(export_folder)

    # go through the input image list
    idx = 0
    for image_path_list, image_path_NIRS_list in zip(input_rgb_image_lists, input_NIRS_image_lists):

        image_path_lists = sorted(glob.glob(image_path_list + "/*"))
        image_path_NIRS_lists = sorted(glob.glob(image_path_NIRS_list + "/*"))

        for image_path, image_path_NIRS in zip(image_path_lists, image_path_NIRS_lists):
            # 读取rgb图像
            # read and enhance current image frame
            image = cv2.imread(image_path, 3)
            enhanced_image = enhance_image_2(image)
            # file_path = os.path.splitext(image_path)[0]
            # folder_name = file_path.split('/')[-2]
            # export_dir = osp.join(cfg.paths.export_dir,folder_name)
            # if not osp.exists(export_dir):
            #     os.makedirs(export_dir)
            # 读取近红外图像
            image2 = cv2.imread(image_path_NIRS, 3)
            enhanced_image_NIRS = enhance_image_2(image2)

            keypoints, _ = extract_image_keypoints(enhanced_image, cfg.params.keypoint_extractor)
            nb_keypoints = len(keypoints)
            if nb_keypoints == 0:
                print("nb_keypoints==0")
                continue
            # convert keypoints to numpy fp32
            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)

            # keep sparse keypoints by removing closest points under threshold in a greedy way
            # 通过以贪婪的方式删除阈值以下的最近点来保留稀疏的关键点
            to_be_removed_ids = []
            for i in range(nb_keypoints):
                for j in range(nb_keypoints):
                    if i != j and j not in to_be_removed_ids:
                        # print(source_keypoints_coords[i])
                        dist = distance.euclidean(source_keypoints_coords[i].flatten(), source_keypoints_coords[j].flatten())
                        if dist < cfg.params.keypoint_dist_threshold:
                            to_be_removed_ids.append(j)

            keypoints = list(keypoints)

            for el_idx in sorted(to_be_removed_ids, reverse=True):
                del keypoints[el_idx]

            if len(keypoints) == 0:
                print("keypoints_len==0")
                continue

            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)
            nb_keypoints = len(source_keypoints_coords)
            image_height = enhanced_image.shape[0]
            image_width = enhanced_image.shape[1]
            #print(image_height,image_width)
            (center_x, center_y) = (image_width // 2, image_height // 2)

            # select random transformation between predefined transformation list
            # 在预定义的转换列表中选择随机转换
            transformation = random.choice(cfg.params.transformation_list)

            # TODO just for debug
            # transformation = "rotation" #旋转

            if transformation == "rotation":
                rotation_angle = random.choice(cfg.params.predefined_angle_degrees)
                rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
                warped_image = cv2.warpAffine(enhanced_image_NIRS, rotation_matrix, (image_width, image_height))

                triplet_counter = 0
                for b in range(0, len(keypoints) - 1):
                    rotated_point = rotation_matrix.dot(
                        np.array((int(source_keypoints_coords[b][0, 0]), int(source_keypoints_coords[b][0][1])) + (1,))
                    )
                    #positive--同一位置近红外 作rotation变换
                    xp = int(rotated_point[0])
                    yp = int(rotated_point[1])
                    #anchor--rgb
                    xa = int(source_keypoints_coords[b][0][0])
                    ya = int(source_keypoints_coords[b][0][1])
                    #negative--其他位置的近红外
                    xn = int(source_keypoints_coords[b + 1][0][0])
                    yn = int(source_keypoints_coords[b + 1][0][1])
                    z = cfg.params.patch_size  #64

                    # check if the  patch is inside the image canvas
                    if (
                        ((yp - z) > 0)
                        & ((xp - z) > 0)
                        & ((yp + z) < image_height)
                        & ((xp + z) < image_width)
                        & ((ya - z) > 0) #a
                        & ((xa - z) > 0) #a
                        & ((ya + z) < image_height) #a
                        & ((xa + z) < image_width) #a
                        & ((yn - z) > 0)
                        & ((xn - z) > 0)
                        & ((yn + z) < image_height)
                        & ((xn + z) < image_width)
                    ):
                        # do crop patch from the  warped image
                        crop_img_p = warped_image[yp - z : yp + z, xp - z : xp + z]
                        crop_img_a = enhanced_image[ya - z : ya + z, xa - z : xa + z]
                        crop_img_n = enhanced_image_NIRS[yn - z : yn + z, xn - z : xn + z]

                        # construct output filenames for triplet patches
                        curr_output_folder = osp.join(cfg.paths.export_dir,f"{idx}_{triplet_counter}")
                        filename_p = curr_output_folder + "/p.png"
                        filename_a = curr_output_folder + "/a.png"
                        filename_n = curr_output_folder + "/n.png"

                        # 生成对应位置的近红外图像
                        # crop_img_m = enhanced_image_NIRS[ya - z: ya + z, xa - z: xa + z]
                        # curr_output_match_folder = osp.join(cfg.paths.export_match_dir,f"{idx}_{triplet_counter}")
                        # filename_m = curr_output_match_folder + "/m.png"

                        if not osp.exists(curr_output_folder):
                            os.makedirs(curr_output_folder)
                        # if not osp.exists(curr_output_match_folder):
                        #     os.makedirs(curr_output_match_folder)

                        # save the triplet patches
                        cv2.imwrite(filename_p, crop_img_p)
                        cv2.imwrite(filename_a, crop_img_a)
                        cv2.imwrite(filename_n, crop_img_n)

                        # cv2.imwrite(filename_m, crop_img_m)

                        triplet_counter += 1
                idx += 1
            # -------------------------------scaling-----------------------------------------------------------------------------------
            elif transformation == 'zoom':
                zoom_tab = random.choice(cfg.params.predefined_zoom_factors)
                zf = zoom_tab
                transformed = clipped_zoom(enhanced_image_NIRS, zf)
                triplet_counter = 0
                for b in range(0, len(keypoints) - 1):
                    xa = int(source_keypoints_coords[b][0][0])
                    ya = int(source_keypoints_coords[b][0][1])
                    xp, yp = zoom_coordinates(enhanced_image_NIRS, xa, ya, zf)
                    xn = int(source_keypoints_coords[b + 1][0][0])
                    yn = int(source_keypoints_coords[b + 1][0][1])
                    z = cfg.params.patch_size  #64

                    if  (
                        ((yp - z) > 0)
                        & ((xp - z) > 0)
                        & ((yp + z) < image_height)
                        & ((xp + z) < image_width)
                        & ((ya - z) > 0) #a
                        & ((xa - z) > 0) #a
                        & ((ya + z) < image_height) #a
                        & ((xa + z) < image_width) #a
                        & ((yn - z) > 0)
                        & ((xn - z) > 0)
                        & ((yn + z) < image_height)
                        & ((xn + z) < image_width)
                    ):
                        # do crop patch from the  warped image
                        crop_img_p = transformed[yp - z: yp + z, xp - z: xp + z]
                        crop_img_a = enhanced_image[ya - z: ya + z, xa - z: xa + z]
                        crop_img_n = enhanced_image_NIRS[yn - z: yn + z, xn - z: xn + z]

                        # crop_img_p = transformed[yp - z:yp + z, xp - z: xp + z]
                        # crop_img_a = image_enhanced[ya - z:ya + z, xa - z: xa + z]
                        # crop_img_n = image_enhanced[yn - z:yn + z, xn - z: xn + z]

                        curr_output_folder = osp.join(cfg.paths.export_dir, f"{idx}_{triplet_counter}")
                        filename_p = curr_output_folder + "/p.png"
                        filename_a = curr_output_folder + "/a.png"
                        filename_n = curr_output_folder + "/n.png"
                        if not osp.exists(curr_output_folder):
                            os.makedirs(curr_output_folder)

                        # save the triplet patches
                        cv2.imwrite(filename_p, crop_img_p)
                        cv2.imwrite(filename_a, crop_img_a)
                        cv2.imwrite(filename_n, crop_img_n)

                        # if not os.path.exists("/home/crns/Desktop/code/base_train1/%i_%i" % (idx, i)):
                        #     os.makedirs("/home/crns/Desktop/code/base_train1/%i_%i" % (idx, i))
                        # filename_p = "/home/crns/Desktop/code/base_train1/%i_%i/p.png" % (idx, i)
                        # filename_a = "/home/crns/Desktop/code/base_train1/%i_%i/a.png" % (idx, i)
                        # filename_n = "/home/crns/Desktop/code/base_train1/%i_%i/n.png" % (idx, i)
                        # cv.imwrite(filename_p, crop_img_p)
                        # cv.imwrite(filename_a, crop_img_a)
                        # cv.imwrite(filename_n, crop_img_n)
                        triplet_counter += 1
                idx += 1
            # --------------------------------------------Translation---------------------------------------------------------------------
            elif transformation == 'translation':
                M = np.float32([[1, 0, 8], [0, 1, 8]])
                transformed = cv2.warpAffine(enhanced_image_NIRS, M, (image_width, image_height))
                triplet_counter = 0
                for b in range(0, len(keypoints) - 1):
                    rotated_point = (M.dot(np.array((int(source_keypoints_coords[b][0][0]), int(source_keypoints_coords[b][0][1])) + (1,))))
                    xp = int(rotated_point[0])
                    yp = int(rotated_point[1])
                    xa = int(source_keypoints_coords[b][0][0])
                    ya = int(source_keypoints_coords[b][0][1])
                    xn = int(source_keypoints_coords[b + 1][0][0])
                    yn = int(source_keypoints_coords[b + 1][0][1])
                    z = cfg.params.patch_size  #64

                    if (
                            ((yp - z) > 0)
                            & ((xp - z) > 0)
                            & ((yp + z) < image_width)
                            & ((xp + z) < image_height)
                            & ((ya - z) > 0)
                            & ((xa - z) > 0)
                            & ((ya + z) < image_width)
                            & ((xa + z) < image_height)
                            & ((yn - z) > 0)
                            & ((xn - z) > 0)
                            & ((yn + z) < image_width)
                            & ((xn + z) < image_height)
                    ):

                        crop_img_p = transformed[yp - z: yp + z, xp - z: xp + z]
                        crop_img_a = enhanced_image[ya - z: ya + z, xa - z: xa + z]
                        crop_img_n = enhanced_image_NIRS[yn - z: yn + z, xn - z: xn + z]
                        # crop_img_p = transformed[yp - z:yp + z, xp - z: xp + z]
                        # crop_img_a = enhanced_image[ya - z:ya + z, xa - z: xa + z]
                        # crop_img_n = enhanced_image[yn - z:yn + z, xn - z: xn + z]

                        curr_output_folder = osp.join(cfg.paths.export_dir, f"{idx}_{triplet_counter}")
                        filename_p = curr_output_folder + "/p.png"
                        filename_a = curr_output_folder + "/a.png"
                        filename_n = curr_output_folder + "/n.png"
                        if not osp.exists(curr_output_folder):
                            os.makedirs(curr_output_folder)

                        # save the triplet patches
                        cv2.imwrite(filename_p, crop_img_p)
                        cv2.imwrite(filename_a, crop_img_a)
                        cv2.imwrite(filename_n, crop_img_n)

                        # if not os.path.exists("/home/crns/Desktop/code/base_train1/%i_%i" % (idx, i)):
                        #     os.makedirs("/home/crns/Desktop/code/base_train1/%i_%i" % (idx, i))
                        # filename_p = "/home/crns/Desktop/code/base_train1/%i_%i/p.png" % (idx, i)
                        # filename_a = "/home/crns/Desktop/code/base_train1/%i_%i/a.png" % (idx, i)
                        # filename_n = "/home/crns/Desktop/code/base_train1/%i_%i/n.png" % (idx, i)
                        #
                        # # export the current triplet patches
                        # cv2.imwrite(filename_p, crop_img_p)
                        # cv2.imwrite(filename_a, crop_img_a)
                        # cv2.imwrite(filename_n, crop_img_n)
                        triplet_counter += 1
                idx += 1



if __name__ == "__main__":
    main()
