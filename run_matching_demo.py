import os
import logging

from _version import __version__
import hydra
import json
from omegaconf import OmegaConf
from tqdm import tqdm

import imageio

import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils.path import get_cwd
import numpy as np
import cv2
from math import sqrt


# 设置GPU卡号
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
writer = SummaryWriter()
logger = logging.getLogger("Demo_matching")
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import statistics

from src.utils.matcher import feature_match, feature_extraction, evaluate_matches
from src.datasets.test_patch_loader import PatchDataset
from src.models.arch_factory import model_factory


def export_gif(frame_path_list, out_gif_filename, fps=24):
    frame_list = []
    for frame_path in tqdm(frame_path_list):
        frame_list.append(cv2.imread(frame_path))
        # os.remove(frame_path)
    imageio.mimsave(out_gif_filename, frame_list, fps=fps)


@hydra.main(version_base=None, config_path="config", config_name="config_matching_ltformer")
def main(cfg):
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    output_dir = get_cwd()
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Export dir: {output_dir}")

    logger.info("Loading parameters from config file")
    validation_data_root = cfg.paths.demo_sequence_data
    model_name = cfg.params.model_name
    model_weights_path = cfg.params.weights_path
    image_size = cfg.params.image_size
    patch_size = cfg.params.patch_size
    distance_matching_threshold = cfg.params.distance_matching_threshold
    matching_threshold = cfg.params.matching_threshold

    # load the model to be evaluated
    model = model_factory(model_name, model_weights_path)

    # generate patches from video frames
    # generate testing data
    test_dataset = PatchDataset(validation_data_root, patch_size)

    # list to store output git frames
    frame_path_list = []

    # back metrics
    precision = []
    matching_score = []

    # go through the patches, frame by frame
    # for i, data in enumerate(test_dataset):
    model.eval()
    model.cuda()

    # go through the patches, frame by frame
    for id, data in enumerate(tqdm(test_dataset)):

        patch_src = data["patch_src"]
        patch_dst = data["patch_dst"]
        keypoint_src = data["keypoint_src"]
        keypoint_dst = data["keypoint_dst"]
        # frame_src = data["image_src_name"]
        # Next_frame = data["image_dst_name"]
        frame_src = data["ori_src"]
        Next_frame = data["ori_dst"]
        folder_name = data["image_name"]

        if not patch_src:
            continue

        # extract feature vector for all patches
        list_desc_src = feature_extraction(patch_src, model, image_size)
        list_desc_dst = feature_extraction(patch_dst, model, image_size)

        # do matching
        matches, distance_list = feature_match(
            list_desc_src, list_desc_dst, matching_threshold
        )
        # compute evaluation metrics 计算评估指标
        nb_false_matching, nb_true_matches, nb_rejected_matches = evaluate_matches(
            keypoint_src,
            keypoint_dst,
            matches,
            distance_matching_threshold,
            distance_list,
            matching_threshold,
        )
        nb_received_matching = nb_false_matching + nb_true_matches
        if nb_received_matching <= 0:
            nb_received_matching = 0.1
        precision.append(nb_true_matches / (nb_received_matching))
        matching_score.append(
            nb_true_matches
            / (nb_received_matching + nb_rejected_matches)
        )

        h, w ,_= frame_src.shape

        # -------------------------------------------------------------------------------------------------
        image_match = np.concatenate((frame_src, Next_frame), axis=1)
        image_match_rgb = cv2.cvtColor(image_match, 3)

        for i in range(0, len(keypoint_src)):

            # if matches[i] != -1:
            xa = int(keypoint_src[i][0])
            ya = int(keypoint_src[i][1])
            xp = int(keypoint_dst[i][0])
            yp = int(keypoint_dst[i][1])
            x = int(keypoint_dst[matches[i]][0])
            y = int(keypoint_dst[matches[i]][1])
            dist = sqrt((yp - y) ** 2 + (xp - x) ** 2)

            cv2.circle(
                image_match_rgb, (xa, ya), radius=2, color=(255, 0, 0), thickness=2
            )
            cv2.circle(
                image_match_rgb, (xp + w, yp), radius=2, color=(255, 0, 0), thickness=2
            )
            if dist > distance_matching_threshold:
                cv2.line(image_match_rgb, (xa, ya), (x + w, y), (0, 0, 255), thickness=1)
            else:
                cv2.line(image_match_rgb, (xa, ya), (x + w, y), (0, 255, 0), thickness=1)


        if not os.path.exists(os.path.join(output_dir, folder_name)):
            os.makedirs(os.path.join(output_dir, folder_name))
        file_name = os.path.join(output_dir, folder_name, f"matches{id}_{id + 1}.png")
        frame_path_list.append(file_name)
        cv2.imwrite(file_name, cv2.resize(image_match_rgb, (0, 0), fx=0.6, fy=0.6))

        # break

    logger.info(f"Precision= {statistics.mean(precision):0.4f}")
    logger.info(f"Matching_score= {statistics.mean(matching_score):0.4f}")

    # logger.info("Start exporting demo GIF")
    # export_git_filename = os.path.join(output_dir, "matching_demo_1.gif")
    # export_gif(
    #     frame_path_list=frame_path_list, out_gif_filename=export_git_filename, fps=20
    # )


if __name__ == "__main__":
    main()
