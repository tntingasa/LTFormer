import argparse
import os
import pickle
from logging import getLogger
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
from munkres import Munkres
import warnings
import json


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """Re-start from checkpoint."""
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info(f'Found checkpoint at {ckp_path}')

    # open checkpoint file
    checkpoint = torch.load(ckp_path)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(
                key, ckp_path))
        else:
            logger.warning("=> failed to load {} from checkpoint '{}'".format(
                key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]