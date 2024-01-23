import os.path as osp
import torch

from .hardnet_model import HardNet128
from .hynet_model import HyNet
from .sosnet_model import SOSNet
from .swim_model import SwinTransformer
from .ltformer_model import ltformer_d128

ACCEPTED_MODEL_NAMES = ["HyNet", "SOSNet", "HardNet128","SwinTransformer", "LTFormer"]


def model_factory(model_name, model_weights_path):
    assert model_name in ACCEPTED_MODEL_NAMES
    assert osp.exists(model_weights_path)

    state_dict = torch.load(model_weights_path)
    if model_name == "HyNet":
        model = HyNet()
        model.load_state_dict(state_dict)
    elif model_name == "SOSNet":
        model = SOSNet()
        model.load_state_dict(state_dict)
    elif model_name == "HardNet128":
        model = HardNet128()
        model.load_state_dict(state_dict["state_dict"])
    elif model_name == "SwinTransformer":
        model = SwinTransformer()
        model.load_state_dict(state_dict["state_dict"])
    elif model_name == 'LTFormer':
        model = ltformer_d128()
        model.load_state_dict(state_dict["state_dict"])
    else:
        raise
    return model
