import hydra
import os


def get_cwd():
    """
    custom function to get the current hydra output directory while keeping the original working directory
    自定义函数，用于获取当前 Hydra 输出目录，同时保留原始工作目录
    """
    original_cwd = hydra.utils.get_original_cwd()
    cwd_dir = os.getcwd()
    os.chdir(original_cwd)
    return cwd_dir
