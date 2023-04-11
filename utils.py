import os
import datetime
from os.path import join


def make_save_dir(save_path: str) -> str:
    """Make a model/data/experiment save directory
    Args:
        save_path (str): The path to the save directory
    """
    save_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = join(save_path, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path