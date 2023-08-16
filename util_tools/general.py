"""
General tools

This script contains functions for general/misc. tools
"""
import io

import torch
import numpy as np

from PIL import Image

def to_numpy(ten):
    if type(ten) is torch.Tensor:
        return ten.cpu().detach().numpy()
    elif type(ten) is list:
        return np.array(ten)

    return ten


def get_PIL_image_from_matplotlib_figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)