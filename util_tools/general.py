"""
General tools

This script contains functions for general/misc. tools
"""

import torch
import numpy as np

def to_numpy(ten):
    if type(ten) is torch.Tensor:
        return ten.cpu().detach().numpy()
    elif type(ten) is list:
        return np.array(ten)

    return ten