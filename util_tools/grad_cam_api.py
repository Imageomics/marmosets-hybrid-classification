"""
API to pytorch_grad_cam external module

API to access functionality in the pytorch_grad_cam module to handle
import errors.
"""

import sys
sys.path.insert(0, "externals/pytorch_grad_cam")

from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image as show_cam_on_image_ex

def run_grad_cam(model, input_tensor, target_layers, target_ids, use_cuda):

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    targets = [ClassifierOutputTarget(tgt_id) for tgt_id in target_ids]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam

def show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True):
    return show_cam_on_image_ex(rgb_img, grayscale_cam, use_rgb=use_rgb)