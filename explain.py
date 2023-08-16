"""
Script for explaining predictions of a model

The purpose of this script is the run gradcam-like functions to 
help explain the predictions of visual models.
"""

from argparse import ArgumentParser

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.transforms import Resize

from util_tools.transforms import test_transforms, resize_and_to_tensor
from util_tools.grad_cam_api import run_grad_cam, show_cam_on_image

def load_model(opts):
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, opts.num_classes)
    model.load_state_dict(torch.load(opts.model))
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--model", type=str, default="data/model_weights/marmoset_classifier.pt")
    parser.add_argument("--img", type=str, default="data/marmosets/cropped/BJT207_face_cp.png")
    parser.add_argument("--save_path", type=str, default="tmp/cam.png")
    parser.add_argument("--cls_tgt", type=int, default=2)
    args = parser.parse_args()
    
    model = load_model(args)

    target_layers = [model.layer4[-1]]
    rgb_img = Image.open(args.img).convert('RGB')

    input_tensor = test_transforms()(rgb_img).unsqueeze(0)

    grayscale_cam = run_grad_cam(model, input_tensor, target_layers, [args.cls_tgt], use_cuda=False)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(np.array(Resize((224, 224))(rgb_img)).astype(np.float32)/255, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization).save(args.save_path)