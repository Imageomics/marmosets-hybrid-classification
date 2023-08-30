import torch
import torch.nn as nn

import torchvision.models as tv_models

from util_tools.transforms import test_transforms

def infer(image, model_path):
    model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    input_tensor = test_transforms()(image).unsqueeze(0)

    out = model(input_tensor)
    m = nn.Softmax(dim=1)
    confidences = m(out)[0]
    _, predictions = torch.max(out, dim=1)

    prediction = predictions[0].item()
    confidences = confidences.detach().cpu().numpy().tolist()

    return {
        "prediction" : prediction,
        "confidences" : confidences
    }

    