import os
import types

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as tv_models

from datasets import MarmosetCroppedDataset, get_marmoset_datasets
from util_tools.general import to_numpy
from util_tools.evaluation import calculate_accuracy, create_accuracy_column_chart, create_confusion_matrix
from util_tools.transforms import train_transforms, test_transforms


def test(test_dl, model, opts):
    if opts.use_gpu:
        model.cuda()

    model.eval()

    all_preds = []
    all_lbls = []
    for imgs, lbls in tqdm(test_dl, desc="Testing", position=1, colour='white', leave=False):
        if opts.use_gpu:
            imgs = imgs.cuda()
            lbls = lbls.cuda()
        
        out = model(imgs)

        _, preds = torch.max(out, dim=1)
        all_preds += to_numpy(preds).tolist()
        all_lbls += to_numpy(lbls).tolist()

        acc = calculate_accuracy(all_preds, all_lbls)
        print(f"Testing Accuracy: {round(acc, 4) * 100}%")
        per_class_bar_chart_im = create_accuracy_column_chart(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
        per_class_bar_chart_im.save(os.path.join(opts.exp_dir, "bar_chart_accuracies_test.png"))
        confusion_matrix_img = create_confusion_matrix(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
        confusion_matrix_img.save(os.path.join(opts.exp_dir, "confusion_matrix_test.png"))

def load_data(opts):
    tr_dset, val_dset, test_dset = get_marmoset_datasets(opts.dset_dir, splits=[0.8, 0.1, 0.1], transforms=[test_transforms(), test_transforms(), test_transforms()])

    tr_dl = DataLoader(tr_dset, batch_size=opts.batch_size, shuffle=False, num_workers=4)
    val_dl = DataLoader(val_dset, batch_size=opts.batch_size, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_dset, batch_size=opts.batch_size, shuffle=False, num_workers=4)

    return tr_dl, val_dl, test_dl

def get_model(opts):
    model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, opts.num_classes)
    model_path = os.path.join(opts.exp_dir, "best.pt")
    model.load_state_dict(torch.load(model_path))
    return model

def get_options():
    opts = types.SimpleNamespace()
    opts.exp_dir = "data/train/marmosets/exp_resnet34_8_30_2023"
    opts.dset_dir = "/local/scratch/carlyn.1/marmosets"
    opts.num_classes = 5
    opts.batch_size = 128
    opts.lbl_to_name_map = dict(zip(range(5), ["A", "AH", "J", "P", "PJ"]))
    opts.use_gpu = True
    os.makedirs(opts.exp_dir, exist_ok=True)
    return opts

if __name__ == "__main__":
    opts = get_options()
    _, _, test_dataloader = load_data(opts)
    model = get_model(opts)
    test(test_dataloader, model, opts)