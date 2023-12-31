import os
import types
import random

from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as tv_models

from datasets import MarmosetCroppedDataset, get_marmoset_datasets
from util_tools.general import to_numpy
from util_tools.evaluation import calculate_accuracy, create_accuracy_column_chart, create_confusion_matrix
from util_tools.transforms import train_transforms, test_transforms


def save_model(state, path):
    torch.save(state, path)

def train(train_dl, val_dl, test_dl, model, opts):
    if opts.use_gpu:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=opts.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 99999
    for epoch in tqdm(range(opts.epochs), desc="Epochs", position=0, colour='green', leave=True):
        model.train()
        total_loss = 0
        all_preds = []
        all_lbls = []
        for imgs, lbls in tqdm(train_dl, desc="Training", position=1, colour='white', leave=False):
            if opts.use_gpu:
                imgs = imgs.cuda()
                lbls = lbls.cuda()
            optimizer.zero_grad()
            
            out = model(imgs)

            _, preds = torch.max(out, dim=1)
            all_preds += to_numpy(preds).tolist()
            all_lbls += to_numpy(lbls).tolist()
            
            loss = loss_fn(out, lbls)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = calculate_accuracy(all_preds, all_lbls)
        print(f"Epoch ({epoch+1}) | Training Loss: {round(total_loss, 4)} | Training Accuracy: {round(acc, 4) * 100}%")
        per_class_bar_chart_im = create_accuracy_column_chart(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
        per_class_bar_chart_im.save(os.path.join(opts.exp_dir, "bar_chart_accuracies_train.png"))
        confusion_matrix_img = create_confusion_matrix(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
        confusion_matrix_img.save(os.path.join(opts.exp_dir, "confusion_matrix_train.png"))


        model.eval()
        total_loss = 0
        all_preds = []
        all_lbls = []
        with torch.no_grad():
            for imgs, lbls in tqdm(val_dl, desc="Validating", position=1, colour='blue', leave=False):
                if opts.use_gpu:
                    imgs = imgs.cuda()
                    lbls = lbls.cuda()
                out = model(imgs)

                _, preds = torch.max(out, dim=1)
                all_preds += to_numpy(preds).tolist()
                all_lbls += to_numpy(lbls).tolist()

                loss = loss_fn(out, lbls)
                total_loss += loss.item()

            acc = calculate_accuracy(all_preds, all_lbls)
            print(f"Epoch ({epoch+1}) | Validation Loss: {round(total_loss, 4)} | Validation Accuracy: {round(acc, 4) * 100}%")
            per_class_bar_chart_im = create_accuracy_column_chart(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
            per_class_bar_chart_im.save(os.path.join(opts.exp_dir, "bar_chart_accuracies_val.png"))
            confusion_matrix_img = create_confusion_matrix(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
            confusion_matrix_img.save(os.path.join(opts.exp_dir, "confusion_matrix_val.png"))

            if acc > best_val:
                print("Saving Best model")
                best_val = acc
                save_model(model.state_dict(), os.path.join(opts.exp_dir, "best.pt"))

    save_model(model.state_dict(), os.path.join(opts.exp_dir, "last.pt"))

    # Testing
    model.load_state_dict(torch.load(os.path.join(opts.exp_dir, "best.pt")))
    model.eval()
    total_loss = 0
    all_preds = []
    all_lbls = []
    with torch.no_grad():
        for imgs, lbls in tqdm(val_dl, desc="Testing", position=1, colour='blue', leave=False):
            if opts.use_gpu:
                imgs = imgs.cuda()
                lbls = lbls.cuda()
            out = model(imgs)

            _, preds = torch.max(out, dim=1)
            all_preds += to_numpy(preds).tolist()
            all_lbls += to_numpy(lbls).tolist()

        acc = calculate_accuracy(all_preds, all_lbls)
        print(f"Epoch ({epoch+1}) | Testing Accuracy: {round(acc, 4) * 100}%")
        per_class_bar_chart_im = create_accuracy_column_chart(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
        per_class_bar_chart_im.save(os.path.join(opts.exp_dir, "bar_chart_accuracies_test.png"))
        confusion_matrix_img = create_confusion_matrix(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
        confusion_matrix_img.save(os.path.join(opts.exp_dir, "confusion_matrix_test.png"))

def load_data(opts):
    tr_dset, val_dset, test_dset = get_marmoset_datasets(opts.dset_dir, splits=[0.8, 0.1, 0.1], transforms=[train_transforms(), test_transforms(), test_transforms()])
    print("Train Dataset size", len(tr_dset))
    print("Validation Dataset size", len(val_dset))
    print("Test Dataset size", len(test_dset))
    tr_dl = DataLoader(tr_dset, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_dset, batch_size=opts.batch_size, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_dset, batch_size=opts.batch_size, shuffle=False, num_workers=4)

    return tr_dl, val_dl, test_dl

def get_model(opts):
    model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, opts.num_classes)
    return model

def get_options():
    opts = types.SimpleNamespace()
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--exp_dir", type=str, default="data/train/marmosets/exp_resnet34_9_15_2023")
    parser.add_argument("--dset_dir", type=str, default="/local/scratch/carlyn.1/marmosets")
    parser.add_argument("--use_gpu", default=True)
    opts = parser.parse_args()
    
    opts.lbl_to_name_map = dict(zip(range(5), ["A", "AH", "J", "P", "PJ"]))
    os.makedirs(opts.exp_dir, exist_ok=True)
    return opts

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    opts = get_options()
    set_seed(opts.seed)
    train_dataloader, val_dataloader, test_dataloader = load_data(opts)
    model = get_model(opts)
    train(train_dataloader, val_dataloader, test_dataloader, model, opts)