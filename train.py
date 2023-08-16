import os
import types

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as tv_models

from datasets import MarmosetCroppedDataset
from util_tools.general import to_numpy
from util_tools.evaluation import calculate_accuracy, create_accuracy_column_chart, create_confusion_matrix
from util_tools.transforms import train_transforms, test_transforms


def save_model(state, path):
    torch.save(state, path)

def train(train_dl, val_dl, model, opts):
    optimizer = optim.SGD(model.parameters(), lr=opts.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 99999
    for epoch in tqdm(range(opts.epochs), desc="Epochs", position=0, colour='green', leave=True):
        model.train()
        total_loss = 0
        all_preds = []
        all_lbls = []
        for imgs, lbls in tqdm(train_dl, desc="Training", position=1, colour='white', leave=False):
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

        model.eval()
        total_loss = 0
        all_preds = []
        all_lbls = []
        with torch.no_grad():
            for imgs, lbls in tqdm(val_dl, desc="Validating", position=1, colour='blue', leave=False):
                out = model(imgs)

                _, preds = torch.max(out, dim=1)
                all_preds += to_numpy(preds).tolist()
                all_lbls += to_numpy(lbls).tolist()

                loss = loss_fn(out, lbls)
                total_loss += loss.item()

            acc = calculate_accuracy(all_preds, all_lbls)
            print(f"Epoch ({epoch+1}) | Validation Loss: {round(total_loss, 4)} | Validation Accuracy: {round(acc, 4) * 100}%")
            per_class_bar_chart_im = create_accuracy_column_chart(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
            per_class_bar_chart_im.save(os.path.join(opts.exp_dir, "bar_chart_accuracies.png"))
            confusion_matrix_img = create_confusion_matrix(all_preds, all_lbls, lbl_to_name_map=opts.lbl_to_name_map)
            confusion_matrix_img.save(os.path.join(opts.exp_dir, "confusion_matrix.png"))

            if total_loss < best_val:
                print("Saving Best model")
                best_val = total_loss
                save_model(model.state_dict(), os.path.join(opts.exp_dir, "best.pt"))

    save_model(model.state_dict(), os.path.join(opts.exp_dir, "last.pt"))

def load_data(opts):
    tr_dl = DataLoader(MarmosetCroppedDataset(opts.dset_dir, transforms=train_transforms()), batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(MarmosetCroppedDataset(opts.dset_dir, transforms=test_transforms()), batch_size=opts.batch_size, shuffle=False, num_workers=4)

    return tr_dl, val_dl

def get_model(opts):
    model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, opts.num_classes)
    return model

def get_options():
    opts = types.SimpleNamespace()
    opts.epochs = 100
    opts.lr = 0.0001
    opts.exp_dir = "data/train/marmosets/exp_resnet_001"
    opts.dset_dir = "data/marmosets"
    opts.num_classes = 5
    opts.batch_size = 4
    opts.lbl_to_name_map = dict(zip(range(5), ["A", "AH", "J", "P", "PJ"]))
    os.makedirs(opts.exp_dir, exist_ok=True)
    return opts

if __name__ == "__main__":
    opts = get_options()
    train_dataloader, val_dataloader = load_data(opts)
    model = get_model(opts)
    train(train_dataloader, val_dataloader, model, opts)