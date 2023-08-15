import os
import types

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from datasets import DummyDataset

def save_model(state, path):
    torch.save(state, path)

def train(train_dl, val_dl, model, opts):
    optimizer = optim.SGD(model.parameters(), lr=opts.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(opts.epochs), desc="Epochs", position=0, colour='green', leave=True):
        model.train()
        total_loss = 0
        best_val = 99999
        for imgs, lbls in tqdm(train_dl, desc="Training", position=1, colour='white', leave=False):
            optimizer.zero_grad()
            
            out = model(imgs)
            loss = loss_fn(out, lbls)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        
        print(f"Epoch ({epoch+1}) | Training Loss: {round(total_loss, 4)} ")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for imgs, lbls in tqdm(val_dl, desc="Validating", position=1, colour='blue', leave=False):
                out = model(imgs)
                loss = loss_fn(out, lbls)
                total_loss += loss.item()

            print(f"Epoch ({epoch+1}) | Validation Loss: {round(total_loss, 4)} ")
            if total_loss < best_val:
                print("Saving Best model")
                best_val = total_loss
                save_model(model.state_dict(), os.path.join(opts.exp_dir, "best.pt"))

    save_model(model.state_dict(), os.path.join(opts.exp_dir, "last.pt"))


def load_data():
    tr_dl = DataLoader(DummyDataset(80), batch_size=4, shuffle=True, num_workers=4)
    val_dl = DataLoader(DummyDataset(20), batch_size=4, shuffle=False, num_workers=4)

    return tr_dl, val_dl

def get_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 2)
    return model


def get_options():
    opts = types.SimpleNamespace()
    opts.epochs = 2
    opts.lr = 0.0001
    opts.exp_dir = "data/train/exp01"
    os.makedirs(opts.exp_dir, exist_ok=True)
    return opts

if __name__ == "__main__":
    options = get_options()
    train_dataloader, val_dataloader = load_data()
    model = get_model()
    train(train_dataloader, val_dataloader, model, options)