"""
Dataset classes

This script contains the classes to different datasets used in this project.
"""

import os
import random

from PIL import Image

import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, num_imgs=100):
        super().__init__()
        self.lbls = torch.randint(0, 2, (num_imgs, ))
        self.imgs = []
        for lbl in self.lbls:
            if lbl.item() == 0:
                self.imgs.append(torch.zeros(3, 224, 224))
            else:
                self.imgs.append(torch.ones(3, 224, 224))

    def __getitem__(self, index):
        img = self.imgs[index]
        lbl = self.lbls[index]

        return img, lbl
    
    def __len__(self):
        return len(self.imgs)
    
def get_marmoset_datasets(root, splits, transforms):
    lbl_map = {}
    counts = {}
    lbl_names = set()
    with open(os.path.join(root, "marmoset_hybrids.csv")) as f:
        # Assumes csv column order: Individual,locality,lat,lon,Species,Subspecies,hybrid_stat,View,Sex,Age,Weight(g)
        lines = f.readlines()
        for line in lines[1:]:
            vals = line.split(",")
            id = vals[0]
            species = vals[4]
            hybrid = vals[6]
            lbl_map[id] = {
                "species" : species,
                "hybrid" : hybrid
            }
            lbl_names.add(species)

    lbl_names = sorted(list(lbl_names))
    species_to_lbl_map = dict(zip(lbl_names, range(len(lbl_names))))

    species_paths = {}
    total = 0
    for dir_root, _, files in os.walk(os.path.join(root, "cropped")):
        for fname in files:
            total += 1
            file_id = fname.split("_")[0]
            species = lbl_map[file_id]['species']
            if species not in species_paths:
                species_paths[species] = []
            species_paths[species].append(os.path.join(dir_root, fname))

    data_splits = [{"paths" : [], "lbls" : []} for i in range(3)]
    for species in species_paths:
        amt = len(species_paths[species])
        random.shuffle(species_paths[species])

        train_idx = int(amt * splits[0])
        species_split = species_paths[species][:train_idx]
        data_splits[0]["paths"] += species_split
        data_splits[0]["lbls"] += [species_to_lbl_map[species] for i in range(len(species_split))]
        
        val_idx = int(amt * (splits[0] + splits[1]))
        species_split = species_paths[species][train_idx:val_idx]
        data_splits[1]["paths"] += species_split
        data_splits[1]["lbls"] += [species_to_lbl_map[species] for i in range(len(species_split))]
        
        species_split = species_paths[species][val_idx:]
        data_splits[2]["paths"] += species_split
        data_splits[2]["lbls"] += [species_to_lbl_map[species] for i in range(len(species_split))]

    return (
        MarmosetCroppedDataset(data_splits[0]["paths"], data_splits[0]["lbls"], transforms=transforms[0]),
        MarmosetCroppedDataset(data_splits[1]["paths"], data_splits[1]["lbls"], transforms=transforms[1]),
        MarmosetCroppedDataset(data_splits[2]["paths"], data_splits[2]["lbls"], transforms=transforms[2])
    )


    
class MarmosetCroppedDataset(Dataset):
    def __init__(self, paths, lbls, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.lbls = lbls
        self.paths = paths


    def __getitem__(self, idx):
        lbl = self.lbls[idx]
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        return img, lbl


    def __len__(self):
        return len(self.lbls)