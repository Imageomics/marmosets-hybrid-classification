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