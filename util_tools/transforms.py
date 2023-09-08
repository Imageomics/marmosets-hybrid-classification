import torchvision.transforms as T

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], #TODO: Update this for the marmosets dataset
                    std=[0.229, 0.224, 0.225])

def train_transforms():
    return T.Compose([
        T.RandomCrop((256, 256)),
        T.Resize((224, 224)),
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(90),
        T.RandomAffine(15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        T.ToTensor(),
        NORMALIZE
    ])

def resize_and_to_tensor():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

def test_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        NORMALIZE
    ])