# datos.py
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATA = Path("dogs-vs-cats")      # o Path("cache_144") si usas la caché
IMG_SIZE = 128
MEDIA = [0.485, 0.456, 0.406] # Media y std de ImagesNet
STD = [0.229, 0.224, 0.225]

transform_eval = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, STD),
])
transform_train_aug = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33)), # Elige un recorte de la imagen 
                                                                                  # original con un área del 50 % al 100 % del 
                                                                                  # total y una proporción entre 3:4 y 4:3, en 
                                                                                  # una posición al azar y Lo reescala a 128×128.
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # varía un ±20 % el brillo, el contraste y la saturación
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, STD),
])    # en el paso 5 añadiremos augmentation


class ImagenesDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = list(paths)
        self.labels = None if labels is None else torch.tensor(list(labels), dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is None:
            return img
        return img, self.labels[i]


def crear_loaders(num_workers=8, batch_size=64, augmentation=True):
    # Todo lo pesado va DENTRO de la función: los workers importan este módulo
    # y no queremos que cada uno vuelva a listar 37.500 ficheros.
    train_files = sorted((DATA / "train").glob("*.jpg"))
    df = pd.DataFrame({"path": train_files})
    df["label"] = df["path"].apply(lambda p: 0 if p.name.startswith("cat") else 1)
    df_tr, df_va = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    test_files = sorted((DATA / "test1").glob("*.jpg"), key=lambda p: int(p.stem))

    transform_train = transform_train_aug if augmentation else transform_eval
    ds_tr = ImagenesDataset(df_tr["path"], df_tr["label"], transform_train)
    ds_va = ImagenesDataset(df_va["path"], df_va["label"], transform_eval)
    ds_te = ImagenesDataset(test_files, transform=transform_eval)

    kw_train = dict(num_workers=num_workers,
              pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0)

    nw_eval = min(4, num_workers) # Menos workers poruqe hay que procesar menos datos y conviene siempre reducir procesos vivos
    kw_eval = dict(num_workers=nw_eval,
              pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0)


    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, **kw_train)
    val_loader = DataLoader(ds_va, batch_size=128, **kw_eval)
    test_loader = DataLoader(ds_te, batch_size=128, **kw_eval)
    return train_loader, val_loader, test_loader, test_files