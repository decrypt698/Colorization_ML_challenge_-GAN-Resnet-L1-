import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
import os
from PIL import Image


class ColorizationDataset(Dataset):          # 👈 UN seul nom (celui qu'importe train.py)
    def __init__(self, root_dir, size=(128, 128), train=False):
        self.root_dir = root_dir
        self.size = size
        self.train = train
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]     # 👈 déjà le chemin complet
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self))
        img = np.array(img)
        img = resize(img, self.size)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1, :].copy()
        lab_img = rgb2lab(img)
        l_channel = lab_img[:, :, 0]
        ab_channels = lab_img[:, :, 1:3]
        l_channel = (l_channel / 50.0) - 1.0
        ab_channels = ab_channels / 128.0
        l_tensor = torch.from_numpy(l_channel).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_channels).permute(2, 0, 1).float()
        l_tensor = l_tensor.repeat(3, 1, 1)   # à décommenter SEULEMENT si tu utilises ResNet
        return {'L': l_tensor, 'ab': ab_tensor}


# ---- Fonctions HORS de la classe (collées à gauche, au même niveau que 'class') ----
def get_dataloader(root_dir, batch_size=16, shuffle=True, num_workers=4):
    dataset = ColorizationDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def lab_to_rgb_image(l_channel, ab_channels):
    l_channel=l_channel[0] #prend un seul canal (les 3 sont identique )
    l = (l_channel.squeeze().cpu().numpy() + 1.0) * 50.0     # 👈 .cpu() AVEC parenthèses
    ab = ab_channels.permute(1, 2, 0).cpu().numpy() * 128.0
    lab_img = np.zeros((l.shape[0], l.shape[1], 3))
    lab_img[:, :, 0] = l
    lab_img[:, :, 1:] = ab
    rgb_img = lab2rgb(lab_img)
    return (rgb_img * 255).astype(np.uint8)


def split_paths(files, seed=42, val=0.1, test=0.1):
    random.Random(seed).shuffle(files)
    n = len(files); nt = int(n * test); nv = int(n * val)
    return files[nt+nv:], files[nt:nt+nv], files[:nt]   # train, val, test