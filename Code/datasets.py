import os
import torch
import random
from PIL import Image
from glob import glob
from torch.utils.data import Dataset

class SEMDataset(Dataset):
    def __init__(self, root, transform_img, seed = 3543032):
        self.img_root = os.path.join('./', root)
        self.mode = os.path.basename(self.img_root)
        self.labels = sorted([f for f in os.listdir(
            self.img_root) if os.path.isdir(os.path.join(self.img_root, f))])
        self.img_names = glob(os.path.join(self.img_root, '*/*.jpg'))

        self.transform_img = transform_img
        self.seed = seed
        self.class2idx = {}
        self.idx2class = {}
        
        for i in range(len(self.labels)):
            self.class2idx[self.labels[i]] = i
            self.idx2class[i] = self.labels[i]

    def __getitem__(self, index):
        img_name = self.img_names[index]

        # Seed the random generator
        random.seed(self.seed)
        img = self.transform_img(Image.open(img_name).convert('RGB'))

        label = os.path.basename(os.path.dirname(img_name))
        return img, torch.tensor(self.class2idx[label]), os.path.basename(img_name)

    def __len__(self):
        return len(self.img_names)
