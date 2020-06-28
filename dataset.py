import pandas as pd
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class MLCDataset(Dataset):
    def __init__(self, data_root, cls_dict, enc_dict, train=True):
        self.img_root = os.path.join(data_root, "images")
        if train:
            ann_path = os.path.join(data_root, "train_ann.csv")
        else:
            ann_path = os.path.join(data_root, "validation_ann.csv")
        self.ann_pd = pd.read_csv(ann_path)
        self.cls_dict = cls_dict
        self.enc_dict = enc_dict
        self.img_id = self.ann_pd.image_id.tolist()
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.img_id[idx] + ".jpg")
        img = np.array(Image.open(img_path).resize((256,256)))
        label = self._get_label(idx)
        #label = self._get_label(self.img_id[idx])
        img_tensor = transforms.ToTensor()(img)
        label_tensor = torch.tensor(label)
        
        return img_tensor, label_tensor
    
    def _get_label(self, idx):
        labels = self.ann_pd[self.ann_pd.image_id == self.img_id[idx]].object_class_name.tolist()
        for lid, lab in enumerate(np.unique(labels)):
            if lid == 0:
                label = np.array(self.enc_dict[lab])
            else:
                label = label + np.array(self.enc_dict[lab])
        
        return (label > 0).astype(np.float32)
    
    def __len__(self):
        return len(os.listdir(self.img_root))

class TestDataset(Dataset):
    def __init__(self, data_root):
        self.img_root = data_root
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, "cjss1_(" + str(idx) + ").jpg")
        #os.listdir(self.img_root)
        img = np.array(Image.open(img_path).resize((256,256)))
        img_tensor = transforms.ToTensor()(img)
        
        return img_tensor
    def __len__(self):
        return len(os.listdir(self.img_root))
        
    