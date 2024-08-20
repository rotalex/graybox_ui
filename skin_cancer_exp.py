import pdb
from typing import List, Set
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import datasets as ds

from graybox.experiment import Experiment
# from deubg_experiment import Experiment
from graybox.dash import Dash
from graybox.monitoring import PairwiseNeuronSimilarity

from graybox.data_samples_with_ops import DataSampleTrackingWrapper
from graybox.model_with_ops import NetworkWithOps
from graybox.model_with_ops import DepType
from graybox.modules_with_ops import Conv2dWithNeuronOps
from graybox.modules_with_ops import LinearWithNeuronOps
from graybox.modules_with_ops import BatchNorm2dWithNeuronOps
from graybox.tracking import TrackingMode
from graybox.tracking import add_tracked_attrs_to_input_tensor

from datasets import load_dataset
import torchvision.transforms as transforms
from torchvision.transforms import PILToTensor



import h5py
import os
import io
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from collections.abc import Iterable
import pandas as pd
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
# import timm

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# device=("cuda" if torch.cuda.is_available() else "cpu")
# torch.device(device)
# print(device)



class SkinCancerModel(NetworkWithOps):
    def __init__(self):
        super(SkinCancerModel, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED
        self.layer0 = Conv2dWithNeuronOps(
            in_channels=3, out_channels=16, kernel_size=3)
        self.dropout0 = nn.Dropout2d(p=0.45)
        self.mpool0 = nn.MaxPool2d(3)
        self.layer1 = Conv2dWithNeuronOps(
            in_channels=16, out_channels=16, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.45)
        self.mpool1 = nn.MaxPool2d(3)
        self.layer2 = Conv2dWithNeuronOps(
            in_channels=16, out_channels=16, kernel_size=3)
        self.dropout2 = nn.Dropout2d(p=0.45)
        self.mpool2 = nn.MaxPool2d(3)
        self.out = LinearWithNeuronOps(in_features=784, out_features=2)

    def children(self):
        return [self.layer0, self.layer1, self.layer2, self.out]

    def define_deps(self):
        self.register_dependencies([
            (self.layer0, self.layer1, DepType.INCOMING),
            (self.layer1, self.layer2, DepType.INCOMING),
            (self.layer2, self.out, DepType.INCOMING),
        ])

    def forward(self, input: th.Tensor):
        self.maybe_update_age(input)
        x = self.layer0(input)
        x = F.relu(x)
        x = self.mpool0(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.mpool1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.mpool2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x, skip_register=True)
        one_hot = F.one_hot(
            output.argmax(dim=1), num_classes=self.out.out_features)
        add_tracked_attrs_to_input_tensor(
            one_hot, in_id_batch=input.in_id_batch,
            label_batch=input.label_batch)
        self.out.register(one_hot)
        return output

class ISICDataset(Dataset):
    def __init__(self, isic_id, images, targets, transforms, device):
        self.transforms=transforms
        self.isic_id=isic_id
        self.targets=targets
        self.device=device
        self.images=images

    @property
    def labels(self):
        return torch.tensor(self.targets)

    def __len__(self):
        return len(self.isic_id)

    def __preprocess(self,image):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        clahe=cv2.createCLAHE(tileGridSize=(8,8),clipLimit=1)
        image[:,:,0]=clahe.apply(image[:,:,0])
        image=cv2.cvtColor(image,cv2.COLOR_LAB2RGB)
        return image

    def __getitem__(self, index):
        key=self.isic_id[index]
        if len(self.images.split('.'))>1:
            image=open_image(self.images,key)
        else:
            image=cv2.imread(f'{self.images}/{key}.jpg')
        image=self.__preprocess(image) #one of them
        #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #one of them
        image=np.moveaxis(image,-1,0)
        image=torch.from_numpy(image)
        if self.transforms:
            image=self.transforms(image)
        try:
            label=torch.tensor(data=self.targets[index])
            return image, label
        except:
            return image


batch_size=256
img_size=224

train_dir='/home/rotaru/Desktop/work/kaggle/isic-2024-challenge/train-metadata.csv'
train_images='/home/rotaru/Desktop/work/kaggle/isic-2024-challenge/train-image/image'

def open_image(images,key):
    f=h5py.File(images,'r')
    image_data=f[key][()]
    nparr = np.frombuffer(image_data, np.uint8)
    img=cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

df=pd.read_csv(train_dir,usecols=['isic_id','target'])
df.head()

from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid=train_test_split(df['isic_id'],df['target'],test_size=0.1,random_state=42)
train=pd.concat([x_train,y_train],axis=1)
valid=pd.concat([x_valid,y_valid],axis=1)
train.head()

def get_transforms(image_size):

    transforms_train = v2.Compose([
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2,contrast=0.2),
        v2.RandomAffine(translate=(0.1,0.1), 
                        scale=(0.1,0.2), 
                        degrees=15, 
                        interpolation=InterpolationMode.BILINEAR),
        v2.Resize((image_size, image_size),
                  interpolation=InterpolationMode.BILINEAR,
                  antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
    ])

    transforms_val = v2.Compose([
        v2.Resize((image_size, image_size),
                  interpolation=InterpolationMode.BILINEAR,
                  antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
    ])

    return transforms_train, transforms_val

transform_train, transform_val = get_transforms(img_size)

def loading_data(isic_id,images,targets,batch_size,shuffle,transforms,device):
    dataset=ISICDataset(isic_id,images,targets,transforms,device)
    # dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    # return dataloader

    return dataset


device = th.device("cuda:0")
dataset_train=loading_data(train['isic_id'].values,
                          train_images,
                          train['target'].values,
                          batch_size,
                          True,
                          transform_train,
                          device)
dataset_eval=loading_data(valid['isic_id'].values,
                        train_images,
                        valid['target'].values,
                        batch_size,
                        False,
                        transform_val,
                        device)


class_counts = torch.bincount(dataset_train.labels)
class_weights = 1. / class_counts.float()
# Assign weights to each sample in the datase
sample_weights = class_weights[dataset_train.labels]
# Create a WeightedRandomSampler to balance the classes
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights), replacement=True)

eval_sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(dataset_eval), replacement=True)

dataset_train = DataSampleTrackingWrapper(dataset_train)
dataset_eval = DataSampleTrackingWrapper(dataset_eval)


def get_train_data_loader(batch_size=32):
    dataloader = DataLoader(
        dataset_train, batch_size=batch_size, sampler=train_sampler)
    return dataloader, dataset_train


def get_eval_data_loader(batch_size=32):
    dataloader = DataLoader(
        dataset_eval, batch_size=batch_size)
    return dataloader, dataset_eval


def get_exp():
    model = SkinCancerModel()
    model.define_deps()
    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        # train_dataset=dataset_train,
        # eval_dataset=dataset_eval,
        train_dataset=None,
        eval_dataset=None,
        device=device, learning_rate=1e-3, batch_size=128,
        name="v0",
        root_log_dir='isic',
        logger=Dash("isic"),
        get_train_data_loader=get_train_data_loader,
        get_eval_data_loader=get_eval_data_loader
    )

    return exp