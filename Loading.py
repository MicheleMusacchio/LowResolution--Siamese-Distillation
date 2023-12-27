import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.utils
import torchvision.transforms as transforms
import torch
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import resnet50
from tqdm import tqdm
from time import time
from torch.autograd import profiler
import warnings
warnings.filterwarnings('ignore')


class MyDataset(Dataset):
    def __init__(self, high_res_folder, low_res_folder, stud ,num_pairs, Rand=True, seed=None ):
        self.high_res_folder = high_res_folder
        self.low_res_folder = low_res_folder
        self.num_pairs = int(num_pairs)
        self.Rand = Rand
        self.stud = stud

        if seed is not None:
            random.seed(seed)

        self.subfolders = [f.name for f in os.scandir(high_res_folder) if f.is_dir()]

        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        if self.Rand:
            k = random.randint(0, len(self.subfolders) - 1)
            j = random.randint(0, len(self.subfolders) - 1)
        else:
            k = random.randint(0, len(self.subfolders) - 1)
            f_path = os.path.join(self.high_res_folder, self.subfolders[k])
            while len([f for f in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, f))]) == 1 :
                k = random.randint(0, len(self.subfolders) - 1)
                f_path = os.path.join(self.high_res_folder, self.subfolders[k])
            j = k

        high_res_x_path = os.path.join(self.high_res_folder, self.subfolders[k])
        low_res_x_path = os.path.join(self.low_res_folder, self.subfolders[k])

        high_res_x_image = self.load_random_image(high_res_x_path)
        low_res_x_image = self.load_random_image(low_res_x_path)

        high_res_y_path = os.path.join(self.high_res_folder, self.subfolders[j])
        low_res_y_path = os.path.join(self.low_res_folder, self.subfolders[j])

        high_res_y_image = self.load_random_image(high_res_y_path)
        low_res_y_image = self.load_random_image(low_res_y_path)

        label = 1 if k == j else 0
        
        assert self.stud in ['HR','LR','ALL']
        
        if self.stud=='LR':
            return low_res_x_image, low_res_y_image, label
        elif self.stud=='HR':
            return high_res_x_image, high_res_y_image, label
        elif self.stud=='ALL':
            return high_res_x_image, low_res_x_image, high_res_y_image, low_res_y_image, label

    def load_random_image(self, folder_path):
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        image_path = os.path.join(folder_path, random.choice(image_files))

        image = Image.open(image_path).convert('RGB')
        return self.transform(image)
    
    
#___________________________________________________________________________________________________________________



def Create_dataloader(Res_quality='ALL',Batch_size=32,
                      number_of_training=10**5,number_of_validation=2*10**4,number_of_test=10**4,
                      PIN=True,Phase='Train',Seed=1,MAX=os.cpu_count(),
                      HRpath = 'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/224x224_lfw',
                      LRpath =  'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/LR_64'):

    #DATALOADER PER LR
    assert Phase in ['Train','Validation','Train_Validation','Test','ALL']
    dataset_train_same=MyDataset(high_res_folder=HRpath , low_res_folder=LRpath, stud= Res_quality, num_pairs= number_of_training/2, Rand=True, seed=Seed)
    dataset_train_diff =MyDataset(high_res_folder=HRpath , low_res_folder=LRpath, stud= Res_quality, num_pairs= number_of_training/2, Rand=False, seed=5*Seed)
    dataloader_train = DataLoader(ConcatDataset([dataset_train_same,dataset_train_diff]), batch_size=Batch_size, shuffle=True,pin_memory=PIN,num_workers=MAX)
    if Phase=='Train':
        return dataloader_train
    #_______________________________________________________________________________________________________________________________________
    dataset_validation_same = MyDataset(high_res_folder=HRpath , low_res_folder=LRpath, stud= Res_quality, num_pairs= number_of_validation/2, Rand=True, seed=10*Seed)
    dataset_validation_diff = MyDataset(high_res_folder=HRpath , low_res_folder=LRpath, stud= Res_quality, num_pairs= number_of_validation/2, Rand=False, seed=11*Seed)
    dataloader_validation = DataLoader(ConcatDataset([dataset_validation_same,dataset_validation_diff]), batch_size=Batch_size, shuffle=True,pin_memory=PIN,num_workers=MAX)
    
    if Phase=='Validation':
        return dataloader_validation
    elif Phase=='Train_Validation':
        return dataloader_train,dataloader_validation
    #________________________________________________________________________________________________________________________________________
    dataset_test_same = MyDataset(high_res_folder=HRpath , low_res_folder=LRpath, stud= Res_quality, num_pairs= number_of_test/2, Rand=True, seed=13*Seed)
    dataset_test_diff = MyDataset(high_res_folder=HRpath , low_res_folder=LRpath, stud= Res_quality, num_pairs= number_of_test/2, Rand=False, seed=24*Seed)
    dataloader_test = DataLoader(ConcatDataset([dataset_test_same,dataset_test_diff]), batch_size=Batch_size, shuffle=True, pin_memory=PIN,num_workers=MAX)
    
    if Phase=='Test':
        return dataloader_test
    elif Phase=='ALL':
        return dataloader_train, dataloader_validation, dataloader_test
