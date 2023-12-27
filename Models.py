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

#__________________________TEACHER MODEL___________________________________________________
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.resnet_model = resnet50(pretrained=True)
        self.resnet_model.fc = nn.Identity()

        for param in self.resnet_model.parameters():
            param.requires_grad = False
            
        self.fcOut = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048,1024), 
            nn.ReLU(),
            nn.Linear(1024, 128)
            ) 
        self.fcOut1=nn.Linear(1,1)
        
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            init.constant_(m.bias, 0)
            
    def forward(self, x1, x2):
        x1 = self.resnet_model(x1)
        x1 = self.fcOut(x1)
        x2 = self.resnet_model(x2)
        x2 = self.fcOut(x2)

        distance=F.pairwise_distance(x1, x2,keepdim=True)
        logits = self.fcOut1(distance)
        output = self.sigmoid(logits)

        return output
#______________________STUDENT MODEL___________________________________________________________
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=8 ),  
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=4),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.sigmoid = nn.Sigmoid()
        
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128,64))  

        self.fcOut1 = nn.Linear(1,1)
        
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            init.constant_(m.bias, 0)


    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = x1.view(x1.size()[0], -1)
        x1 = self.fc1(x1)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = x2.view(x2.size()[0], -1)
        x2 = self.fc1(x2)

        distance=F.pairwise_distance( x1, x2, keepdim = True)
        
        logits = self.fcOut1(distance)
        output = self.sigmoid(logits)

        return output
