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

#_________________________________LOSSES IN THE KD_____________________________________________________
class KD_Loss(nn.Module):
    def __init__(self,alpha,beta):
        super(KD_Loss, self).__init__()
        self.alpha=alpha
        self.beta=beta
        
    def forward(self,soft,hard):
        return self.alpha * soft + self.beta * hard
    
    
class Soft_loss(nn.Module):
    def __init__(self,tau=1):
        super(Soft_loss,self).__init__()
        self.tau=tau
    def forward(self,hrOut,lrOut):
        return (self.tau**2)*torch.mean(hrOut * torch.log(hrOut/ lrOut) + (1-hrOut) * torch.log((1-hrOut)/ (1-lrOut)))