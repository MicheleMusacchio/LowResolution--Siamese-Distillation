# Import libraries
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
from Loading import MyDataset, Create_dataloader
from Models import TeacherModel,StudentModel_2
from Losses import KD_Loss,Soft_loss
import warnings
warnings.filterwarnings('ignore')



def threshold(vald_loader,threshold_list,log_step = 1000):
    val_acc = []
    val_acc_best = (0,0)
    
    for t in tqdm(threshold_list):
        model.eval()
        with torch.no_grad():
            running_acc=0
            n=0
            for cnt, (img1, img2, label) in enumerate(vald_loader):
                batch_dim=label.size(0)
                n+=batch_dim
                img1 = img1.to(device)                
                img2 = img2.to(device)
                label=label.view(-1,1).float().to(device)
                
                distance = model(img1, img2)     
                predictions=(distance>=t).float().to(device)   
                
                acc=torch.mean((predictions==label).float()).to(device)                   
                running_acc+=acc*batch_dim 
                
                if cnt % log_step == 0:
                    #print('\nValidation: Mean of class 1 prediction',torch.mean(predictions.float()).item())
                    print('\n[Threshold: %.2f, Iteration: %5d]  validation Accuracy: %.3f' %(t, cnt + 1, acc.item()))
        print('______________________________________')              
        print('\n[Threshold: %.2f,  ACCURACY: %.3f' %(t,running_acc.detach().cpu()/n))
        print('______________________________________') 
        val_acc.append(running_acc.detach().cpu()/n)
        if running_acc/n > val_acc_best[0]:
            val_acc_best = (running_acc/n,t)
    return val_acc,val_acc_best
#____________________________________________________________________________________________________________________________

def main():
    val_list, val_best=threshold(dataloader_validation,THRESHOLD,LOG_STEP)
    if len(THRESHOLD)>1:
        print('\nALL THE ACCURACIES:', val_list)
        print('\nBEST ONE:', val_best)

#____________________________________________________________________________________________________________________________

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0=time()
    #with profiler.profile(use_cuda=True) as prof:
    #________________INITIALIZATIONS_______________________________________________________________________________
    BATCH_SIZE=128
    N_TEST=10**4
    PIN_MEMORY=True
    PHASE='Test'
    dataloader_validation = Create_dataloader(Res_quality='LR',Batch_size=BATCH_SIZE,number_of_test=N_TEST,Phase=PHASE) 
    model=StudentModel_2().to(device)
    #LOADING THE ALREADY TRAINED EPOCH
    epoch_path='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/OFFICIAL_CHECKPOINT/DISTILLED/DistilledKnowledge_9.pt'
    model.load_state_dict(torch.load(epoch_path)) 

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")
    #'model_name' USED TO SAVE THE CHECKPOINT, HAS TO BE INSERT BY HAND
    THRESHOLD=[0.5] #,0.6]
    LOG_STEP=int(np.ceil( (N_TEST/BATCH_SIZE) /3 ))
    #________________TRAINING___________________________________________________________________________________________
    main()
    #print(prof.key_averages().table(sort_by="cuda_time_total"))
    print('Total time', time()-t0)
    