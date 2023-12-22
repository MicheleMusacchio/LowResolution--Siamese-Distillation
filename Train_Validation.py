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



def train_val(dataloader=None,n_epochs=10,threshold=1,log_step = 100,start_epoch=0,path_to_save_model=None,save_and_plot = True, run_validation=False,vald_loader=None, clip_grad=None,use_scheduler=False):
    train_loss = []
    val_loss = []
    val_loss_best = 0 
    
    for epoch in tqdm(range(start_epoch,n_epochs-1)):
        running_loss=0
        running_accuracy=0
        n=0
        model.train()
        for cnt, (img1, img2, label) in enumerate(dataloader):

            batch_dim=label.size(0)
            n+=batch_dim
            
            img1 = img1.to(device)
            img2 = img2.to(device)
            label=label.view(-1,1).float().to(device)
            
            optimizer.zero_grad()

            distance = model(img1, img2)
            
            #loss = criterion(distance, label)
            loss=BCELOSS(distance, label)
            running_loss += loss*batch_dim       
            
            predictions=(distance>=threshold).to(device) 
            training_accuracy=torch.mean((predictions==label).float()).to(device) #.item() 
            running_accuracy+=training_accuracy*batch_dim 
            
            if cnt % log_step == 0:
                #print('\nTraining: Mean of class 1 prediction',torch.mean(predictions.float()).item())
                print('\n[Epoch: %d, Iteration: %5d]  training loss: %.3f training accuracy: %.3f' %(epoch + 1, cnt + 1, loss.item(), training_accuracy.item()))

            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad)
            optimizer.step()
        print('______________________________________') 
        print('\nEpoch: %d,   TRAINING ACCURACY: %.3f' %(epoch+1,running_accuracy.detach().cpu()/n))
        print('\n[Epoch: %d,  TRAINING LOSS: %.3f' %(epoch + 1,running_loss.detach().cpu()/n))
        print('______________________________________') 
        train_loss.append(running_loss.detach().cpu()/n)
        if use_scheduler==True:
            scheduler.step()

        if save_and_plot==True:

            checkpoint_path = f"{path_to_save_model}/{model_name}_{epoch + 1}.pt" 

            torch.save(model.state_dict(), checkpoint_path)

            print(f"\nSaved checkpoint at epoch {epoch + 1} to {checkpoint_path}")
        #___________________________________ONLY IF NEEDED______________________________________________________
        if run_validation==True:
            model.eval()
            with torch.no_grad():
                running_loss=0
                n=0
                for cnt, (img1, img2, label) in enumerate(vald_loader):

                    batch_dim=label.size(0)
                    n+=label.size(0)
                    img1 = img1.to(device)
                    img2 = img2.to(device)
                    label=label.view(-1,1).float().to(device)
                    
                    distance = model(img1, img2)
                    predictions=(distance>=threshold).to(device) 
                    loss=torch.mean((predictions==label).float()).to(device) 
                    running_loss+=loss*batch_dim 

                if cnt % log_step == 0:
                    #print('\nValidation: Mean of class 1 prediction',torch.mean(predictions.float()).item())
                    print('\n[Epoch: %d, Iteration: %5d]  validation Accuracy: %.3f' %(epoch + 1, cnt + 1, loss.item()))
            print('______________________________________')              
            print('\n[Epoch: %d,  VALIDATION ACCURACY: %.3f' %(epoch + 1,running_loss.detach().cpu()/n))
            print('______________________________________') 
            val_loss.append(running_loss.detach().cpu()/n)
            if running_loss/n > val_loss_best:
                val_loss_best = running_loss/n
                
#_________________________________________________________________________________________________________________________

def main():
    train_val(dataloader_train,  N_EPOCH, THRESHOLD, LOG_STEP, START_EPOCH, MYPATH,SAVE, RUN_VAL, dataloader_validation)
#_________________________________________________________________________________________________________________________

if __name__ == '__main__':
    # Impostazione del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0=time()
    #with profiler.profile(use_cuda=True) as prof:
    #__________________ INITIALIZATIONS_______________________________________________________________________________
    BATCH_SIZE=64
    N_TRAIN=7*10**4
    N_VALID=2*10**4
    N_TEST=10**4
    PIN_MEMORY=True
    RUN_VAL=True
    PHASE='Train_Validation' #create the dataloader only for training
    dataloader_train, dataloader_validation = Create_dataloader('HR',BATCH_SIZE,N_TRAIN,N_VALID,N_TEST,PIN_MEMORY,PHASE) 
    model=TeacherModel().to(device)
    #LOADING THE ALREADY TRAINED EPOCH
    #'''
    epoch_path='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/OFFICIAL_CHECKPOINT/TEACHER/TeacherModel_15.pt'
    model.load_state_dict(torch.load(epoch_path))        
    #'''
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    #optimizer = optim.SGD(model.parameters(), lr = 0.01)
    BCELOSS=nn.BCELoss().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")
    
    #'model_name' USED TO SAVE THE CHECKPOINT, HAS TO BE INSERT BY HAND
    model_name = 'TeacherModel' 
    MYPATH='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/OFFICIAL_CHECKPOINT/TEACHER'
    SAVE=False
    LOG_STEP=int(np.ceil( (N_TRAIN/BATCH_SIZE) /3 ))
    START_EPOCH=15 #ONLY IF ALREADY TRAINED AND WE'RE LOADING THE PREVIOUS PARAMETERS
    N_EPOCH=START_EPOCH+2
    THRESHOLD=0.5
    #________________TRAINING___________________________________________________________________________________________
    main()
    #print(prof.key_averages().table(sort_by="cuda_time_total"))
    print('Total time', time()-t0)
    