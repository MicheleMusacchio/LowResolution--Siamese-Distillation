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



def train(dataloader,n_epochs=3,threshold=0.5,log_step = 100,path_to_save_model=None,save_and_plot = True, clip_grad=None,use_scheduler=False):
    train_loss = []

    for epoch in tqdm(range(n_epochs)):
        teacherModel.eval()
        studentModel.train()
        runningLoss = 0
        running_accuracy=0
        n=0
        for cnt, batch in enumerate(dataloader):
            imgLR1 = batch[1].to(device)
            imgLR2 = batch[3].to(device)
            imgHR1 = batch[0].to(device)
            imgHR2 = batch[2].to(device)
            label = batch[4].view(-1,1).float().to(device)
            batch_dim = label.size(0)
            n+=batch_dim
            
            optimizer.zero_grad()
            with torch.no_grad():
                hrOut = teacherModel(imgHR1, imgHR2)
                
            lrOut  = studentModel(imgLR1, imgLR2) 

            softTargetsLoss = SLoss(hrOut,lrOut)
            hardTargetsLoss = contLoss(lrOut , label) 
            
            Loss=KD_loss(softTargetsLoss,hardTargetsLoss)
            
            Loss.backward()
            optimizer.step()
            runningLoss += Loss*batch_dim
            predictions=(lrOut >= threshold).float().to(device) 

            training_accuracy=torch.mean((predictions==label).float()).to(device) 
            running_accuracy+=training_accuracy*batch_dim 
            if cnt % log_step == 0:
                #print('Training: Mean of class 1 prediction',torch.mean(predictions.float()).item())
                print('[Epoch: %d, Iteration: %5d]  training loss: %.3f training accuracy: %.3f' %(epoch + 1, cnt + 1, Loss.item(), training_accuracy.item()))

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(studentModel.parameters(),clip_grad)
            
        print('______________________________________')
        print('Epoch: %d,   TRAINING ACCURACY: %.3f' %(epoch+1,running_accuracy.detach().cpu()/n))
        print('[Epoch: %d,  TRAINING LOSS: %.3f' %(epoch + 1,runningLoss.detach().cpu()/n))
        print('______________________________________')
        train_loss.append(runningLoss.detach().cpu()/n)
        if save_and_plot:

                checkpoint_path = f"{path_to_save_model}/{model_name}_{epoch + 1}.pt" # EDIT FRANCESCO

                # Save the model's state dictionary and architecture
                torch.save(studentModel.state_dict(), checkpoint_path)

                print(f"Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}")

                
###########################################################################################################################

def main():
    train(dataloader_train,  N_EPOCH, THRESHOLD, LOG_STEP, MYPATH, SAVE)

############################################################################################################################

if __name__ == '__main__':
    # Impostazione del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0=time()
    #with profiler.profile(use_cuda=True) as prof:
    #__________________ INITIALIZATIONS_______________________________________________________________________________
    BATCH_SIZE=50
    N_TRAIN=10**4
    N_VALID=2*10**4
    N_TEST=10**4
    PIN_MEMORY=True
    PHASE='Train' #create the dataloader only for training
    dataloader_train = Create_dataloader('ALL',BATCH_SIZE,N_TRAIN,N_VALID,N_TEST,PIN_MEMORY,PHASE) 
    
    teacherModel = TeacherModel().to(device)
    epoch_path_teacher='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/OFFICIAL_CHECKPOINT/TEACHER/TeacherModel_14.pt'
    teacherModel.load_state_dict(torch.load(epoch_path_teacher))   
    for param in teacherModel.parameters():
        param.requires_grad=False
    # Number of trainable params in our resnet( Just for check)
    trainable_params = print('\nNumber of trainable par in TEACHER:', sum(p.numel() for p in teacherModel.parameters() if p.requires_grad)) #should be 0
    
    studentModel = StudentModel_2().to(device)
    '''
    epoch_path_student='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/OFFICIAL_CHECKPOINT/DISTILLED/DistilledKnowledge_9.pt'
    studentModel.load_state_dict(torch.load(epoch_path_student))
    '''
    print(f"\nTotal number of parameters for STUDENT: {sum(p.numel() for p in studentModel.parameters())}")
    print(f"\nTotal number of trainable parameters for STUDENT: {sum(p.numel() for p in studentModel.parameters() if p.requires_grad)}")
    #LOADING THE ALREADY TRAINED EPOCH
    
    optimizer = optim.Adam(studentModel.parameters(), lr = 0.01) #0.1)
    contLoss=nn.BCELoss().to(device)
    KD_loss=KD_Loss(alpha=1,beta=0.8).to(device)
    SLoss=Soft_loss().to(device)

    #'model_name' USED TO SAVE THE CHECKPOINT, HAS TO BE INSERT BY HAND
    model_name = 'DistilledKnowledge' 
    MYPATH='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/OFFICIAL_CHECKPOINT/DISTILLED'
    SAVE=False
    LOG_STEP=int(np.ceil( (N_TRAIN/BATCH_SIZE) /3 ))
    START_EPOCH=0 #ONLY IF ALREADY TRAINED AND WE'RE LOADING THE PREVIOUS PARAMETERS
    N_EPOCH=10
    THRESHOLD=0.5
    #________________TRAINING___________________________________________________________________________________________
    main()
    #print(prof.key_averages().table(sort_by="cuda_time_total"))
    print('Total time', time()-t0)
    