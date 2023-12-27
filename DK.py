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
from Models import TeacherModel,StudentModel
from Losses import KD_Loss,Soft_loss
import warnings
warnings.filterwarnings('ignore')



def dk_train_val(dataloader=None,run_validation=True,dataloader_val=None,
                 start_epoch=0,n_epochs=10,
                 threshold=0.5,log_step = 100,
                 path_to_save_model=None,save_and_plot = True,
                 clip_grad=None,use_scheduler=False):
    
    train_loss = []
    train_acc= []
    val_loss = []
    val_acc = []

    for epoch in tqdm(range(start_epoch,n_epochs-1)):
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
        print('Epoch: %d,   TRAINING ACCURACY: %.3f]' %(epoch+1,running_accuracy.detach().cpu()/n))
        print('[Epoch: %d,  TRAINING LOSS: %.3f]' %(epoch + 1,runningLoss.detach().cpu()/n))
        print('______________________________________')
        train_loss.append(runningLoss.detach().cpu()/n)
        train_acc.append(running_accuracy.detach().cpu()/n)
        
        if save_and_plot:
                checkpoint_path = f"{path_to_save_model}/{model_name}_{epoch + 1}.pt" 
                torch.save(studentModel.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}")
                
        #___________________________________ONLY IF NEEDED______________________________________________________
        if run_validation==True:
            studentModel.eval()
            teacherModel.eval()
            with torch.no_grad():
                running_loss_val=0
                running_acc_val=0
                n=0
                for cnt, batch in enumerate(dataloader_val):
                    imgLR1 = batch[1].to(device)
                    imgLR2 = batch[3].to(device)
                    imgHR1 = batch[0].to(device)
                    imgHR2 = batch[2].to(device)
                    label = batch[4].view(-1,1).float().to(device)
                    batch_dim = label.size(0)
                    n+=batch_dim


                    HROut = teacherModel(imgHR1, imgHR2)
                    LROut  = studentModel(imgLR1, imgLR2) 

                    SoftTargetsLoss = SLoss(HROut,LROut)
                    HardTargetsLoss = contLoss(LROut , label) 
                    loss=KD_loss(SoftTargetsLoss,HardTargetsLoss)
                    
                    loss=contLoss(LROut, label)
                    running_loss_val += loss*batch_dim  
                    
                    predictions=(LROut>=threshold).to(device) 
                    acc=torch.mean((predictions==label).float()).to(device) 
                    running_acc_val+=acc*batch_dim 

                    if cnt % log_step == 0:
                        #print('\nValidation: Mean of class 1 prediction',torch.mean(predictions.float()).item())
                        print('\n[Epoch: %d, Iteration: %5d] validation loss %.3f validation accuracy: %.3f' %(epoch + 1, cnt + 1, loss.item(), acc.item()))
            print('\n______________________________________')              
            print('\n[Epoch: %d,  VALIDATION ACCURACY: %.3f]' %(epoch + 1,running_acc_val.detach().cpu()/n))
            print('\n[Epoch: %d,  VALIDATION LOSS: %.3f]' %(epoch + 1,running_loss_val.detach().cpu()/n))
            print('\n______________________________________') 
            
            val_loss.append(running_loss_val.detach().cpu()/n)
            val_acc.append(running_acc_val.detach().cpu()/n)

    return np.array([i.item() for i in train_loss ]),np.array([i.item() for i in train_acc ]),np.array([i.item() for i in val_loss ]),np.array([i.item() for i in val_acc ])

#________________________________________________________________________________________________________________________

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #__________________CHANGE  WITH YOUR OWN PATHS______________________________________________________________________
    HRPATH= 'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/224x224_lfw'
    LRPATH= 'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/LR_64'
    MYPATH_TEACHER='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/CHECKPOINT/TEACHER/100K'
    MYPATH_STUDENT_DISTILLED='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/CHECKPOINT/STUDENT_DISTILLED'
    epoch_path_teacher=f"{MYPATH_TEACHER}/TeacherModel_xx.pt"
    
    #__________________INITIALIZATION___________________________________________________________________________________
    BATCH_SIZE=64
    TOT_COUPLES=10**5
    N_TRAIN=int(0.7*TOT_COUPLES)
    N_VALID=int(0.2*TOT_COUPLES)
    cpu_cores=os.cpu_count()
    
    
    START_EPOCH=0 
    TOT_EPOCH=10
    N_EPOCH=(START_EPOCH+1) + TOT_EPOCH
    LOG_STEP=int(np.ceil( (N_TRAIN/BATCH_SIZE) /5 ))
    
    
    #_____________________DO NOT CHANGE THE FOLLOWING LINES___________________________________________________________
    
    dataloader_train, dataloader_validation = Create_dataloader(Res_quality='ALL',Batch_size=BATCH_SIZE,
                                                                number_of_training=N_TRAIN, number_of_validation=N_VALID,
                                                                Phase='Train_Validation', MAX=cpu_cores,
                                                                HRpath = HRPATH,
                                                                LRpath = LRPATH )
    
    teacherModel = TeacherModel().to(device) 
    teacherModel.load_state_dict(torch.load(epoch_path_teacher))
    for param in teacherModel.parameters():
        param.requires_grad=False    
    print('\n_______________TEACHER_______________')
    print(f"\nTotal number of parameters: {sum(p.numel() for p in teacherModel.parameters())}")
    print(f"\nTotal number of trainable parameters: {sum(p.numel() for p in teacherModel.parameters() if p.requires_grad)}")
    
    
    studentModel = StudentModel().to(device)
    '''
    epoch_path_student=f"{MYPATH_STUDENT_DISTILLED}/DistilledKnowledge_xx.pt"
    studentModel.load_state_dict(torch.load(epoch_path_student))
    '''
    print('\n_______________STUDENT_______________')
    print(f"\nTotal number of parameters: {sum(p.numel() for p in studentModel.parameters())}")
    print(f"\nTotal number of trainable parameters: {sum(p.numel() for p in studentModel.parameters() if p.requires_grad)}")
    print('\n_____________________________________________________________________________________________________________')
    
    optimizer = optim.Adam(studentModel.parameters(), lr = 0.01) #optimizer = optim.SGD(model.parameters(), lr = 0.01)
    contLoss=nn.BCELoss().to(device)
    KD_loss=KD_Loss(alpha=1,beta=0.8).to(device)
    SLoss=Soft_loss().to(device)

    model_name = 'DistilledKnowledge' 
    
    student_train_loss, student_train_acc, student_val_loss, student_val_acc = dk_train_val(dataloader=dataloader_train, dataloader_val=dataloader_validation,
                                                                                            start_epoch=START_EPOCH, n_epochs=N_EPOCH,
                                                                                            log_step = LOG_STEP,path_to_save_model=MYPATH_STUDENT_DISTILLED)
    np.save(f"{MYPATH_STUDENT_DISTILLED}/DK_train_loss.npy",student_train_loss)
    np.save(f"{MYPATH_STUDENT_DISTILLED}/DK_train_acc.npy",student_train_acc)
    np.save(f"{MYPATH_STUDENT_DISTILLED}/DK_val_loss.npy",student_val_loss)
    np.save(f"{MYPATH_STUDENT_DISTILLED}/DK_val_acc.npy",student_val_acc)                                                                                 
                                                                                              
