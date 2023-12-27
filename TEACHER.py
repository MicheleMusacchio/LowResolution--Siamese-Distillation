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
import warnings
warnings.filterwarnings('ignore')



def train_val(model, dataloader=None,
              n_epochs=10,threshold=0.5,log_step = 100,start_epoch=0,
              path_to_save_model=None,save_and_plot = True, run_validation=True,vald_loader=None,
              clip_grad=None,use_scheduler=False):
    
    train_loss = []
    train_acc= []
    val_loss = []
    val_acc = []
    #val_loss_best = 0 
    
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
        print('\n______________________________________') 
        print('\nEpoch: %d,   TRAINING ACCURACY: %.3f]' %(epoch+1,running_accuracy.detach().cpu()/n))
        print('\n[Epoch: %d,  TRAINING LOSS: %.3f]' %(epoch + 1,running_loss.detach().cpu()/n))
        print('\n______________________________________') 
        
        train_loss.append(running_loss.detach().cpu()/n)
        train_acc.append(running_accuracy.detach().cpu()/n)
        
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
                running_loss_val=0
                running_acc_val=0
                n=0
                for cnt, (img1, img2, label) in enumerate(vald_loader):

                    batch_dim=label.size(0)
                    n+=label.size(0)
                    img1 = img1.to(device)
                    img2 = img2.to(device)
                    label=label.view(-1,1).float().to(device)
                    
                    distance = model(img1, img2)
                    loss=BCELOSS(distance, label)
                    running_loss_val += loss*batch_dim  
                    
                    predictions=(distance>=threshold).to(device) 
                    acc=torch.mean((predictions==label).float()).to(device) 
                    running_acc_val+=acc*batch_dim 

                    if cnt % log_step == 0:
                        #print('\nValidation: Mean of class 1 prediction',torch.mean(predictions.float()).item())
                        print('\n[Epoch: %d, Iteration: %5d] validation loss: %.3f validation accuracy: %.3f' %(epoch + 1, cnt + 1, loss.item(), acc.item()))
            print('\n______________________________________')              
            print('\n[Epoch: %d,  VALIDATION ACCURACY: %.3f]' %(epoch + 1,running_acc_val.detach().cpu()/n))
            print('\n[Epoch: %d,  VALIDATION LOSS: %.3f]' %(epoch + 1,running_loss_val.detach().cpu()/n))
            print('\n______________________________________') 
            
            val_loss.append(running_loss_val.detach().cpu()/n)
            val_acc.append(running_acc_val.detach().cpu()/n)

    return np.array([i.item() for i in train_loss ]),np.array([i.item() for i in train_acc ]),np.array([i.item() for i in val_loss ]),np.array([i.item() for i in val_acc ])

                
#_______________________________________________________________________________________________________________________

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #__________________CHANGE  WITH YOUR OWN PATHS______________________________________________________________________
    HRPATH= 'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/224x224_lfw'
    LRPATH= 'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/LR_64'
    MYPATH_TEACHER='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/CHECKPOINT/TEACHER'    
    #__________________INITIALIZATION___________________________________________________________________________________
    BATCH_SIZE=32
    TOT_COUPLES=10**5
    N_TRAIN=int(0.7*TOT_COUPLES)
    N_VALID=int(0.2*TOT_COUPLES)
    cpu_cores=os.cpu_count()
    
    
    START_EPOCH=0 
    TOT_EPOCH=10
    N_EPOCH=(START_EPOCH+1) + TOT_EPOCH
    LOG_STEP=int(np.ceil( (N_TRAIN/BATCH_SIZE) /5 ))
    
    
    #_____________________DO NOT CHANGE THE FOLLOWING LINES___________________________________________________________
    
    #_______________________TEACHER____________________________________________________________________________________
    dataloader_train, dataloader_validation = Create_dataloader(Res_quality='HR',Batch_size=BATCH_SIZE,
                                                                number_of_training=N_TRAIN, number_of_validation=N_VALID,
                                                                Phase='Train_Validation', MAX=cpu_cores,
                                                                HRpath = HRPATH,
                                                                LRpath = LRPATH )
    
    teacher=TeacherModel().to(device)
    
    '''
    #LOADING THE ALREADY TRAINED EPOCH
    epoch_path=f"{MYPATH_TEACHER}/first 15/TeacherModel_12.pt"
    teacher.load_state_dict(torch.load(epoch_path))        
    '''
    
    optimizer = optim.Adam(teacher.parameters(), lr = 0.01) #optimizer = optim.SGD(model.parameters(), lr = 0.01)
    BCELOSS=nn.BCELoss().to(device)
    
    print('\n___________________________TEACHER___________________________')
    print(f"\nTotal number of parameters: {sum(p.numel() for p in teacher.parameters())}")
    print(f"\nTotal number of trainable parameters: {sum(p.numel() for p in teacher.parameters() if p.requires_grad)}")
    
    
    model_name = 'TeacherModel' 
    
    teacher_train_loss, teacher_train_acc, teacher_val_loss, teacher_val_acc =  train_val(model=teacher,
                                                                                          dataloader=dataloader_train, vald_loader=dataloader_validation,
                                                                                          n_epochs=N_EPOCH, start_epoch=START_EPOCH,
                                                                                          log_step = LOG_STEP, path_to_save_model=MYPATH_TEACHER)
    np.save(f"{MYPATH_TEACHER}/teacher_train_loss.npy",teacher_train_loss)
    np.save(f"{MYPATH_TEACHER}/teacher_train_acc.npy",teacher_train_acc)
    np.save(f"{MYPATH_TEACHER}/teacher_val_loss.npy",teacher_val_loss)
    np.save(f"{MYPATH_TEACHER}/teacher_val_acc.npy",teacher_val_acc)
    
    