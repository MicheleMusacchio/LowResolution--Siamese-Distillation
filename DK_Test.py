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



def dk_test(test_loader,log_step,threshold=0.5):
    
    test_loss = []
    test_acc= []

    with torch.no_grad():
        teacherModel.eval()
        student_dk.eval()
        runningLoss = 0
        running_accuracy=0
        n=0
        for cnt, batch in enumerate(test_loader):
            imgLR1 = batch[1].to(device)
            imgLR2 = batch[3].to(device)
            imgHR1 = batch[0].to(device)
            imgHR2 = batch[2].to(device)
            label = batch[4].view(-1,1).float().to(device)
            batch_dim = label.size(0)
            n+=batch_dim


            hrOut = teacherModel(imgHR1, imgHR2)
            lrOut  = student_dk(imgLR1, imgLR2) 

            softTargetsLoss = SLoss(hrOut,lrOut)
            hardTargetsLoss = contLoss(lrOut , label) 

            Loss=KD_loss(softTargetsLoss,hardTargetsLoss)


            runningLoss += Loss*batch_dim
            predictions=(lrOut >= threshold).float().to(device) 

            accuracy=torch.mean((predictions==label).float()).to(device) 
            running_accuracy+=training_accuracy*batch_dim 
            if cnt % log_step == 0:
                #print('Training: Mean of class 1 prediction',torch.mean(predictions.float()).item())
                print('[Iteration: %5d]  test loss: %.3f test accuracy: %.3f' %( cnt + 1, Loss.item(), accuracy.item()))


            
        print('______________________________________')
        print('[TEST ACCURACY: %.3f]' %(running_accuracy.detach().cpu()/n))
        print('[TEST LOSS: %.3f]' %(runningLoss.detach().cpu()/n))
        print('______________________________________')
        test_loss.append(runningLoss.detach().cpu()/n)
        test_acc.append(running_accuracy.detach().cpu()/n)
        
        
    return np.array([i.item() for i in test_loss ]), np.array([i.item() for i in test_acc ])

#________________________________________________________________________________________________________________________

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #__________________CHANGE  WITH YOUR OWN PATHS______________________________________________________________________
    HRPATH= 'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/224x224_lfw'
    LRPATH= 'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/LR_64'
    MYPATH_TEACHER='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/CHECKPOINT/TEACHER'
    MYPATH_STUDENT_DISTILLED='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/CHECKPOINT/STUDENT_DISTILLED'
    epoch_path_teacher=f"{MYPATH_TEACHER}/TeacherModel_xx.pt"
    epoch_path_student_dk=f"{MYPATH_STUDENT_DISTILLED}/DistilledKnowledge_xx.pt"
    
    #__________________INITIALIZATION___________________________________________________________________________________
    BATCH_SIZE=64
    TOT_COUPLES=10**5
    N_TEST=int(0.1*TOT_COUPLES)
    cpu_cores=os.cpu_count()
    LOG_STEP=int(np.ceil( (N_TEST/BATCH_SIZE) /5 ))
    
    #_____________________DO NOT CHANGE THE FOLLOWING LINES___________________________________________________________                                                                           
                                                                                              
    print('\n_______________STUDENT DISTILLED_________________________________________________________________________________')
    dataloader_test = Create_dataloader(Res_quality='ALL',Batch_size=BATCH_SIZE,
                                                                number_of_test=N_TEST,
                                                                Phase='Test', MAX=cpu_cores,
                                                                HRpath = HRPATH,
                                                                LRpath = LRPATH )
    teacherModel = TeacherModel().to(device)
    teacherModel.load_state_dict(torch.load(epoch_path_teacher))   

    student_dk=StudentModel().to(device)
    student_dk.load_state_dict(torch.load(epoch_path_student_dk))
    
    contLoss=nn.BCELoss().to(device)
    KD_loss=KD_Loss(alpha=1,beta=0.8).to(device)
    SLoss=Soft_loss().to(device)

    student_dk_test_loss, student_dk_test_acc= dk_test(test_loader=dataloader_test, log_step=LOG_STEP)
    np.save(f"{MYPATH_STUDENT_DISTILLED}/student_dk_test_loss.npy",student_dk_test_loss)
    np.save(f"{MYPATH_STUDENT_DISTILLED}/student_dk_test_acc.npy",student_dk_test_acc)