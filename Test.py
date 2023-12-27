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


def test(model, test_loader,log_step,threshold=0.5):
    test_acc = []
    test_loss=[]
    
    model.eval()
    with torch.no_grad():
        running_acc=0
        running_loss=0
        n=0
        for cnt, (img1, img2, label) in enumerate(test_loader):
            batch_dim=label.size(0)
            n+=batch_dim
            img1 = img1.to(device)                
            img2 = img2.to(device)
            label=label.view(-1,1).float().to(device)

            distance = model(img1, img2)
            loss=BCELOSS(distance, label)
            running_loss += loss*batch_dim
            
            predictions=(distance>=threshold).float().to(device)   
            acc=torch.mean((predictions==label).float()).to(device)                   
            running_acc+=acc*batch_dim 

            if cnt % log_step == 0:
                #print('\nValidation: Mean of class 1 prediction',torch.mean(predictions.float()).item())
                print('\n[Threshold: %.2f, Iteration: %5d]  test loss: %.3f test accuracy: %.3f' %(threshold, cnt + 1, loss.item(), acc.item()))
    print('______________________________________')
    print('\n[Threshold: %.2f, TEST ACCURACY: %.3f]' %(threshold,running_acc.detach().cpu()/n))
    print('\n[Threshold: %.2f, TEST LOSS: %.3f]' %(threshold,running_loss.detach().cpu()/n))
    print('______________________________________') 
    test_acc.append(running_acc.detach().cpu()/n)
    test_loss.append(running_loss.detach().cpu()/n)
    
    return np.array([i.item() for i in test_loss]), np.array([i.item() for i in test_acc])


#____________________________________________________________________________________________________________________________

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #__________________CHANGE  WITH YOUR OWN PATHS______________________________________________________________________
    HRPATH= 'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/224x224_lfw'
    LRPATH= 'C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/lfw_new/lfw_NEW/LR_64'
    MYPATH_TEACHER='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/CHECKPOINT/TEACHER'
    MYPATH_STUDENT_BASELINE='C:/Users/Michele/Desktop/Università/2nd year/AML/Final Project/CHECKPOINT/STUDENT_BASELINE' 
    
    epoch_path_teacher=f"{MYPATH_TEACHER}/TeacherModel_10.pt"
    epoch_path_student_baseline=f"{MYPATH_STUDENT_BASELINE}/StudentBaseline_10.pt"
    
    
    #__________________INITIALIZATION___________________________________________________________________________________
    BATCH_SIZE=64
    TOT_COUPLES=10**5
    N_TEST=int(0.1*TOT_COUPLES)
    cpu_cores=os.cpu_count()
    LOG_STEP=int(np.ceil( (N_TEST/BATCH_SIZE) /5 ))
    BCELOSS=nn.BCELoss().to(device)
    
    #_____________________DO NOT CHANGE THE FOLLOWING LINES___________________________________________________________
    
    print('\n_______________________TEACHER____________________________________________________________________________________')
    dataloader_test = Create_dataloader(Res_quality='HR',Batch_size=BATCH_SIZE,
                                                                number_of_test=N_TEST,
                                                                Phase='Test', MAX=cpu_cores,
                                                                HRpath = HRPATH,
                                                                LRpath = LRPATH )

    teacher=TeacherModel().to(device)
    teacher.load_state_dict(torch.load(epoch_path_teacher))            
    teacher_test_loss, teacher_test_acc= test(model=teacher, test_loader=dataloader_test, log_step=LOG_STEP)
    np.save(f"{MYPATH_TEACHER}/teacher_test_loss.npy",teacher_test_loss)
    np.save(f"{MYPATH_TEACHER}/teacher_test_acc.npy",teacher_test_acc)
    
    print('\n___________________STUDENT BASELINE______________________________________________________________________________')
    
    dataloader_test = Create_dataloader(Res_quality='LR',Batch_size=BATCH_SIZE,
                                                                number_of_test=N_TEST,
                                                                Phase='Test', MAX=cpu_cores,
                                                                HRpath = HRPATH,
                                                                LRpath = LRPATH )

    student_baseline=StudentModel().to(device)
    student_baseline.load_state_dict(torch.load(epoch_path_student_baseline))            
    student_baseline_test_loss, student_baseline_test_acc= test(model=student_baseline, test_loader=dataloader_test, log_step=LOG_STEP)
    np.save(f"{MYPATH_STUDENT_BASELINE}/student_baseline_test_loss.npy",student_baseline_test_loss)
    np.save(f"{MYPATH_STUDENT_BASELINE}/student_baseline_test_acc.npy",student_baseline_test_acc)
    

    