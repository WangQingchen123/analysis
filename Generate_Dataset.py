import numpy as np
import torch 
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from pytorch_lightning.core import LightningModule
# from pytorch_lightning.metrics.functional import  accuracy 
from pytorch_lightning import loggers as pl_loggers
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import string


# 以HAR数据集为例
# 30个志愿者，30个域，也可以是多域
# 先生成闭集域适应数据集，每个志愿者一个.pt，包含6种状态


def data_generator(data_dir,save=True):
    # domain labels
    subject_train = np.loadtxt('./train/subject_train.txt')
    subject_test = np.loadtxt('./test/subject_test.txt')

    # samples
    train_body_acc_x = np.loadtxt('./train/Inertial Signals/body_acc_x_train.txt')
    train_body_acc_y = np.loadtxt('./train/Inertial Signals/body_acc_y_train.txt')
    train_body_acc_z = np.loadtxt('./train/Inertial Signals/body_acc_z_train.txt')
    train_body_gyro_x = np.loadtxt('./train/Inertial Signals/body_gyro_x_train.txt')
    train_body_gyro_y = np.loadtxt('./train/Inertial Signals/body_gyro_y_train.txt')
    train_body_gyro_z = np.loadtxt('./train/Inertial Signals/body_gyro_z_train.txt')
    train_total_acc_x = np.loadtxt("./train/Inertial Signals/total_acc_x_train.txt")
    train_total_acc_y = np.loadtxt("./train/Inertial Signals/total_acc_y_train.txt")
    train_total_acc_z = np.loadtxt("./train/Inertial Signals/total_acc_z_train.txt")


    test_body_acc_x = np.loadtxt('./test/Inertial Signals/body_acc_x_test.txt')
    test_body_acc_y = np.loadtxt('./test/Inertial Signals/body_acc_y_test.txt')
    test_body_acc_z = np.loadtxt('./test/Inertial Signals/body_acc_z_test.txt')
    test_body_gyro_x = np.loadtxt('./test/Inertial Signals/body_gyro_x_test.txt')
    test_body_gyro_y = np.loadtxt('./test/Inertial Signals/body_gyro_y_test.txt')
    test_body_gyro_z = np.loadtxt('./test/Inertial Signals/body_gyro_z_test.txt')
    test_total_acc_x = np.loadtxt("./test/Inertial Signals/total_acc_x_test.txt")
    test_total_acc_y = np.loadtxt("./test/Inertial Signals/total_acc_y_test.txt")
    test_total_acc_z = np.loadtxt("./test/Inertial Signals/total_acc_z_test.txt")

    train_data = np.stack((train_body_acc_x,train_body_acc_y,train_body_acc_z,
                            train_body_gyro_x,train_body_gyro_y,train_body_gyro_z,
                            train_total_acc_x,train_total_acc_y,train_total_acc_z),axis=1)
    test_data = np.stack((test_body_acc_x,test_body_acc_y,test_body_acc_z,
                        test_body_gyro_x,test_body_gyro_y,test_body_gyro_z,
                        test_total_acc_x,test_total_acc_y,test_total_acc_z),axis=1)
    
    # labels
    train_labels = np.loadtxt('./train/y_train.txt')
    train_labels -= np.min(train_labels)
    test_labels = np.loadtxt('./test/y_test.txt')
    test_labels -= np.min(test_labels)

    all_subjects_data = np.concatenate((train_data,test_data))
    all_subjects_labels= np.concatenate((train_labels,test_labels))
    subject_indices = np.concatenate((subject_train,subject_test))

    domain_names = ["29","30"]  # 改这里
    ## 根据标签在进行筛选
    desired_labels = np.array([2,5,6])-1  # 改这里
    # domain_names = [str(i+1) for i in range(30)]
    for i in domain_names:
        domain_index = int(i)
        domain_data = all_subjects_data[subject_indices == domain_index]
        domain_labels = all_subjects_labels[subject_indices == domain_index]
        mask = np.isin(domain_labels,desired_labels)
        domain_data = domain_data[mask]
        domain_labels = domain_labels[mask]
        X_train,X_test,Y_train,Y_test = train_test_split(domain_data,domain_labels,test_size=0.2,random_state=1)
        data_id = [i for i in range(Y_train.shape[0]+Y_test.shape[0])]
        HAR_dataset_processed = {'train':{'samples':X_train,'labels':Y_train,'ids':np.array(data_id[:Y_train.shape[0]])},
            'test':{'samples':X_test,'labels':Y_test,'ids':np.array(data_id[Y_train.shape[0]:])}}
        if save:
            torch.save(HAR_dataset_processed['train'],f'./train_test_pt/train_{i}.pt')
            torch.save(HAR_dataset_processed['test'],f'./train_test_pt/test_{i}.pt')
        # domain_data.append(all_subjects_data[np.where((i< subject_indices)&( subject_indices<=j))])
        # domain_labels.append(all_subjects_labels[np.where((i< subject_indices)&( subject_indices<=j))])



    # ## 根据标签在进行筛选
    # desired_labels = [0,1,2,3]  # 改这里
    # label_mask = np.isin(all_subjects_labels,desired_labels)

    # domain_data_indices = np.where(label_mask)
    # domain_data_desired = all_subjects_data[domain_data_indices]
    # domain_labels_desired = all_subjects_labels[domain_data_indices]


    
    # domain_data.append(domain_data_desired[np.where((subject_indices[domain_data_indices]>20)&(subject_indices[domain_data_indices]<=30))])
    # domain_labels.append(domain_labels_desired[np.where((subject_indices[domain_data_indices]>20)&(subject_indices[domain_data_indices]<=30))])

    
    # HAR_dataset_processed = {}
    # for domain_data,domain_labels,name in zip(domain_data,domain_labels,domain_names):
    #     X_train,X_test,Y_train,Y_test = train_test_split(domain_data,domain_labels,test_size=0.2,random_state=1)
    #     data_id = [i for i in range(Y_train.shape[0]+Y_test.shape[0])]
    #     HAR_dataset_processed[name] = {'train':{'samples':X_train,'labels':Y_train,'ids':np.array(data_id[:Y_train.shape[0]])},
    #         'test':{'samples':X_test,'labels':Y_test,'ids':np.array(data_id[Y_train.shape[0]:])}}
        
    #     if save:
    #         torch.save(HAR_dataset_processed[name]['train'],f'train_{name}.pt')
    #         torch.save(HAR_dataset_processed[name]['test'],f'test_{name}.pt')


data_generator('./',save=True)








