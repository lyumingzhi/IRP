import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F

import torchvision

import utils
from madrys import MadrysLoss
import pickle
from torch.utils.data import Subset, Dataset, DataLoader
import os
import argparse
import numpy as np
from PIL import Image
import random
from sklearn.linear_model import LinearRegression


from util import setup_logger, progress_bar
# from models.vit import ViT
from models.MLP import MLP
from models.vgg import VGG
from augmentations import *

# load images
def get_img( im_name,cnns):
    sample = Image.open(im_name).convert('RGB') 
    if sample is None:
        print(im_name)
    
    transform_train = transforms.Compose([])
    transform_train.transforms.append(transforms.RandomResizedCrop(224))
    transform_train.transforms.append(transforms.RandomHorizontalFlip())
    transform_train.transforms.append(transforms.ToTensor())
    sample = transform_train(sample)

    
    # perform CUDA poison
    with torch.no_grad():                
        kernel = cnns[label]
        cuda_tensor = torch.nn.functional.conv2d(sample, torch.from_numpy(kernel).float(), stride=1, groups=3, padding='same')
        ### clip convolution image to valid value
        
        # cuda_tensor = sample
        # sample /= sample.max()
        # cudaimg =sample.numpy().transpose((1,2,0))*255
        # cudaimg = np.clip(cudaimg, a_min=0, a_max=255)
        # img_save = Image.fromarray(cudaimg.astype(np.uint8))
        # to_img = transforms.ToTensor()
        # sample =to_img(img_save)

                
    return sample,  cuda_tensor 

# get CUDA kernel
def get_filter_unlearnable(blur_parameter, center_parameter, grayscale, kernel_size, seed, same):

    np.random.seed(seed)
    cnns = []
    with torch.no_grad():
        for i in range(100): 
            # cnns.append(torch.nn.Conv2d(3, 3, kernel_size, groups=3, padding=4))
            if blur_parameter is None:
                blur_parameter = 1

            w = np.random.uniform(low=0, high=blur_parameter, size=(3,1,kernel_size,kernel_size))
            if center_parameter is not None:
                shape = w[0][0].shape
                w[0, 0, np.random.randint(shape[0]), np.random.randint(shape[1])] = center_parameter
                # w[0, 0, shape[0]//2, shape[1]//2] = center_parameter



            w[1] = w[0]
            w[2] = w[0]

            cnns.append(w)

    cnns = np.stack(cnns)
    return cnns

if __name__ == '__main__':

    
    load =True

    grayscale = False # grayscale
    blur_parameter = 0.06
    center_parameter = 1.0
    kernel_size = 9
    seed = 0
    same = False
    cnns = get_filter_unlearnable(blur_parameter, center_parameter, grayscale, kernel_size, seed, same) # initialize cnns kernel
    A_list = []



    im_names = []
    targets = []

    with open('/home2/huangyi/ImageShortcutSqueezing/list/clean_list.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(' ')
            im_names.append(data[0])
            targets.append(int(data[1]))
    label_cnt = [0]*100
    image_cnt =[0]*100
    targets = np.array(targets)
    for i in range(100):
        image_cnt[i] = np.count_nonzero(targets == i)

    if load == False: # if there is no existing CUDA kernel, CUDA kernels are required to generated in advance.
        for idx in range(len(im_names)):
            im_name = im_names[idx]
            label = int(targets[idx])
            label_cnt[label]+=1
            # print(label_cnt[label])
            if label_cnt[label]<=image_cnt[label]:
                img, cuda_tensor = get_img(im_name,cnns=cnns) # load img and perform CUDA poison
           
                cudaimg =cuda_tensor.numpy().transpose((1,2,0))
          
                img =img.numpy().transpose((1,2,0))
           
                patch_size = 9 
                pnum= 224//9
                y_list=[]
                cudalist = []
                # crop patches from the CUDA poisoned images
                for i in range(pnum):
                    for j in range(pnum):
                        patch = cudaimg[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,0]   
                        center = patch_size//2
                        y_gt = img[i*patch_size+center, j*patch_size+center, 0]
                        cudalist.append(np.reshape(patch,(-1,)))
                        y_list.append(y_gt)
                            
            
            # do regression to get the recovery matrix A
            if label_cnt[label]==image_cnt[label]:
                print(label)
                cudalist=np.asarray(cudalist)
                y_list = np.asarray(y_list)
                reg = LinearRegression().fit(cudalist, y_list) 
                score = reg.score(cudalist, y_list)
                print('recory score', score) 
                A =  reg.coef_
                A = np.reshape(A,(1,1,9,9))
                A = np.repeat(A,3, axis=0)
                A_list.append(A)
        A = np.asarray(A_list)
        np.save('A.npy', A, allow_pickle=True)



    if load ==True: # if there is a CUDA kernel, load it first. Here, take A.npy as an example. The formate of A is the same as above generated kernel.
        A_arr = np.load('A.npy',allow_pickle=True)
        for idx in range(len(im_names)):
            
            im_name = im_names[idx]
            label = int(targets[idx])
            label_cnt[label]+=1
            if label_cnt[label]<2 and label<5:
                # img, cuda_tensor = get_img(im_name,cnns=cnns)
                img, cuda_tensor = get_img(im_name,cnns=A_arr)
         
                cudaimg =cuda_tensor.numpy().transpose((1,2,0))
 
                img =img.numpy().transpose((1,2,0))

                patch_size = 9 
                pnum= 224//9
                y_list=[]
                cudalist = []
                # crop patches from the CUDA poisoned images 
                for i in range(pnum):
                    for j in range(pnum):
                        patch = cudaimg[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,0]   
                        center = patch_size//2
                        y_gt = img[i*patch_size+center, j*patch_size+center, 0]
                        cudalist.append(np.reshape(patch,(-1,)))
                        y_list.append(y_gt)
                            
                cudalist=np.asarray(cudalist)
                y_list = np.asarray(y_list)

                # do regression to get the recovery matrix A
                reg = LinearRegression().fit(cudalist, y_list) 
                score = reg.score(cudalist, y_list)
                print(score)
                A =  reg.coef_
                A = np.reshape(A,(1,1,9,9))
                A = np.repeat(A,3, axis=0)

                reocver_tensor = torch.nn.functional.conv2d(cuda_tensor, torch.from_numpy(A).float(), stride=1, groups=3, padding='same')
                ### clip convolution image to valid value
                reocver_tensor /= reocver_tensor.max()
                reocver_tensor =reocver_tensor.numpy().transpose((1,2,0))*255
                recoverimg = np.clip(reocver_tensor, a_min=0, a_max=255)
                img_save = Image.fromarray(recoverimg.astype(np.uint8))
                img_save.save('recover_{}.png'.format(label))

                # img =img.numpy().transpose((1,2,0))
                img_save = Image.fromarray((img*255).astype(np.uint8))
                img_save.save('ori_{}.png'.format(label))



                cuda_tensor /= cuda_tensor.max()
                cudaimg =cuda_tensor.numpy().transpose((1,2,0))
                cudaimg = np.clip(cudaimg*255, a_min=0, a_max=255)
                img_save = Image.fromarray(cudaimg.astype(np.uint8))
                img_save.save('cuda_{}.png'.format(label))
            