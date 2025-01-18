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

from models import *
from util import setup_logger, progress_bar
# from models.vit import ViT
from models.MLP import MLP
from augmentations import *
from poison_loaders import *
import torchattacks


parser = argparse.ArgumentParser(description='PyTorch  Training')
parser.add_argument('--poison_type', default='CUDA', help='poison type')
# parser.add_argument('--poison_path', default=None, help='path to the folder of poisoned images')
parser.add_argument('--poison_rate', default=1, type=float, help='poison rate (by a random seed)')

parser.add_argument('--grayscale', default=False, type=bool, help='grayscale compression')
parser.add_argument('--jpeg', default=None, type=int, help='JPEG quality factor')
parser.add_argument('--bdr', default=None, type=int, help='bit depth')
parser.add_argument('--AT', default=False, type=bool, help='PGD Adversarial training')
parser.add_argument('--AT_eps', default=0.031, type=float, help='poison_rate')
parser.add_argument('--TrainAUG', default='', help='Train augmentations')
parser.add_argument('--lowpass', default='', help='filtering')
parser.add_argument('--ISS_both_train_test', default=False)
parser.add_argument('--adaptive_path', default=None)

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--net', default='resnet18', help='models to train')

parser.add_argument('--exp_path', default='../EXPERIMENTS/TEMP/', help='exp_path')
parser.add_argument('--progress_bar_show', default=False, type=bool)
parser.add_argument('--resume', type=str,default='')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eval', type=bool, default=False)

args = parser.parse_args()


torch.cuda.set_device(args.gpu)


if not os.path.exists(args.exp_path):
    os.makedirs(args.exp_path)

log_file_path = os.path.join(args.exp_path, args.poison_type)
logger = setup_logger(name=args.poison_type, log_file=log_file_path + ".log")
logger.info("PyTorch Version: %s" % (torch.__version__))
logger.info('Poisons are: %s', args.poison_type)

logger.info('Mixup / Cutout / CutMix:  %s', str(args.TrainAUG))
logger.info('Grayscale compression for both train and test:  %s', str(args.grayscale))
logger.info('Bit Depth Reduction:  %s', str(args.bdr))
logger.info('JPEG compression quality:  %s', str(args.jpeg))
logger.info('Lowpass:  %s', str(args.lowpass))

logger.info('Training on:  %s', str(args.net))
logger.info('AT:  %s' 'with epsilon %s', str(args.AT), str(args.AT_eps))
logger.info('both train and test:  %s', str(args.ISS_both_train_test))
logger.info('poison rate:  %s', str(args.poison_rate))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

   
class ImageDataset(Dataset):
    def __init__(self, ann_file, transform=None, patch_noise=False):
        self.ann_file = ann_file
        self.transform = transform
        self.patch_noise = patch_noise
        self.init()

    def init(self):
        
        self.im_names = []
        self.targets = []
        self.labels= []
        self.classes=[]
        self.labels_to_target={}
        self.target_to_label={}
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(' ')
                self.im_names.append(data[0])
                self.targets.append(data[1])
        with open(self.ann_file[:-8]+'name.txt', 'r') as f:
            lines = f.readlines()
            i=0
            for line in lines:
                data = line.strip().split(' ')
                self.labels.append(int(data[0]))
                self.classes.append(data[1])
                self.labels_to_target[data[0]] = i
                self.target_to_label[i]=data[0]
                i+=1




    def __getitem__(self, index):
        im_name = self.im_names[index]
        # target = self.targets[index]
        label = self.labels_to_target[self.targets[index]]
        sample = Image.open(im_name).convert('RGB') 
        if sample is None:
            print(im_name)
        
        sample = self.transform(sample)




       # generate mixed sample
        lam = 0.8
        rand_index = random.choice(range(len(self)))


        img2 = self.im_names[rand_index]
        lb2 = self.labels_to_target[self.targets[rand_index]]
        img2 = Image.open(img2).convert('RGB') 
        img2 = self.transform(img2)


        sample = sample * lam + img2 * (1-lam)

        return sample, label

    def __len__(self):
        return len(self.im_names)

def train_loader(data_path, transform, patch_noise):

    # [NO] do not use normalize here cause it's very hard to converge
    # [NO] do not use colorjitter cause it lead to performance drop in both train set and val set

    # [?] guassian blur will lead to a significantly drop in train loss while val loss remain the same
    
    train_dataset = ImageDataset(data_path, transform=transform,patch_noise =patch_noise)  
    
     
    # if len(train_dataset)>30000:
    #     # Define the desired subset size
    #     subset_size = len(train_dataset)//5

    #     # Create a subset of the CIFAR10 dataset
    #     subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    #     subset_dataset = Subset(train_dataset, subset_indices)
    # else:
    #     subset_dataset = train_dataset
    
    subset_dataset = train_dataset


    batch_size = args.batch_size
    train_loader = DataLoader(
        subset_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True, sampler=None)


    return train_loader


# Data
print('==> Preparing data augmentation')

transform_train = aug_train(args.jpeg, args.grayscale, args.bdr, args.TrainAUG, args.lowpass,'IN100')
transform_test = aug_test(args.ISS_both_train_test, args.jpeg, args.grayscale, args.bdr,'IN100')

logger.info("Training transformation %s" % (transform_train))
logger.info("Test transformation %s" % (transform_test))



trainloader = train_loader(data_path='/home2/huangyi/ImageShortcutSqueezing/list/{}_list.txt'.format(args.poison_type), transform= transform_train, patch_noise = False)
# testloader =  train_loader(data_path='/home2/huangyi/ImageShortcutSqueezing/list/IN100val_list.txt',  transform= transform_test, patch_noise = False)
testloader =   train_loader(data_path='/home2/huangyi/ImageShortcutSqueezing/list/clean_list.txt',  transform= transform_test, patch_noise = False)



print('==> Building model..')

if args.net == 'resnet18':
    net = ResNet18(num_classes=100)

print(net)

net = net.cuda()


if 'mixup' in args.TrainAUG:
    criterion = cross_entropy
elif 'cutmix' in args.TrainAUG:
    criterion = CutMixCrossEntropyLoss()
else:
    criterion = nn.CrossEntropyLoss()

criterion = nn.CrossEntropyLoss()



optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.AT:
    epochs = 100
else:
    epochs = 100
    
if args.net == 'vit':
    epochs = 200
args.epochs = epochs

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

start_epoch = 0



if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        start_epoch = checkpoint['epoch']
        try:
            net.load_state_dict(checkpoint['state_dict'])
        except:
            state_dict = checkpoint['state_dict']
            def load_my_state_dict(model, state_dict):
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                            continue
                    if isinstance(param, torch.nn.parameter.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    own_state[name].copy_(param)
            load_my_state_dict(net,state_dict)
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


# Training
def train(epoch):

    attacker = torchattacks.PGD
    eps = args.AT_eps
    steps = 10
    atk = attacker(net, eps=eps, alpha=eps/steps * 1.5, steps=steps)
    correct, total = 0, 0   

    losses_ce = utils.AverageMeter('CELoss', ':.4f')
    # learning_rate = utils.AverageMeter('LR', ':.4f')
    top1 = utils.AverageMeter('Top1', ':.4f')
    top5 = utils.AverageMeter('Top5', ':.4f')
    print('\nEpoch: %d' % epoch)
    net.train()
    progress = utils.ProgressMeter(
        len(trainloader),
        [  losses_ce,top1,top5],
        prefix="Epoch:[{}]".format(epoch+1))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        # if args.AT:
        #     outputs, loss = MadrysLoss(cutmix=('cutmix' in args.TrainAUG), epsilon=args.AT_eps)(net,  inputs, targets, optimizer)
        # else:
        #     outputs = net(inputs)
        #     loss = criterion(outputs, targets)

        if args.AT:
            inputs = atk(inputs, targets)
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0], inputs.size(0))    
        top5.update(acc5[0], inputs.size(0))
        losses_ce.update(loss.item(), inputs.size(0))

        _ , predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        if batch_idx % 100 == 0 :
            progress.display(batch_idx)
    acc = correct / total
    # print('Train Accuracy %.2f\n' % (acc*100))

    logger.info('Train Accuracy %.2f\n' % (acc*100))


def test(epoch):
    net.eval()

    correct, total = 0, 0
    top1 = utils.AverageMeter('Top1', ':.4f')
    top5 = utils.AverageMeter('Top5', ':.4f')
    
    progress = utils.ProgressMeter(
        len(testloader),
        [ top1, top5],
        prefix="Epoch: [{}]".format(epoch+1))


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            loss = nn.CrossEntropyLoss()(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))     
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            
            _ , predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if batch_idx % 100 == 0:
                progress.display(batch_idx)
            
        acc = correct / total
        # print('Clean Accuracy %.2f\n' % (acc*100))
        logger.info('Clean Accuracy %.2f\n' % (acc*100))


if args.eval:
    test(0)
else:

    for epoch in range(start_epoch, epochs):
        train(epoch)
        test(epoch)
            
        scheduler.step()
        if (epoch+1) % 5 == 0:

            state= {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            } 

            filename = str(state['epoch']) + '.pth.tar'
            torch.save(state, '{}/{}'.format(args.exp_path ,filename))
            print(' : save pth for epoch {}'.format(epoch + 1))



