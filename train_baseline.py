#!/usr/bin/env python
import torchattacks
import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchattacks
from PIL import Image, ImageFilter
import utils
import utils
# import models.builer as builder
# import dataloader
# import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import warnings
import torch.utils.data as data
from torch.utils.data import Subset
import numpy as np
warnings.filterwarnings("ignore")
def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser()

   
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--parallel', type=int, default=0, 
                        help='1 for parallel, 0 for non-parallel')

    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg16-bn', 'densenet-121', 'wrn-34-10','resnet18-jump','simplecnn','resnet-split'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100','imagenet'])
    parser.add_argument('--train_type', type=str, default='adv', choices=['erm', 'adv'], help='ERM or Adversarial training loss')
    parser.add_argument('--eps', type=float, default=0.015)
    parser.add_argument('--pgd_steps', type=int, default=10)
    parser.add_argument('--pgd_norm', type=str, default='linf', choices=['linf', 'l2', ''])

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mix', type=float, default=1.0) # percent of poisoned data, default 100% data is poisoned.
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--layer', type=str,default = '')
    parser.add_argument('--poison',type=str, default='CUDA')

    parser.add_argument('--grayscale', default=True, type=bool, help='grayscale compression')
    parser.add_argument('--jpeg', default=10, type=int, help='JPEG quality factor')
    parser.add_argument('--output_dir', default='test', type=str)
    parser.add_argument('--num_cls', default=100, type=int)
    parser.add_argument('--ae', default=False, type=bool)
    parser.add_argument('--resume_ckpt',type=str)


    args = parser.parse_args()

    return args


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


def load_dict(resume_path, model):
    if os.path.isfile(resume_path):

        loc = 'cuda:{}'.format('0')
        checkpoint = torch.load(resume_path, map_location=loc)
        # checkpoint = torch.load(resume_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model
    
class ImageDataset(data.Dataset):
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

        with torch.no_grad():

            kernel_size = random.sample([7,9,11],1)[0]
            blur_para = 0.01* random.randint(4,8)
   
            kernel = np.random.uniform(low=0, high=blur_para, size=(1, 1, kernel_size, kernel_size))
            
            kernel[0, 0, np.random.randint(kernel.shape[2]), np.random.randint(kernel.shape[3])] = 1
            kernel = np.repeat(kernel, 3, axis=0)
            sample= torch.nn.functional.conv2d(sample, torch.from_numpy(kernel).float(), stride=1, groups=3, padding='same')
            sample /= sample.max()

        if self.patch_noise: 
            # random.seed(index)
            patch_size=16
            p_n = 224//patch_size
            noise = np.random.uniform(0.2, 0.8, (3, p_n , p_n))*(2*np.random.randint(2, size=(3, p_n , p_n))-1)*0.1
            noise = np.repeat(noise, patch_size, 1) 
            noise = np.repeat(noise, patch_size, 2) 
            noise = torch.tensor(noise)
            sample += noise
            sample = torch.clip(sample,0,1)


        return sample, label

    def __len__(self):
        return len(self.im_names)

def train_loader(data_path, augtype, patch_noise):

    # [NO] do not use normalize here cause it's very hard to converge
    # [NO] do not use colorjitter cause it lead to performance drop in both train set and val set

    # [?] guassian blur will lead to a significantly drop in train loss while val loss remain the same
    if augtype==0:
        augmentation = [
            transforms.RandomResizedCrop(224),    
            transforms.RandomHorizontalFlip(),               
            transforms.ToTensor(),    
        ]
    elif augtype==1:
            augmentation = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]
    elif augtype==2:
        augmentation = [
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.Grayscale(3),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
    else:
        augmentation = [
            transforms.Resize(256),                   
            transforms.CenterCrop(224),
            transforms.ToTensor(),    
        ]
    


    train_trans = transforms.Compose(augmentation)
    train_dataset = ImageDataset(data_path, transform=train_trans,patch_noise =patch_noise)  
    
     
    if len(train_dataset)>30000:
        # Define the desired subset size
        subset_size = len(train_dataset)//5

        # Create a subset of the CIFAR10 dataset
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
        subset_dataset = Subset(train_dataset, subset_indices)
    else:
        subset_dataset = train_dataset
    
    # subset_dataset = train_dataset



    train_loader = torch.utils.data.DataLoader(
        subset_dataset, batch_size=args.batch_size, shuffle=True,
    pin_memory=True, sampler=None)


    return train_loader


def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    ngpus_per_node = torch.cuda.device_count()
    print('=> ngpus : {}'.format(ngpus_per_node))

    if args.parallel == 1: 
        # single machine multi card       
        args.gpus = ngpus_per_node
        args.nodes = 1
        args.nr = 0
        args.world_size = args.gpus * args.nodes

        args.workers = int(args.workers / args.world_size)
        args.batch_size = int(args.batch_size / args.world_size)
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        args.world_size = 1
        main_worker(ngpus_per_node, args)
  
def main_worker(gpu, args):
    utils.init_seeds(1 + gpu, cuda_deterministic=False)
    if args.parallel == 1:
        args.gpu = gpu
        args.rank = args.nr * args.gpus + args.gpu

        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)  
           
    else:
        # two dummy variable, not real
        args.rank = 0
        args.gpus = args.gpuid
        torch.cuda.set_device(args.gpus)
    if args.rank == 0:
        print('=> modeling the network {} ...'.format(args.arch))
    args.mask= False
    


    import resnet

    configs, bottleneck = resnet.get_configs(args.arch)
    model = resnet.ResNet(configs, bottleneck, num_classes=args.num_cls)
    model = model.cuda()  

    autoencoder = resnet.ResNetAutoEncoder(configs, bottleneck,num_cls=100)
    load_dict('clean200_20_fix.pth', autoencoder)
    autoencoder.eval()
    autoencoder.cuda()


    optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.resume_ckpt:
        if os.path.isfile(args.resume_ckpt):
            print("=> loading checkpoint '{}'".format(args.resume_ckpt))
            if args.gpuid is None:
                checkpoint = torch.load(args.resume_ckpt)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpuid)
                checkpoint = torch.load(args.resume_ckpt, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            try:
                model.load_state_dict(checkpoint['state_dict'])
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
                load_my_state_dict(model,state_dict)
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_ckpt))

       
    # if args.rank == 0:       
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    
    # if args.rank == 0:
    #     print('=> building the oprimizer ...')
   
    # if args.rank == 0:
    #     print('=> building the dataloader ...')


    patch_noise=False
    if args.poison =='clean':
        aug_type=1
    # elif 'LSP' in args.poison:
    #     aug_type=2
    #     patch_noise = True
    else:
        aug_type=1


    unlearnable_loader = train_loader(data_path='/home2/huangyi/ImageShortcutSqueezing/list/{}_list.txt'.format(args.poison), augtype= aug_type, patch_noise = patch_noise)
    clean_test_loader =  train_loader(data_path='/home2/huangyi/ImageShortcutSqueezing/list/IN100val_list.txt', augtype = 4, patch_noise = False)

    
    # if args.rank == 0:
    #     print('=> building the criterion ...')
    criterion = nn.CrossEntropyLoss()

    global iters
    iters = 0
    


    # if args.rank == 0:
        # print('=> starting training engine ...')
    for epoch in range(args.start_epoch, args.epochs):
        
        # global current_lr
        # current_lr = utils.adjust_learning_rate_cosine(optimizer, epoch, args)
        
        # train for one epoch
        model.train()
     
        do_train(unlearnable_loader, model,autoencoder,criterion, optimizer, epoch, args)
        
        do_eval(clean_test_loader, model,autoencoder, epoch, args)

        scheduler.step()
        # save pth
        if (epoch+1) % 5 == 0 :

            state= {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' :scheduler.state_dict(),
           } 
            args.output_dir = os.path.join('baseline','{}_{}_{}_{}'.format(args.poison,args.train_type,args.eps,args.pgd_steps))
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            filename = str(state['epoch']) + '.pth.tar'
            torch.save(state, '{}/{}'.format(args.output_dir ,filename))
            print(' : save pth for epoch {}'.format(epoch + 1))





def do_train(train_loader, model,autoencoder, criterion, optimizer, epoch, args):
   
   
    losses_ce = utils.AverageMeter('CELoss', ':.4f')
    learning_rate = utils.AverageMeter('LR', ':.4f')
    top1 = utils.AverageMeter('Top1', ':.4f')
    top5 = utils.AverageMeter('Top5', ':.4f')
    
    progress = utils.ProgressMeter(
        len(train_loader),
        [  losses_ce,top1,top5],
        prefix="Epoch:[{}]".format(epoch+1))
   
    # update lr
    # learning_rate.update(current_lr)



    attacker = torchattacks.PGD
    eps = args.eps
    steps = args.pgd_steps
    # print(steps)
    atk = attacker(model, eps=eps, alpha=eps/steps * 1.5, steps=steps)




    for i, (input, label) in enumerate(train_loader):
        # measure data loading time
        
        global iters
        iters += 1
         
        input = input.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        if args.ae==True:
            input = autoencoder(input)
        if args.train_type=='adv':

            input = atk(input, label)
       

        logit = model(input)
        loss = criterion(logit, label)
     
        acc1, acc5 = accuracy(logit, label, topk=(1, 5))
        top1.update(acc1[0], input.size(0))    
        top5.update(acc5[0], input.size(0))

        # compute gradient and do solver step
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weights
        optimizer.step()
        # syn for logging
        torch.cuda.synchronize()

        # record loss
        losses_ce.update(loss.item(), input.size(0)) 

        if i % args.print_freq == 0:
            progress.display(i)
        
        

def do_eval(val_loader, model ,autoencoder,epoch, args):
       
    
    correct, total = 0, 0
    top1 = utils.AverageMeter('Top1', ':.4f')
    top5 = utils.AverageMeter('Top5', ':.4f')
    
    progress = utils.ProgressMeter(
        len(val_loader),
        [ top1, top5],
        prefix="Epoch: [{}]".format(epoch+1))

    model.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):            
            input = input.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            
            if args.ae==True:
                input = autoencoder(input)
            

            logit = model(input)
            acc1, acc5 = accuracy(logit, label, topk=(1, 5))     
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            _ , predicted = torch.max(logit.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            

            if i % args.print_freq == 0 :
                progress.display(i)
                
        acc = correct / total
        print('Clean Accuracy %.2f\n' % (acc*100))


if __name__ == '__main__':

    args = get_args()

    main(args)


