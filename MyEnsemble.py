from __future__ import print_function

import argparse
import os
import random
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

from CifarImbalanced import CIFAR10, CIFAR100
from FocalLossV5 import FocalLoss
from Evaluation import acc

from EnsembleV1 import resnext29_16_64, resnext47_16_64, resnext56_16_64

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train(model, trainLoader, criterion, optimizer, use_cuda=True, weights_for_classes=None):
    model.train()
    runningLoss = 0.0
    runningCorrects = 0
    runningCorrectsEnsembleUnweighted = 0.0

    for inputs, labels in trainLoader:
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.set_grad_enabled(True):
            outputs, out1x1, out2x1,out3x1,out4x1,out5x1 = model(x=inputs)
            acc0 = acc(outputs, labels)
            acc1 = acc(out1x1, labels)
            acc2 = acc(out2x1, labels)
            acc3 = acc(out3x1, labels)
            acc4 = acc(out4x1, labels)
            acc5 = acc(out5x1, labels)

            w_for_classifier_sum = acc0+acc1+acc2+acc3+acc4+acc5
            w_for_classifier0 = acc0/w_for_classifier_sum
            w_for_classifier1 = acc1/w_for_classifier_sum
            w_for_classifier2 = acc2/w_for_classifier_sum
            w_for_classifier3 = acc3/w_for_classifier_sum
            w_for_classifier4 = acc4/w_for_classifier_sum
            w_for_classifier5 = acc5/w_for_classifier_sum

            w_for_instances = torch.zeros(outputs.size(0), 1)
            if use_cuda:
                w_for_instances = w_for_instances.cuda()
            #    weights_for_classes = weights_for_classes.cuda()

            weights_for_classes = None
            
            loss1, w_for_instances = criterion(out1x1, labels, w_for_instances, weight=weights_for_classes)
            loss2, w_for_instances = criterion(out2x1, labels, w_for_instances, weight=weights_for_classes)
            loss3, w_for_instances = criterion(out3x1, labels, w_for_instances, weight=weights_for_classes)
            loss4, w_for_instances = criterion(out4x1, labels, w_for_instances, weight=weights_for_classes)
            loss5, w_for_instances = criterion(out5x1, labels, w_for_instances, weight=weights_for_classes)

            loss0, w_for_instances = criterion(outputs, labels, w_for_instances, weight=weights_for_classes)
            
            #loss = loss0+0.3*loss1+0.3*loss2+0.3*loss3+0.3*loss4+0.3*loss5
            loss = loss0+loss1+loss2+loss3+loss4+loss5


            _, preds = torch.max(outputs, 1)
            _, preds1 = torch.max(out1x1, 1)
            _, preds2 = torch.max(out2x1, 1)
            _, preds3 = torch.max(out3x1, 1)
            _, preds4 = torch.max(out4x1, 1)
            _, preds5 = torch.max(out5x1, 1)
            
            outputsEnsemble_unweighted = torch.zeros(outputs.size())

            for i in range(inputs.size(0)):
                outputsEnsemble_unweighted[i, preds[i]] = outputsEnsemble_unweighted[i, preds[i]] + 1
                outputsEnsemble_unweighted[i, preds1[i]] = outputsEnsemble_unweighted[i, preds1[i]] + 1
                outputsEnsemble_unweighted[i, preds2[i]] = outputsEnsemble_unweighted[i, preds2[i]] + 1
                outputsEnsemble_unweighted[i, preds3[i]] = outputsEnsemble_unweighted[i, preds3[i]] + 1
                outputsEnsemble_unweighted[i, preds4[i]] = outputsEnsemble_unweighted[i, preds4[i]] + 1
                outputsEnsemble_unweighted[i, preds5[i]] = outputsEnsemble_unweighted[i, preds5[i]] + 1
            
            _, predsEnsembleUnweighted = torch.max(outputsEnsemble_unweighted, 1)
            
            if use_cuda:
                runningCorrectsEnsembleUnweighted = runningCorrectsEnsembleUnweighted+ torch.sum(predsEnsembleUnweighted.cuda()==labels.data).cpu()
            else:
                runningCorrectsEnsembleUnweighted = runningCorrectsEnsembleUnweighted+ torch.sum(predsEnsembleUnweighted==labels.data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            
        runningLoss += loss.item() * inputs.size(0)
        runningCorrects = runningCorrects + torch.sum(preds == labels.data)

        
        overallLoss = [w_for_classifier0,w_for_classifier1,w_for_classifier2,w_for_classifier3,w_for_classifier4,w_for_classifier5]
        if use_cuda:
            model.module.setWeights(overallLoss)
        else:
            model.setWeights(overallLoss)
    
    epochLoss = runningLoss / len(trainLoader.dataset)
    epochAcc = runningCorrects.double() / len(trainLoader.dataset)
    epochAccEnsembleUnweighted = runningCorrectsEnsembleUnweighted.double() / len(trainLoader.dataset)
    print('Train Loss: {:.4f} Acc:{:.4f} EnsembleAcc:{:.4f}'.format(epochLoss, epochAcc, epochAccEnsembleUnweighted))
    return (epochLoss, epochAcc, epochAccEnsembleUnweighted)

def validate(model, testLoader, criterion, optimizer, use_cuda=True):
    #print('test on validation')
    model.eval()
    runningLoss = 0.0
    runningCorrects = 0.0
   
    runningCorrectsEnsemble=0.0
    runningCorrectsEnsembleUnweighted = 0.0
    runningCorrectsEnsemble_plus = 0.0
    runningCorrectsEnsembleUnweighted_plus = 0.0
    if use_cuda:
        w = model.module.getWeights()
    else:
        w = model.getWeights()
    #w = [1,1,1,1,1,1]
    for inputs, labels in testLoader:
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        with torch.set_grad_enabled(False):
            optimizer.zero_grad()
            outputs,out1x1,out2x1,out3x1,out4x1,out5x1 = model(x=inputs)
            _, preds = torch.max(outputs, 1)
            _, preds1 = torch.max(out1x1, 1)
            _, preds2 = torch.max(out2x1, 1)
            _, preds3 = torch.max(out3x1, 1)
            _, preds4 = torch.max(out4x1, 1)
            _, preds5 = torch.max(out5x1, 1)
            
            outputsEnsemble = torch.zeros(outputs.size())
            outputsEnsemble_unweighted = torch.zeros(outputs.size())

            outputEnsemble_plus = outputs*w[0]+out1x1*w[1]+out2x1*w[2]+out3x1*w[3]+out4x1*w[4]+out5x1*w[5]
            outputEnsemble_unweighted_plus = outputs+out1x1+out2x1+out3x1+out4x1+out5x1

            for i in range(inputs.size(0)):
                outputsEnsemble[i, preds[i]] = outputsEnsemble[i, preds[i]] + w[0]
                outputsEnsemble[i, preds1[i]] = outputsEnsemble[i, preds1[i]] + w[1]
                outputsEnsemble[i, preds2[i]] = outputsEnsemble[i, preds2[i]] + w[2]
                outputsEnsemble[i, preds3[i]] = outputsEnsemble[i, preds3[i]] + w[3]
                outputsEnsemble[i, preds4[i]] = outputsEnsemble[i, preds4[i]] + w[4]
                outputsEnsemble[i, preds5[i]] = outputsEnsemble[i, preds5[i]] + w[5]

                outputsEnsemble_unweighted[i, preds[i]] = outputsEnsemble_unweighted[i, preds[i]] + 1
                outputsEnsemble_unweighted[i, preds1[i]] = outputsEnsemble_unweighted[i, preds1[i]] + 1
                outputsEnsemble_unweighted[i, preds2[i]] = outputsEnsemble_unweighted[i, preds2[i]] + 1
                outputsEnsemble_unweighted[i, preds3[i]] = outputsEnsemble_unweighted[i, preds3[i]] + 1
                outputsEnsemble_unweighted[i, preds4[i]] = outputsEnsemble_unweighted[i, preds4[i]] + 1
                outputsEnsemble_unweighted[i, preds5[i]] = outputsEnsemble_unweighted[i, preds5[i]] + 1
            
            
            _, preds = torch.max(outputs, 1)
            _, predsEnsemble = torch.max(outputsEnsemble, 1)
            _, predsEnsembleUnweighted = torch.max(outputsEnsemble_unweighted, 1)
            _, predsEnsemble_plus = torch.max(outputEnsemble_plus, 1)
            _, predsEnsembleUnweighted_plus = torch.max(outputEnsemble_unweighted_plus, 1)
            
            if use_cuda:
                runningCorrects = runningCorrects + torch.sum(preds==labels.data).cpu()
                runningCorrectsEnsemble = runningCorrectsEnsemble + torch.sum(predsEnsemble.cuda()==labels.data).cpu()
                runningCorrectsEnsembleUnweighted = runningCorrectsEnsembleUnweighted+ torch.sum(predsEnsembleUnweighted.cuda()==labels.data).cpu()
                runningCorrectsEnsemble_plus = runningCorrectsEnsemble_plus + torch.sum(predsEnsemble_plus.cuda()==labels.data).cpu()
                runningCorrectsEnsembleUnweighted_plus = runningCorrectsEnsembleUnweighted_plus+ torch.sum(predsEnsembleUnweighted_plus.cuda()==labels.data).cpu()
            else:
                runningCorrects = runningCorrects + torch.sum(preds==labels.data)
                runningCorrectsEnsemble = runningCorrectsEnsemble + torch.sum(predsEnsemble==labels.data)
                runningCorrectsEnsembleUnweighted = runningCorrectsEnsembleUnweighted+ torch.sum(predsEnsembleUnweighted==labels.data)
                runningCorrectsEnsemble_plus = runningCorrectsEnsemble_plus + torch.sum(predsEnsemble_plus==labels.data)
                runningCorrectsEnsembleUnweighted_plus = runningCorrectsEnsembleUnweighted_plus+ torch.sum(predsEnsembleUnweighted_plus==labels.data)
            
    
    epochAcc = runningCorrects.double() / len(testLoader.dataset)
    epochAccEnsemble = runningCorrectsEnsemble.double() / len(testLoader.dataset)
    epochAccEnsembleUnweighted = runningCorrectsEnsembleUnweighted.double() / len(testLoader.dataset)
    epochAccEnsemble_plus = runningCorrectsEnsemble_plus.double() / len(testLoader.dataset)
    epochAccEnsembleUnweighted_plus = runningCorrectsEnsembleUnweighted_plus.double() / len(testLoader.dataset)
    
    print('Acc:{:.4f}   EnsembleAcc:{:.4f}   EnsembleUnweightedAcc:{:.4f}   Ensemble_plus:{:.4f}  Ensemble_unweighted:{:.4f}'.format(epochAcc, epochAccEnsemble, epochAccEnsembleUnweighted, epochAccEnsemble_plus, epochAccEnsembleUnweighted_plus))
    
    return ( epochAcc, epochAccEnsembleUnweighted )

def save_checkpoint(state, is_best, checkpoint='checkpoint', acc=100., epoch=1):
    if os.path.exists(checkpoint)==False:
        os.makedirs(checkpoint)

    filename = "Epoch_"+str(epoch)+"_"+str(acc)+"_checkpoint.pth.tar"
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, lr, epoch, schedule):
    """Sets the learning rate to the initial LR decayed by 10 following schedule"""
    if epoch in schedule:
        lr *= 0.1
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_checkpoint(epoch, checkpoint='checkpoint'):
    if os.path.exists(checkpoint)==False:
        os.makedirs(checkpoint)
    
    filename = "Epoch_"+str(epoch)+"_checkpoint.pth.tar"
    filepath = os.path.join(checkpoint, filename)
    return filepath

def MyEnsemble(IR=10., Cifar10_used=True, step_imbalance=True, root="./Result", fine_tunning = 0):
    if Cifar10_used:
        num_class = 10
    else:
        num_class = 100
    
    learningRate = 0.05
    momentum = 0.9
    weight_decay = 1e-4
    batchSize = 100

    start_epoch = 0
    epochs = 100
    schedule = [30, 60, 90, 100]
    gamma = 2.0
    if step_imbalance:
        model_save_path = os.path.join(root, "step_imbalance", "cifar"+str(num_class), "IR="+str(IR), 'MyEnsemble')
    else:
        model_save_path = os.path.join(root, 'long_tail', "cifar"+str(num_class), "IR="+str(IR), str(gamma), 'MyEnsemble')
    
    best_prec1 = 0
    use_cuda = torch.cuda.is_available()
    # create model
    model = resnext47_16_64(num_classes=num_class)
    criterion = FocalLoss()
    
    if use_cuda:
        #model.cuda()
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    transformTrain = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    transformTest = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
    ])

    if Cifar10_used:
        trainSet = CIFAR10(root='./data', train=True, download=True, transform=transformTrain, step_imbalance=step_imbalance, IR=IR)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize,  shuffle=True, num_workers=4)
        testSet = CIFAR10(root='./data', train=False, download=True, transform=transformTest)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=True, num_workers=4)
        weights_for_classes = trainSet.weight_for_classes
    else:
        trainSet = CIFAR100(root='./data', train=True, download=True, transform=transformTrain, step_imbalance=step_imbalance, IR=IR)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize,  shuffle=True, num_workers=4)
        testSet = CIFAR100(root='./data', train=False, download=True, transform=transformTest)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=True, num_workers=4)
        weights_for_classes = trainSet.weight_for_classes
    if fine_tunning>0:
        resume = get_checkpoint(fine_tunning, model_save_path)
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume, checkpoint['epoch']))
            checkpoint = os.path.dirname(resume)
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    
    singleTrainAcc = []
    singleTestAcc = []
    ensembleTrainAcc = []
    ensembleTestAcc = []
    #training & evaluation
    for epoch in range(start_epoch, epochs):
        learningRate = adjust_learning_rate(optimizer, learningRate, epoch, schedule)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, epochs, learningRate))

        # train for one epoch
        train_loss, train_acc, ensembleAcc = train(model, trainLoader, criterion, optimizer, use_cuda=use_cuda, weights_for_classes=weights_for_classes)
        singleAcc, prec1 = validate(model, testLoader, criterion, optimizer, use_cuda=use_cuda)

        singleTrainAcc.append(train_acc.cpu().numpy())
        singleTestAcc.append(singleAcc.cpu().numpy())
        ensembleTrainAcc.append(ensembleAcc.cpu().numpy())
        ensembleTestAcc.append(prec1.cpu().numpy())

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if (epoch % 5)==0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=model_save_path, acc=prec1.cpu().numpy(), epoch=epoch+1)

    print('Best accuracy:')
    print(best_prec1)
    print(singleTrainAcc)
    print(singleTestAcc)
    print(ensembleTrainAcc)
    print(ensembleTestAcc)

    return best_prec1
    