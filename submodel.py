# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:40:22 2019

@author: qksl2
"""
import sys, os
import argparse

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets
from torch.autograd import Variable
from augmentation import reservoir_augmentation
import AeDataloader as maed
import resnet as rnet
import random, copy

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import skimage
import math

class SubNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SubNet, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3,32,2) #3x32x32->32x31x31
        self.pool = nn.MaxPool2d(2,2) #32x15x15 
        self.conv2 = nn.Conv2d(32,64,2) #64x14x14
        self.pool = nn.MaxPool2d(2,2) #64x7x7

        self.fc1 = nn.Linear(64*7*7, 200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,num_classes)
        
    def forward(self,x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilons=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X
    

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    return Variable(x, requires_grad=requires_grad, volatile=volatile)
  

def pred_batch(x, model):
    """
    batch prediction helper
    """
    y_pred = np.argmax(model(to_var(x)).data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)

def attack_over_test_data(model, adversary, param, loader_test, oracle=None):
    """
    Given target model computes accuracy on perturbed data
    """
    total_correct = 0
    total_samples = len(loader_test.dataset)
    
    for t, (X, y) in enumerate(loader_test):
        y_pred = pred_batch(X, model)
        X_adv = adversary.perturb(X.numpy(), y_pred)
        X_adv = torch.from_numpy(X_adv)

        if oracle is not None:
            y_pred_adv = pred_batch(X_adv, oracle)
        else:
            y_pred_adv = pred_batch(X_adv, model)
        
        total_correct += (y_pred_adv.numpy() == y.numpy()).sum()

    acc = total_correct/total_samples
    
    print('Got %d/%d correct (%.2f%%) on the perturbed data' 
        % (total_correct, total_samples, 100 * acc))

    return acc
       
def test(net, testloader):
   
    net.eval()
    correct = 0
    total = 0
    
    for images, labels in testloader:
        
         outputs = net(images)
    
         _, predicted = torch.max(outputs.data, 1)
    
         total += labels.size(0)
         correct += (predicted == labels).sum()
         
    acc = float(correct) / total
    
    print('Accuracy with Clean images: %.2f %%' % (100 * acc))
    
    return acc
    
def bbox_submodel(param, testloader, best_acc=None):
    
    net = SubNet()
    oracle = rnet.cifar_resnet20('cifar10')
    #oracle = rnet.cifar_resnet20()
    #oracle.load_state_dict(torch.load('cifar10-resnet20-30abc31d.pth'))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], momentum=0.9)
    
    aed = maed.AeDataloader('./CIFAR10', param['init_set_size'], param['k'])    
    
    outputs = oracle(aed.x)
    outputs = outputs.max(dim=1)[1]
    aed.label_save(outputs)
 
    for rho in range(param['data_aug']):  #augmentation 횟수만큼 반복    
    
        for epoch in range(param['nb_epochs']):
    
            trainloader = aed.train_load(param['train_batch_size'])
        
            for i, data in enumerate(trainloader, 0):
        
                inputs, labels = data

                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
         
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        print('loss: '+str(loss.data.numpy()))
        #submodel test dataset 정확도 평가            
        acc = test(net, testloader)
        
         # 모델이 최고 높은 정확도가 나왔을 때의 하이퍼파라미터들을 저장합니다.
        if best_acc is not None and acc > best_acc :
            print('Saving..')
            state = {
                'init_set_size': param['init_set_size'],
                'learning_rate': param['learning_rate'],
                'data_aug': rho + 1,
                'lambda' : param['lambda'],
                'lambda_alt' : param['lambda_alt'],
                'acc' : acc
            }
            
             # 기록한 하이퍼파라미터들을 저장합니다.
            if not os.path.isdir('checkpoint') :
                os.mkdir('checkpoint')
                
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        
        if rho < param['data_aug'] - 1: #Augmentation을 설정횟수만큼 실행

            print("Augmenting substitute training data.")           
            
            lambda_ = param['lambda'] * np.power(-1, int(rho / param['lambda_alt']))
            
            # augmentation() => reservoir_augmentation() 수정 
            syn_data = reservoir_augmentation(net, aed, param['aug_batch_size'], lambda_)
            aed.aug_save(syn_data)
            
        
            print("Labeling substitute training data.")
        
            label_loader = aed.label_load(param['label_batch_size'])
            labels = torch.Tensor().long()
        
            for x_label in label_loader :
            
                outputs = oracle(x_label)
                outputs = outputs.max(dim=1)[1]
                labels = torch.cat((labels, outputs), dim=0)
            
            aed.label_save(labels)
            
    
        torch.save(net.state_dict(), param['oracle_name']+'_sub.pkl')
    
    return best_acc
        
def visualize(newImg, image_adv, pre, y_pred_adv, j, x_pred_prob, x_adv_prob):  
    figure, ax = plt.subplots(1,2, figsize=(9,4))
    ax[0].imshow(newImg)
    ax[0].set_title('Clean Example', fontsize=20)
    
    ax[1].imshow(image_adv)
    ax[1].set_title('Adversarial Example', fontsize=20)
    
    ax[0].axis('off')
    ax[1].axis('off')
    
    ax[0].text(0.5,-0.18, "Prediction: {}\n Probability: {}".format(classes[pre[0]],  math.floor(x_pred_prob*100)/100), size=15, ha="center", 
         transform=ax[0].transAxes)
    ax[1].text(1.75,-0.25, "Prediction: {}\n epsilon: {}\nProbability: {}".format(classes[y_pred_adv[0]], j, math.floor(x_adv_prob*100)/100), size=15, ha="center", 
         transform=ax[0].transAxes)

    plt.show()

# Hyper-parameters 최적화
def random_search(param, testloader, try_num) :
   
    best_acc = 0
    
    for idx in range(try_num) :
        
        hparam = copy.deepcopy(param)
        hparam['init_set_size'] = random.randint(*hparam['init_set_size'])
        hparam['learning_rate'] = random.uniform(*hparam['learning_rate'])
        hparam['lambda'] = random.uniform(*hparam['lambda'])
        hparam['lambda_alt'] = random.randint(*hparam['lambda_alt'])
        
        print('Random Search Try #%d' % (idx + 1))
        print('init_set_size: %d, learning_rate: %.3f, lambda: %.3f, lambda_alt: %d'
              % (hparam['init_set_size'], hparam['learning_rate'], 
                 hparam['lambda'], hparam['lambda_alt']))
    
        best_acc = bbox_submodel(hparam, testloader, best_acc)
        print('Random Search Try #%d Accuracy: %.2f %%' % (idx + 1, 100 * best_acc))
        
    state = torch.load('./checkpoint/ckpt.pth')
    
    hparam['init_set_size'] = state['init_set_size']
    hparam['learning_rate'] = state['learning_rate']
    hparam['data_aug'] = state['data_aug']
    hparam['lambda'] = state['lambda']
    hparam['lambda_alt'] = state['lambda_alt']
    
    bbox_submodel(hparam, testloader) # oracle_sub.pkl 저장을 위해 재호출
    print('Random Search Final Accuracy: %.2f %%' % (100 * state['acc']))
          
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='PROG')    
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    T_parser = subparsers.add_parser('T')
    T_parser.add_argument('-i', '--init_set_size', type=int, nargs=2)
    T_parser.add_argument('-a', '--data_aug', type=int)
    T_parser.add_argument('-l', '--lambda_', type=float, nargs=2)
    T_parser.add_argument('-n', '--try_num', type=int)
    
    A_parser = subparsers.add_parser('A')
    A_parser.add_argument('-e', '--epsilon', type=float)
    A_parser.add_argument('-p', '--image_path', required=True)
    
    args = parser.parse_args(sys.argv[1:])
    
    # Hyper-parameters, 임의로 설정해놨습니다
    param = {
            
        'label_batch_size' : 200,

        'train_batch_size': 100,

        'test_batch_size': 100,
        
        'aug_batch_size' : 100,
        
        'init_set_size' : [1000, 1500],

        'nb_epochs': 10,

        'learning_rate': [0.005, 0.02],

        'data_aug': 5, 

        'oracle_name': 'resnet20', 

        'epsilon': 0.05, #0.02=>41.49, 0.03=>40.79, 0.1=>29.66 , 0.13=>25.66, 0.07=>35.21
        
        'lambda' : [0.05, 0.2], #0.05, 0.2
        
        'lambda_alt' : [1, 5], # lambda 부호를 바꿔주는 epoch 주기 
        
        'k' : 400 # augmentation 시 증가하는 데이터 수, 반드시 k < init_set_size
    }    
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    testset = datasets.CIFAR10('./CIFAR10', train=False, download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
   
    if args.mode == 'T' :
    
        try_num = 5
        
        if args.init_set_size is not None :
            param['init_set_size'] = args.init_set_size
            
        if args.data_aug is not None :
            param['data_aug'] = args.data_aug
            param['lambda_alt'][1] = args.data_aug
            
        if args.lambda_ is not None : 
            param['lambda'] = args.lambda_
            
        if args.try_num is not None : 
            try_num = args.try_num
        
        random_search(param, testloader, try_num)
        
    elif args.mode == 'A' :
        
        net = SubNet()
        oracle = rnet.cifar_resnet20('cifar10')
    
        net.load_state_dict(torch.load(param['oracle_name']+'_sub.pkl'))
    
        
        for p in net.parameters():
    
            p.requires_grad = False
            
            
        net.eval()
    
        oracle.eval()
    
        #############subnet 정확도 평가##############
        print('For the substitute model:')
        test(net, testloader)
       ################# Setup oracle#####################
        print('For the oracle_'+param['oracle_name'])
        test(oracle, testloader)
    
    #    print('agaist blackbox FGSM attacks using gradients from the substitute:')
           
        # Setup adversarial attacks
        #oracle에 adversarial example 적용
    #    adversary = FGSMAttack(net, param['epsilon'])
        
    #    attack_over_test_data(net, adversary, param, testloader, oracle)
        
        ###############결과사진#################
    
        
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        #d6, deer4
        image_path = args.image_path # "image/d6.jpg"
        src = mpimg.imread(image_path)
        plt.imshow(src) 
        plt.title('Original Image')
        plt.axis('off') 
        plt.show()
        
        
        newImg = skimage.transform.resize(src,(32,32))
        
        imagebatch = newImg.reshape(-1,3,32,32)
        image_tensor = torch.from_numpy(imagebatch).float()
        oracle.eval()
        output = oracle(image_tensor)
        x_pred_prob = F.softmax(output.data, dim=1)
        x_pred_prob = (torch.max(x_pred_prob.data, 1)[0][0])*100 #Original image 예측 확률
        _, predicted = torch.max(output.data,1)
        pre = predicted.cpu().numpy()
        
        
        if args.epsilon is not None : 
            
            epsilon = args.epsilon
            
            adversary = FGSMAttack(net, epsilon)
            image_adv = adversary.perturb(image_tensor.numpy(), pre)
            image_adv = torch.from_numpy(image_adv)
            output_adv = oracle(image_adv)
            x_adv_prob = F.softmax(output_adv.data, dim=1)
            x_adv_prob = (torch.max(x_adv_prob.data, 1)[0][0]) * 100 #adversarial image 예측 확률
            y_pred_adv = pred_batch(image_adv, oracle)
            image_adv = image_adv.reshape(-1,32,32,3)
            image_adv = torch.squeeze(image_adv)
            visualize(newImg, image_adv, pre, y_pred_adv, epsilon, x_pred_prob, x_adv_prob)
            #sub모델 adversarial example 정확도
            print('For the substitute model:')
            attack_over_test_data(net, adversary, param, testloader)
            #oracle adversarial example 정확도
            print('For the oracle_'+param['oracle_name']+':')
            acc = 100*attack_over_test_data(net, adversary, param, testloader, oracle)
            print('Epsilon: %.2f, Oracle Accuracy: %.2f %%'% (epsilon, acc))
        
        else :
        
            epsilon = [0.025, 0.04, 0.06, 0.12]
            
            for i, j in enumerate(epsilon):
                adversary = FGSMAttack(net, j)
                image_adv = adversary.perturb(image_tensor.numpy(), pre)
                image_adv = torch.from_numpy(image_adv)
                output_adv = oracle(image_adv)
                x_adv_prob = F.softmax(output_adv.data, dim=1)
                x_adv_prob = (torch.max(x_adv_prob.data, 1)[0][0]) * 100 #adversarial image 예측 확률
                y_pred_adv = pred_batch(image_adv, oracle)
                image_adv = image_adv.reshape(-1,32,32,3)
                image_adv = torch.squeeze(image_adv)
                visualize(newImg, image_adv, pre, y_pred_adv, j, x_pred_prob, x_adv_prob)
                #sub모델 adversarial example 정확도
                print('For the substitute model:')
                attack_over_test_data(net, adversary, param, testloader)
                #oracle adversarial example 정확도
                print('For the oracle_'+param['oracle_name']+':')
                acc = 100*attack_over_test_data(net, adversary, param, testloader, oracle)
                print('Epsilon: %.2f, Oracle Accuracy: %.2f %%'% (j, acc))        
    
    #    print(classes[y_pred_adv[0]])

