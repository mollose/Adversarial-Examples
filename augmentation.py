# -*- coding: utf-8 -*-

import torch
import warnings
from torch.autograd import Variable
from random import randint

warnings.filterwarnings("ignore")

def get_jac_row(model, x, batch_size, output_dims, index_for_select):
    """
    :param model: A network instance (new add parameter)
    :param x: input
    :param batch_size: the size of a batch. [the size of a batch]
    :param output_dims: the dimensions of output. [the number of types]
    :param index_for_select: select rows index_for_select from jacobian matrix in autograd
    :return: batch_size × selected_jac_row
    """
    ## output_dim x input_dim
    repeat_x = x.detach()
    repeat_x.requires_grad_()
    jac_y = model(repeat_x)

    gradient = torch.zeros(batch_size + output_dims)
    gradient.__setitem__((torch.arange(batch_size[0]), index_for_select), 1)
    jac_y.backward(gradient)
    jac_row = repeat_x.grad.data
    return jac_row

def augmentation(model, x, t, aug_batch_size, lambda_):
    """
    :param model:  A network instance (new add parameter)
    :param x: input
    :param aug_batch_size: 
    :param lambda_: A hyperparameter
    :return: synthetic data
    """
    batch_size = [aug_batch_size]
    output_dims = [model.num_classes]
    index_for_select = t.long()
    jac_x = get_jac_row(model, x, batch_size, output_dims, index_for_select)
    
    syn = x + lambda_ * torch.sign(jac_x)
    return syn

# Reservoir Sampling 사용하는 augmentation
def reservoir_augmentation(model, aed, aug_batch_size, lambda_) : 
    """
    :param model:  A network instance (new add parameter)
    :param aed: 
    :param aug_batch_size: 
    :param lambda_: A hyperparameter
    :return: synthetic data
    """
    reservoir_loader = aed.reservoir_load(aug_batch_size)
    syn_data = torch.Tensor() 
    
    for data in reservoir_loader :
        x_aug, t_aug = data
        x_aug = Variable(x_aug)
        
        syn = augmentation(model, x_aug, t_aug, aug_batch_size, lambda_)
        syn_data = torch.cat((syn_data, syn), dim=0)
    
    k = syn_data.shape[0]
    
    for idx in range(k, len(aed)) :
        r = randint(1, idx)
        
        if r < k :
            x = aed[idx][0].unsqueeze(0)
            jac_x = get_jac_row(model, x, [1], [model.num_classes], aed[idx][1].long())
            syn_data[r] = x + lambda_ * torch.sign(jac_x)
            
    return syn_data


