import torch
import torch.nn as nn
import torch.nn.init as init


def adjust_learning_rate(lr, optimizer, gamma, step):
    lr_ = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
