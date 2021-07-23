# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:54:37 2020

@author: haoye
"""

from mxnet import nd
from mxnet.gluon import nn
import unit_split

def net(units, filter_lists, num_classes, **kwargs):
    #epsilon, momentum = kwargs.get('epsilon', 2e-5), kwargs.get('momentum', 0.9)
    epsilon, momentum = 2e-5, 0.9
    num_stages = 4
    num_unit   = len(units)
    assert(num_unit==len(units))
    net = nn.Sequential()
    net.add(
            nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1),
            #nn.Conv2D(channels=64, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(epsilon=epsilon, momentum=momentum),
            nn.Activation('relu')
            )
    #net = nn.Sequential()
    for i in range(num_stages):
        net.add(
                unit_split.Attention_Residual_Unit(filter_lists[i+1], strides=2, same_shape=False))
        for j in range(units[i]-1):
            net.add(
                    unit_split.Attention_Residual_Unit(filter_lists[i+1], strides=1, same_shape=True))
    net.add(
            nn.BatchNorm(momentum=momentum),
            nn.Dropout(0.4),
            nn.GlobalAvgPool2D(),
            nn.Dense(num_classes),
            nn.BatchNorm(momentum=momentum))
    return net


def get_net(num_classes, num_layers, **kwargs):

    filter_list = [64, 64, 128, 256, 512]
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return net(units       = units,
               filter_lists= filter_list,
               num_classes = num_classes,
               **kwargs)
