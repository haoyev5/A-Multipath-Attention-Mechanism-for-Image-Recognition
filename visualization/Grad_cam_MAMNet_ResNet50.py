# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:17:07 2021

@author: haoye
"""

import sys
import os
#sys.path.append(os.getcwd())
os.system('dir')
os.system('ls')
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils,nn
import d2lzh as d2l
sys.path.append(os.getcwd())
import get_net_split
import matplotlib.pyplot as plt
import cv2
import numpy as np


#filename = "img/n03180011_2716.jpeg"
filename = "img/4.jpg"

input_image = cv2.imread(filename)
img = cv2.imread(filename)

#ctx = d2l.try_gpu()

filename = 'res50_48_3.params'
pretrain_net = get_net_split.get_net(num_classes=1000, num_layers=50)
pretrain_net.load_parameters(filename, allow_missing=True)

def image_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (112, 112))
    tensor = nd.array(img)

    rgb_mean = nd.array([0.485, 0.456, 0.406])
    rgb_std = nd.array([0.229, 0.224, 0.225])
    tensor = (tensor.astype('float32') / 255 - rgb_mean) / rgb_std
    tensor = nd.transpose(tensor, [2, 0, 1])
    tensor = nd.expand_dims(tensor, 0)
    return tensor


############################################################################
tensor = image_to_tensor(input_image)
gailv = pretrain_net(tensor)
gailv = nd.softmax(gailv)
index = gailv.argmax(axis=1)
print("class No:", index)
print("Predictive value:",gailv[0][index])
############################################################################
    
m1 = nn.Sequential()
m2 = nn.Sequential()
for i, layer in enumerate(pretrain_net):
    if i in [27, 28, 31]:
        pass #注意不要dropout和BN层, 不然打开自动求导后模型进入训练模式dropout也会被打开
    elif i > 28:
        m2.add(layer)
    else:
        m1.add(layer)

############################################################################
    
tensor = image_to_tensor(input_image)
#第一个模型不用记录梯度
conv_layer_output = m1(tensor)
conv_layer_output.attach_grad()

with autograd.record():
    preds = m2(conv_layer_output)
    num = preds.argmax(axis=1)
    loss = preds[0, num]  
    
loss.backward()

############################################################################  


        
conv_layer_output_value = conv_layer_output[0] #除掉批次维度
#梯度平均值作为重要程度的权重
pooled_grads_value = conv_layer_output.grad.mean(axis=(0,2,3))
for i in range(512):
    #注意通道维度的位置
    conv_layer_output_value[i, :, :] *= pooled_grads_value[i]
#同样注意通道维度的位置
conv_layer_output_value = nd.relu(conv_layer_output_value)
cam = np.mean(conv_layer_output_value.asnumpy(), axis=0)

heatmap = np.maximum(cam, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
#cv2.imwrite("a.png",heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


#heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)

fig = plt.figure()
plt.imshow(heatmap[:, :, ::-1])
#plt.imshow(heatmap)
plt.axis('off')
cv2.imwrite('img/mask.png', heatmap)

superimposed_img = heatmap * 0.4 + img

superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
print(superimposed_img.shape)
fig = plt.figure()
plt.imshow(superimposed_img[:, :, ::-1]) #显示要转rgb
plt.axis('off')

cv2.imwrite('img/heat_map.png', superimposed_img)

'''
# If you want to display the attention activation map, please uncomment
############################################################################
for i, layer in enumerate(pretrain_net):
    if i < 4:
        pass
    #elif i >26:
    elif i > 8:
        pass
    else:
        SA = pretrain_net[i].att.SA
        SA = nd.squeeze(SA)
        SA = SA.asnumpy()
        SA = np.uint8(255 * SA)
        mask = cv2.applyColorMap(SA, cv2.COLORMAP_JET)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        #mask = cv2.applyColorMap(SA, cv2.COLORMAP_BONE)
        fig = plt.figure()
        #plt.imshow(mask[:, :, ::-1], cmap=plt.cm.gray)
        plt.imshow(mask)
        plt.axis('off')
        #cv2.imwrite('1.png',mask)
        #plt.imshow(mask)

'''

