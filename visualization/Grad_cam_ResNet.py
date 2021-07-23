# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:41:37 2021

@author: haoye
"""


from mxnet import gluon
import mxnet.gluon.nn as nn
from mxnet import autograd, nd
import cv2
import matplotlib.pyplot as plt
import numpy as np

#net = gluon.model_zoo.vision.resnet50_v2(pretrained=True)
net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
#查看resnet50_v2结构, gluon中把Activation也作为了单独的一层
#print(net.features)

#filename = "img/n03180011_2716.jpeg"
filename = "img/4.jpg"

input_image = cv2.imread(filename)
img = cv2.imread(filename)


def image_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    tensor = nd.array(img)

    rgb_mean = nd.array([0.485, 0.456, 0.406])
    rgb_std = nd.array([0.229, 0.224, 0.225])
    tensor = (tensor.astype('float32') / 255 - rgb_mean) / rgb_std
    tensor = nd.transpose(tensor, [2, 0, 1])
    tensor = nd.expand_dims(tensor, 0)
    return tensor


############################################################################
m1 = nn.Sequential()
m2 = nn.Sequential()
for i, layer in enumerate(net.features):
    if i in [9, 12]:
        pass #注意不要dropout层, 不然打开自动求导后模型进入训练模式dropout也会被打开
    elif i > 8:
        m2.add(layer)
    else:
        m1.add(layer)
m2.add(net.output)



tensor = image_to_tensor(input_image)
gailv = net(tensor)
gailv = nd.softmax(gailv)
index = gailv.argmax(axis=1)
print("class No:", index)
print("Predictive value:",gailv[0][index])


#第一个模型不用记录梯度
conv_layer_output = m1(tensor)
conv_layer_output.attach_grad()
'''
with autograd.record():
    preds = m2(conv_layer_output)
    loss = preds[0, 285]
'''    
with autograd.record():
    preds = m2(conv_layer_output)
    num = preds.argmax(axis=1)
    #print(preds[num])
    loss = preds[0, num]    
    
loss.backward()

conv_layer_output_value = conv_layer_output[0] #除掉批次维度
#梯度平均值作为重要程度的权重
pooled_grads_value = conv_layer_output.grad.mean(axis=(0,2,3))
for i in range(512):
    #注意通道维度的位置
    conv_layer_output_value[i, :, :] *= pooled_grads_value[i]
#同样注意通道维度的位置
heatmap = np.mean(conv_layer_output_value.asnumpy(), axis=0)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
#plt.matshow(heatmap)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
fig = plt.figure()
plt.imshow(heatmap[:, :, ::-1])
plt.axis('off')
cv2.imwrite('img/mask.png', heatmap)

superimposed_img = heatmap * 0.3 + img

superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

fig = plt.figure()
plt.imshow(superimposed_img[:, :, ::-1]) #显示要转rgb
plt.axis('off')
cv2.imwrite('img/heat_map.png', superimposed_img)






