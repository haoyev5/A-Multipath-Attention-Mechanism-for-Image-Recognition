# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:17:57 2020

@author: haoye
"""
import sys
import os
sys.path.append(os.getcwd())
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils, nn
#from mxnet.gluon import model_zoo
from gluoncv import model_zoo
import time
import d2lzh as d2l
import unit_split
import matplotlib.pyplot as plt

# some superparameters that we can fine-tuning for 'sgd' 
batch_size          = 8  
num_epochs, lr, wd  = 1, 0.1, 5e-4
lr_period, lr_decay = 20, 0.5
epsilon, momentum   = 2e-5, 0.9

ctx = d2l.try_gpu()

# modual parameters
num_classes = 100
num_layers  = 50

# load cifar100 data-set
    # Preprocessing data
transform_train = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(224),
    # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和
    # 宽均为32像素的新图像
    gdata.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                              ratio=(3.0/4.0, 4.0/3.0)),
    gdata.vision.transforms.RandomFlipLeftRight(),
    #gdata.vision.transforms.RandomFlipTopBottom(),
    gdata.vision.transforms.RandomColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.493, 0.480, 0.443], [0.243, 0.239, 0.259])])

transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(32),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.487, 0.482, 0.451], [0.240, 0.238, 0.258])])

num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar100(is_train, augs, batch_size):
    return gdata.DataLoader(
        gdata.vision.CIFAR100(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train, num_workers=num_workers)   

def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

train_iter = load_cifar100(True, transform_train, batch_size)
test_iter = load_cifar100(False, transform_test, batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()

'''
#pretrained_net  = model_zoo.vision.vgg16_bn(pretrained=True)
#pretrained_net  = model_zoo.vision.mobilenet_v2_1_0(pretrained=True)
#pretrained_net  = model_zoo.vision.resnet34_v2(pretrained=True)
#pretrained_net  = model_zoo.vision.resnet18_v2(pretrained=True)
#pretrained_net  = model_zoo.vision.resnet50_v2(pretrained=True)
#pretrained_net  = model_zoo.vision.resnet101_v2(pretrained=True)
net = nn.HybridSequential()
for layer in pretrained_net.features[:-1]:
    net.add(layer)
net.add(nn.Dense(num_classes))

net.initialize(ctx=ctx, init=init.Xavier())
net.hybridize()
net.collect_params().reset_ctx(ctx)

X = nd.random.uniform(shape=(1,3,224,224)).as_in_context(ctx)
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
print('test complit')    

# https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/model_zoo.py

net  = model_zoo.get_model(name='resnext50_32x4d', classes=10, ctx=ctx)
net.initialize(ctx=ctx, init=init.Xavier())

X = nd.random.uniform(shape=(1,3,32,32)).as_in_context(ctx)
for layer in net.features[:]:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

X = net.output(X)
print(net.output.name, 'output shape:\t', X.shape)
print('test complete')
'''
#pretrained_net = model_zoo.get_model(name='mobilenetv2_1.0', ctx=ctx)
pretrained_net = model_zoo.get_model(name='mobilenetv2_0.25', ctx=ctx)
net = nn.HybridSequential()
net.add(pretrained_net.features[0],
        #unit_split.Attention(32),
        unit_split.Attention(8),
        pretrained_net.features[1],        
        pretrained_net.features[2],
        pretrained_net.features[3],
        pretrained_net.features[4],
        #unit_split.Attention(24),
        pretrained_net.features[5],
        #unit_split.Attention(512),
        pretrained_net.features[6],
        #unit_split.Attention(1024),
        pretrained_net.features[7],
        pretrained_net.features[8],
        pretrained_net.features[9],
        pretrained_net.features[10],
        pretrained_net.features[11],
        pretrained_net.features[12],
        pretrained_net.features[13],
        pretrained_net.features[14],
        pretrained_net.features[15],
        pretrained_net.features[16],
        pretrained_net.features[17],
        pretrained_net.features[18],
        pretrained_net.features[19],
        pretrained_net.features[20],
        pretrained_net.features[21],        
        pretrained_net.features[22],        
        #nn.BatchNorm(momentum=2e-5),
        nn.Dropout(0.4),
        pretrained_net.features[23],
        nn.Dense(num_classes),
        nn.BatchNorm(momentum=2e-5))
net.initialize(ctx=ctx, init=init.Xavier())

X = nd.random.uniform(shape=(1,3,224,224)).as_in_context(ctx)
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
print('test complete')    

def train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    train_acc, test_acc = [], []
    train_loss = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        #前40轮每lr_period轮学习率自乘lr_decay
        if epoch>0 and epoch % lr_period == 0 and epoch < 61:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        #超过40轮每10轮学习率自乘lr_decay
        if epoch>61 and epoch % 20 == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for X, y in train_iter:
            y = y.astype('float32').as_in_context(ctx)
            with autograd.record():
                y_hat = net(X.as_in_context(ctx))
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        time_s = "time %.2f sec" % (time.time() - start)
        train_acc.append(train_acc_sum/n)
        train_loss.append(train_l_sum/n)
        if test_iter is not None:
            test_accu = d2l.evaluate_accuracy(test_iter, net, ctx)
            epoch_s = ("epoch %d, loss %f, train acc %f, test_acc %f, "
                       % (epoch + 1, train_l_sum / n, train_acc_sum / n,
                          test_accu))
            test_acc.append(test_accu)
        else:
            epoch_s = ("epoch %d, loss %f, train acc %f, " %
                       (epoch + 1, train_l_sum / n, train_acc_sum / n))
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))
    unit_split.plot(range(1, num_epochs + 1), train_acc, 'epochs', 'accuracy',
              range(1, num_epochs + 1), test_acc, ['train', 'test'])
    plt.figure()
    unit_split.plot(range(1, num_epochs + 1), train_loss, 'epochs', 'loss', legend=['loss'])

#train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay)

'''
# save the model
saved_filename = 'res50cifar10_chuanxing.params'
net.save_parameters(saved_filename)
'''