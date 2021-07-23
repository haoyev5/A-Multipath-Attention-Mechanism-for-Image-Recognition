# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:17:57 2020

@author: haoye
"""
import sys
import os
#sys.path.append(os.getcwd())
os.system('dir')
os.system('ls')
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import time
import d2lzh as d2l
sys.path.append(os.getcwd())
import get_net_split
import unit_split
import matplotlib.pyplot as plt

# some superparameters that we can fine-tuning
batch_size          = 256  
num_epochs, lr, wd  = 300, 0.1, 5e-4
lr_period, lr_decay = 30, 0.5
epsilon, momentum   = 2e-5, 0.9

# modual parameters
num_classes = 1000
num_layers  = 18

#data file name
data_dir   ='D:/data/down_imgnet'
train_dir, test_dir = 'train', 'test'

# try to train model on GPU
ctx = d2l.try_gpu()

# Preprocessing data
transform_train = gdata.vision.transforms.Compose([
    # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和
    # 宽均为224像素的新图像
    gdata.vision.transforms.Resize(40),
    gdata.vision.transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.RandomBrightness(0.1),
    gdata.vision.transforms.RandomFlipTopBottom(),
    gdata.vision.transforms.RandomColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
    #gdata.vision.transforms.RandomLighting(0.1),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(32),
    # 将图像中央的高和宽均为224的正方形区域裁剪出来
    #gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])


# load data
train_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, train_dir), flag=1)
if test_dir is not None:
    test_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, test_dir), flag=1)
else:
    test_ds = None
#test_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, test_dir), flag=1)
train_iter = gdata.DataLoader(train_ds.transform_first(transform_train), batch_size=batch_size, shuffle=True, last_batch='keep')
print('train iter complete!')
if test_ds is not None:
    test_iter = gdata.DataLoader(test_ds.transform_first(transform_test), batch_size=batch_size, shuffle=False, last_batch='keep')
    print('test iter complete!')
else:
    test_iter = None
    print('No test iter! Go ahead---->')
#test_iter = gdata.DataLoader(test_ds.transform_first(transform_test), batch_size=batch_size, shuffle=False, last_batch='keep')

loss = gloss.SoftmaxCrossEntropyLoss()

net = get_net_split.get_net(num_classes=num_classes, num_layers=num_layers)
#net = nn.HybridSequential(prefix='')
net.initialize(ctx=ctx, init=init.Xavier())
net.hybridize()
net.collect_params().reset_ctx(ctx)


X = nd.random.uniform(shape=(1,3,32,32)).as_in_context(ctx)
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
    
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

train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay)

'''
# save the model
saved_filename = 'res18test.params'
net.save_parameters(saved_filename)
'''