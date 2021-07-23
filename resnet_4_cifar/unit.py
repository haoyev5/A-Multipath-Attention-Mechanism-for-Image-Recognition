# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:54:37 2020

@author: haoye
"""
import mxnet as mx
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils

# chuanxing 
class Attention(nn.Block):
    def __init__(self, channels, **kwargs):
        super(Attention, self).__init__(**kwargs)        
        self.mean = nn.GlobalAvgPool2D()
        self.flat = nn.Flatten()               
        self.dense1 = nn.Dense(units=channels//16, activation='relu')
        self.dense2 = nn.Dense(units=channels, activation='relu')        
        #self.conv11 = nn.Conv2D(channels=channels//16, kernel_size=1, strides=1)
        #self.conv12 = nn.Conv2D(channels=channels, kernel_size=1, strides=1)        
        self.spatial_bn    = nn.BatchNorm(epsilon=2e-5, momentum=0.9)
        self.spatial_conv1 = nn.Conv2D(channels=channels//16, kernel_size=7, strides=1, padding=3)
        self.spatial_conv2 = nn.Conv2D(channels=1, kernel_size=3, strides=1, padding=1)
        
    def forward(self, X):
        # channel attention
        c = self.dense1(self.flat(self.mean(X)))
        c = self.dense2(c)
        channel_att = nd.sigmoid(c)
        channel_att = nd.broadcast_mul(X, nd.reshape(channel_att, shape=(-1, channel_att.shape[1], 1, 1)))
        #spatial attention
        spatial_att = nd.relu(self.spatial_bn(self.spatial_conv1(channel_att)))
        spatial_att = nd.sigmoid(self.spatial_conv2(spatial_att))       
        out = nd.broadcast_mul(channel_att, spatial_att)
        return out

'''
# bing xing dense, 
class Attention(nn.Block):
    def __init__(self, channels, **kwargs):
        super(Attention, self).__init__(**kwargs)        
        self.mean = nn.GlobalAvgPool2D()
        self.flat = nn.Flatten()               
        self.dense1 = nn.Dense(units=channels//16, activation='relu')
        self.dense2 = nn.Dense(units=channels, activation='relu')
        
        #self.conv21 = nn.Conv2D(channels=channels, kernel_size=3, strides=1, padding=1)        
        self.spatial_bn    = nn.BatchNorm(epsilon=2e-5, momentum=0.9)
        self.spatial_conv1 = nn.Conv2D(channels=channels//16, kernel_size=5, strides=1, padding=2)
        self.spatial_conv2 = nn.Conv2D(channels=1, kernel_size=5, strides=1, padding=2)
        self.concat_conv = nn.Conv2D(channels=channels, kernel_size=1, strides=1)        
        
    def forward(self, X):
        # chennel attention：avgpool -> conv(c=1/r) -> relu ->  conv(c=c) -> sigmoid -> broadcast_mul
        # spatial attention: conv(3*3) -> BN Relu -> conv(7*7) -> sigmoid -> broadcast_mul
        # channel attention
        c = self.dense1(self.flat(self.mean(X)))
        c = self.dense2(c)
        channel_att = nd.sigmoid(c)
        channel_att = nd.broadcast_mul(X, nd.reshape(channel_att, shape=(-1, channel_att.shape[1], 1, 1)))
        #spatial attention
        spatial_att = nd.relu(self.spatial_bn(self.spatial_conv1(X)))
        spatial_att = nd.sigmoid(self.spatial_conv2(spatial_att))       
        spatial_att = nd.broadcast_mul(X, spatial_att)
        out = self.concat_conv(nd.concat(channel_att, spatial_att))
        return nd.relu(out)

# bing xing 
class Attention(nn.Block):
    def __init__(self, channels, **kwargs):
        super(Attention, self).__init__(**kwargs)        
        self.mean = nn.GlobalAvgPool2D()        
        self.conv11 = nn.Conv2D(channels=channels//16, kernel_size=1, strides=1)
        self.conv12 = nn.Conv2D(channels=channels, kernel_size=1, strides=1)        
        #self.conv21 = nn.Conv2D(channels=channels, kernel_size=3, strides=1, padding=1)        
        self.spatial_bn    = nn.BatchNorm(epsilon=2e-5, momentum=0.9)
        self.spatial_conv1 = nn.Conv2D(channels=channels//16, kernel_size=5, strides=1, padding=2)
        self.spatial_conv2 = nn.Conv2D(channels=1, kernel_size=3, strides=1, padding=1)        
        
    def forward(self, X):
        # chennel attention：avgpool -> conv(c=1/r) -> relu ->  conv(c=c) -> sigmoid -> broadcast_mul
        # spatial attention: conv(3*3) -> BN Relu -> conv(7*7) -> sigmoid -> broadcast_mul
        # channel attention
        c1 = nd.relu(self.conv11(self.mean(X)))
        c1 = self.conv12(c1)
        channel_att = nd.sigmoid(c1)
        channel_att = nd.broadcast_mul(X, channel_att)
        #spatial attention
        spatial_att = nd.relu(self.spatial_bn(self.spatial_conv1(X)))
        spatial_att = nd.sigmoid(self.spatial_conv2(spatial_att))       
        spatial_att = nd.broadcast_mul(X, spatial_att)
        return channel_att+spatial_att
    
X = nd.random.uniform(shape=(1,32,32,32))
blk = Attention(32)
blk.initialize()
y = blk(X)
print(y.shape)

# chuan xing best performance: 92.51
class Attention(nn.Block):
    def __init__(self, channels, **kwargs):
        super(Attention, self).__init__(**kwargs)        
        self.mean = nn.GlobalAvgPool2D()       
        self.conv11 = nn.Conv2D(channels=channels//16, kernel_size=1, strides=1)
        self.conv12 = nn.Conv2D(channels=channels, kernel_size=1, strides=1)       
        #self.conv21 = nn.Conv2D(channels=channels, kernel_size=3, strides=1, padding=1)        
        self.spatial_bn    = nn.BatchNorm(epsilon=2e-5, momentum=0.9)
        self.spatial_conv1 = nn.Conv2D(channels=channels//16, kernel_size=7, strides=1, padding=3)
        self.spatial_conv2 = nn.Conv2D(channels=1, kernel_size=3, strides=1, padding=1)
        
    def forward(self, X):
        # chennel attention：avgpool -> conv(c=1/r) -> relu ->  conv(c=c) -> sigmoid -> broadcast_mul
        # spatial attention: conv(3*3) -> BN Relu -> conv(7*7) -> sigmoid -> broadcast_mul
        # channel attention
        c1 = nd.relu(self.conv11(self.mean(X)))
        c1 = self.conv12(c1)
        #c2 = self.conv21(X)
        channel_att = nd.sigmoid(c1)
        channel_att = nd.broadcast_mul(X, channel_att)
        #spatial attention
        spatial_att = nd.relu(self.spatial_bn(self.spatial_conv1(channel_att)))
        spatial_att = nd.sigmoid(self.spatial_conv2(spatial_att))       
        out = nd.broadcast_mul(channel_att, spatial_att)
        return out
        
# 并行后 concat 经1✖1 卷积 输出
class Attention(nn.Block):
    def __init__(self, channels, **kwargs):
        super(Attention, self).__init__(**kwargs)        
        self.mean = nn.GlobalAvgPool2D()        
        self.conv11 = nn.Conv2D(channels=channels//16, kernel_size=1, strides=1)
        self.conv12 = nn.Conv2D(channels=channels, kernel_size=1, strides=1)        
        #self.conv21 = nn.Conv2D(channels=channels, kernel_size=3, strides=1, padding=1)        
        self.spatial_bn    = nn.BatchNorm(epsilon=2e-5, momentum=0.9)
        self.spatial_conv1 = nn.Conv2D(channels=channels//16, kernel_size=5, strides=1, padding=2)
        self.spatial_conv2 = nn.Conv2D(channels=1, kernel_size=3, strides=1, padding=1)        
        self.concat_conv = nn.Conv2D(channels=channels, kernel_size=1, strides=1)
        
    def forward(self, X):
        # chennel attention：avgpool -> conv(c=1/r) -> relu ->  conv(c=c) -> sigmoid -> broadcast_mul
        # spatial attention: conv(3*3) -> BN Relu -> conv(7*7) -> sigmoid -> broadcast_mul
        # channel attention
        c1 = nd.relu(self.conv11(self.mean(X)))
        c1 = self.conv12(c1)
        channel_att = nd.sigmoid(c1)
        channel_att = nd.broadcast_mul(X, channel_att)
        #spatial attention
        spatial_att = nd.relu(self.spatial_bn(self.spatial_conv1(X)))
        spatial_att = nd.sigmoid(self.spatial_conv2(spatial_att))       
        spatial_att = nd.broadcast_mul(X, spatial_att)
        out = self.concat_conv(nd.concat(channel_att, spatial_att))
        return out
    
  
class Attention3(nn.Block):
    def __init__(self, channels, **kwargs):
        super(Attention3, self).__init__(**kwargs)        
        self.mean = nn.GlobalAvgPool2D()
        self.max = nn.GlobalMaxPool2D()
        self.conv11 = nn.Conv2D(channels=channels//16, kernel_size=1, strides=1)
        self.conv12 = nn.Conv2D(channels=channels, kernel_size=1, strides=1)
        self.conv21 = nn.Conv2D(channels=channels//16, kernel_size=1, strides=1)
        self.conv22 = nn.Conv2D(channels=channels, kernel_size=1, strides=1)
        self.spatial_conv = nn.Conv2D(channels=1, kernel_size=3, strides=1, padding=1)
        
    def forward(self, X):
        # channel attention channel1：avgpool -> conv(c=1/r) -> relu ->  conv(c=c)   
        # channel attention channel2: maxpool -> conv(c=1/r) -> relu ->  conv(c=c)   
        # c1+c2 -> sigmoid -> broadcast_mul
        # spatial attention: concat[mean,max] ->conv(c=c,3*3) -> sigmoid -> broadcast_mul
        # channel attention
        # channel attention
        c1 = nd.relu(self.conv11(self.mean(X)))
        c1 = self.conv12(c1)
        c2 = nd.relu(self.conv21(self.max(X)))
        c2 = self.conv22(c2)
        channel_att = nd.sigmoid(c1+c2)
        channel_att = nd.broadcast_mul(X, channel_att)
        #spatial attention
        spatial_att = nd.concat(nd.mean(channel_att, axis=1, keepdims=True), 
                                nd.max(channel_att, axis=1, keepdims=True))
        spatial_att = nd.sigmoid(self.spatial_conv(spatial_att))
        out = nd.broadcast_mul(channel_att, spatial_att)
        return out

    
class Attention_Residual_Unit(nn.Block):
    def __init__(self, channels, strides, same_shape=True, **kwargs):
        super(Attention_Residual_Unit, self).__init__(**kwargs)
        epsilon=  2e-5        
        momentum= 0.9
        self.same_shape = same_shape
        self.conv0  = nn.Conv2D(channels=channels, kernel_size=1, strides=1, padding=0)
        self.bn1    = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        self.conv1  = nn.Conv2D(channels=channels, kernel_size=3, strides=1, padding=1)
        self.bn2    = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        self.conv2  = nn.Conv2D(channels=channels, kernel_size=1, strides=strides, padding=0)
        self.bn3    = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        self.att    = Attention(channels)
        
        if not same_shape:
            self.conv = nn.Conv2D(channels=channels,kernel_size=1,strides=strides, padding=0)
            self.bn   = nn.BatchNorm(epsilon=epsilon,momentum=momentum)            
        
    def forward(self, X):
        y = nd.relu(self.bn2(self.conv1(self.bn1(self.conv0(X)))))
        y = self.att(self.bn3(self.conv2(y)))
        if not self.same_shape:
            X = self.bn(self.conv(X))
        return nd.relu(X+y)

'''
class Attention_Residual_Unit(nn.Block):
    def __init__(self, channels, strides, same_shape=True, **kwargs):
        super(Attention_Residual_Unit, self).__init__(**kwargs)
        epsilon= 2e-5        
        momentum=0.9
        self.same_shape = same_shape
        self.bn1    = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        self.conv1  = nn.Conv2D(channels=channels, kernel_size=3, strides=1, padding=1)
        self.bn2    = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        self.conv2  = nn.Conv2D(channels=channels, kernel_size=3, strides=strides, padding=1)
        self.bn3    = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        self.att    = Attention(channels)
        
        if not same_shape:
            self.conv = nn.Conv2D(channels=channels,kernel_size=1,strides=strides, padding=0)
            self.bn   = nn.BatchNorm(epsilon=epsilon,momentum=momentum)            
        
    def forward(self, X):
        y = nd.relu(self.bn2(self.conv1(self.bn1(X))))
        y = self.att(self.bn3(self.conv2(y)))
        if not self.same_shape:
            X = self.bn(self.conv(X))
        return X+y

def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

class IBN_block(nn.Block):
    def __init__(self, epsilon=2e-5, momentum=0.9, **kwargs):
        super(IBN_block, self).__init__(**kwargs)
        
        self.ins = nn.InstanceNorm(epsilon=epsilon)
        self.bn  = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        
    def forward(self, x):
        split = nd.split(data=x, axis=1, num_outputs=2)
        c1    = self.ins(split[0])
        c2    = self.bn(split[1])
        return nd.concat(c1, c2, dim=1)

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
        
def plot(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(7, 5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.plot(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.plot(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)