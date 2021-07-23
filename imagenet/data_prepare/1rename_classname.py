# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:40:36 2020

@author: haoye
"""
import os
import linecache

'''
#将文件夹中的文件夹(类别)依次写入txt文件中
dirct = 'C:/Users/84999/.mxnet/datasets/imagenet/train'
dirList=[]
fileList=[]
files=os.listdir(dirct)  #文件夹下所有目录的列表
print('files:',files)
for f in files:
    if os.path.isdir(dirct + '/'+f):   #这里是绝对路径，该句判断目录是否是文件夹
        dirList.append(f)
    elif os.path.isfile(dirct + '/'+f):#这里是绝对路径，该句判断目录是否是文件
        fileList.append(f)
#print("文件夹有：",dirList)
#print("文件有：",fileList)

f=open("D:/data/classes.txt","w")
 
for line in dirList:
    f.write(line+'\n')
f.close()
'''
'''
# 重命名文件夹中的类别文件夹名为：1、2、3······
path = 'E:\Dataset\ImageNet\ILSVRC2012_img_val'
i = 1
for file in os.listdir(path):
    #判断是否是文件夹
    if os.path.isdir(os.path.join(path,file))==True:
        #设置新文件夹名
        new_name=file.replace(file,"%d"%i)
        #重命名
        os.rename(os.path.join(path,file),os.path.join(path,new_name))
        i+=1
#结束
print('end')

'''
# 重命名文件夹中的JPEG文件，1、2、3······
path = 'E:\Dataset\ImageNet\ILSVRC2012_img_val'
i = 1
for file in os.listdir(path):
    #判断是否是文件
    if os.path.isfile(os.path.join(path,file))==True:
        #设置新文件名
        new_name=file.replace(file,"%d.JPEG"%i)
        #重命名
        os.rename(os.path.join(path,file),os.path.join(path,new_name))
        i+=1
#结束
print('end')

