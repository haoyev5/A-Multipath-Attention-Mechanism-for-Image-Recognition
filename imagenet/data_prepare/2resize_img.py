# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 09:32:51 2020

@author: haoye
"""
import os
import shutil
import time
from time import sleep
from PIL import Image
from tqdm import tqdm
from PIL import Image
import os

def makeDirs(path):
    # 创建新的输出子目录，若存在输出该目录则删除该目录及其子文件夹和文件
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def resizePics(size):

    # default_path=os.getcwd()  ##设置默认打开目录，即当前目录
    # default_path = "/pics"

    # './img' 为即将进行遍历操作的父文件夹路径
    inpath = 'E:\Dataset\ImageNet'  
    # outdir 变量为图像处理之后的输出文件夹
    outdir = os.path.basename(inpath) + '_resize'
    outpath = os.path.join(os.path.dirname(inpath), outdir)
    makeDirs(outpath)

    # 列出所在目录下的所有目录和文件
    lists = os.listdir(inpath)

    starttime = time.time()
    for i in tqdm(range(0, len(lists))):
        subdir = lists[i]
        subpath = inpath + "/" + subdir  # 子目录路径
        if os.path.isdir(subpath):
            # outsubdir = subdir + '-resize'
            outsubdir = subdir
            outsubpath = os.path.join(outpath, outsubdir)

            makeDirs(outsubpath)

            # 对文件夹下的照片文件调整大小
            # 列出某个子目录下的所有文件和目录
            flists = os.listdir(subpath)
            for j in tqdm(range(0, len(flists))):
                sleep(0.01)
                fname = flists[j]
                fpath = subpath + "/" + fname
                if os.path.isfile(fpath):
                    resizeSinglePic(fname, fpath, outsubpath)

    passtime = time.time()-starttime
    print('照片压缩完毕，总共花费了 %f s' % (passtime))


def resizeSinglePic(fname, fpath, outsubpath):
    fbasename = os.path.basename(fpath)
    fext = os.path.splitext(fpath)[-1]
    if fext in img_ext:
        img = Image.open(fpath)
        img.thumbnail(size)
        outfile = outsubpath + "/" + fbasename
        img.save(outfile)

img_ext = ['.bmp', '.jpeg', '.JPEG','.gif', '.psd', '.png', '.JPG', '.jpg']
size = (64, 64)
fpath='D:/data/valid/train_valid_test'
#D:/data/ILSVRC2012_img_train   n01440764
resizePics(size)
'''

def cut_image(file, outpath):
    """
    图片压缩尺寸到400*225大小以内，生成到outpath下
    """
    img = Image.open(file)
    print(img.size)
    (image_x,image_y) = img.size
    if not (image_x <= 400 and image_y <= 225):
        if (image_x/image_y) > (400/225):
            new_image_x = 400
            new_image_y = 400 * image_y // image_x
        else:
            new_image_y = 225
            new_image_x = 225 * image_x // image_y
    else:
        new_image_x = image_x
        new_image_y = image_y
        
    new_image_size = (new_image_x,new_image_y)
    print(new_image_size)
        
    new_img = img.resize(new_image_size,Image.ANTIALIAS)
    new_img.save(outpath+file)
    
 
 
if __name__ == "__main__":
    # 当前路径下的所有文件
    path = 'D:/data/n01440764'
    files  = os.listdir(path)
    
    # 生成当以下文件夹下
    outpath = './small/'
    isExists=os.path.exists(outpath)
    if not isExists:
        os.makedirs(outpath)
  
    # 对图片文件逐一处理
    for file in files:
        filename,filetype = os.path.splitext(file)
    #    print(filename,filetype)
        if filetype == '.jpeg' or filetype == '.jpg' or filetype == '.png':
            print(file)
            cut_image(file, outpath)
    
    # exe生成完后，控制台暂停下
    os.system('pause')
'''