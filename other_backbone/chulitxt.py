# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:25:27 2020

@author: haoye
"""


file = open("v.txt", "r")  # 以只读模式读取文件
lines = []
for i in file:
    lines.append(i)  # 逐行将文本存入列表lines中
file.close()
new = []
for line in lines:
    #print(line[29:])
    line = line[29:].replace("\n", "")
    #print(line)
    new_label = int(line)+1
    #print("11111111111111111111111111111111111111111")
    new.append(str(new_label))
    #print("22222222222222222222222222222222222222222")
    '''
    if line=='\n':
        break
    else:
        a = line[29:]
        #print(a)
        new_label = int(a)
        new_label += 1
        new.append(new_label)
    '''

print(111)
        
file_write_obj = open("new.txt", 'w')
for var in new:
    file_write_obj.writelines(var)
    file_write_obj.writelines('\n')


