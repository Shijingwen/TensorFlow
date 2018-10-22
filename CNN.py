#!usr/bin/python2
# encoding: utf-8

import tensorflow as tf
import numpy as np
import os
import glob
try:
    from tqdm import tqdm #long waits are not fun
except:
    tqdm = lambda t : t

workspace = "./"
train_path = os.path.join(workspace, "data/npy_cube_train/")
train_lists = glob.glob(train_path+"*.npy")
train_num = train_lists.__len__()
train_set = tf.zeros([train_num, 256, 256])
for i in range(0,train_lists.__len__()+1):
  print(i)
#arr = np.load(train_lists[11])
#print arr.shape


#设置CNN模型参数
x = tf.placeholder(tf.uint8, [None, 65536]) #占位符，运行计算需要。None可输入任意数量图像，每个展平为256*256
w = tf.Variable(tf.zeros([65536, 65536])) #学习权重，784个输入，784个输出
b = tf.Variable(tf.zeros([65536]))
y = tf.nnsoftmax(tf.matmul(x, w)+b)
y_ = tf.placehold("float32",[None, 65536])
cross_enetropy = -tf.reduce_sum(y*tf.log(y))
sess = tf.Session()
sess = tf.InteractiveSession()