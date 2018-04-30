# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 09:47:53 2017

@author: ZYH
"""
import tensorflow as tf
#Tensor矩阵生成和计算是十分重要的
#创建一个张量矩阵
Tensor_matrix1=tf.constant([1,2,3],shape=[2,3])
#tf.Print(Tensor_matrix1,[Tensor_matrix1],message="This is Tensor_matrix1:")
#[[1 2 3]
# [3 3 3]]
#随机生成
'''
正态分布
tf.random_normal(shape,mean= ,stddev= ,dytype=tf.float32, seed= None,name=None)
截断正态分布
tf.truncated_normal()
均匀分布
tf.random_uniform()
'''
Tensor_matrix2=tf.random_normal([10,10],mean=0,stddev=1,dtype=tf.float32, seed= None,name=None)
#获取形状
print(tf.shape(Tensor_matrix1))
print(tf.shape(Tensor_matrix2))
#改变形状tf.reshape(tensor,shape,name=None) 
#Numpy中也有np.reshape（）使用同理
'''
shape=[-1] 表示将tensor展开成一个list
shape=[a,b,c,...] 常规操作a,b,c>0
shape=[a,-1,c,...] 此时b=-1，tf 根据tensor的原尺度，自动计算b的值。
'''
#对矩阵的操作
'''
取对角 tf.diag_part()
求迹 tf.trace()
调整矩阵维度顺序 tf.transpose()
矩阵乘法 tf.matmul()
求逆 cholesky分解等等
'''
print(tf.diag_part(Tensor_matrix2))
