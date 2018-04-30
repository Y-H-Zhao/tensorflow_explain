# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:23:52 2017

@author: ZYH
"""
import tensorflow as tf

import numpy as np

'''
构建一个数据集
'''
inputX=np.random.rand(3000,1) #随机数
noise=np.random.normal(0,0.05,inputX.shape) #扰动
outputY=inputX*4+1+noise

#这是第一层 4个隐藏结点
#因为weight 和 bias 是我们需要不断迭代去优化的目标，所以定义为变量：
weight1=tf.Variable(np.random.rand(inputX.shape[1],4))
bias1=tf.Variable(np.random.rand(inputX.shape[1],4))
#因为x是输入，而后面计算流图中还要运算，所以使用占位符 在会话中feed
x=tf.placeholder(tf.float64,[None,1])
#y1_是基于以上的张量计算的输出
y1_=tf.matmul(x,weight1)+bias1
'''
如果是线性拟合，直接隐藏层节点为1就可以了，其他任何形式的函数都可以这样拟合，反正变量就是我们需要拟合的参数
weight1=tf.Variable(np.random.rand(1.0))
bias1=tf.Variable(np.random.rand(1.0))
'''

#定义优化目标
#y是输出对比，在优化流图中需要使用，所以占位符 ，在会话中feed
y=tf.placeholder(tf.float64,[None,1])
#优化目标 reduction_indices=1按列求均值 reduction_indices=0按行 reduction_indices=None 所有
#loss=tf.reduce_sum(tf.pow((y1_-y),2))  #最小二乘
loss=tf.reduce_mean(tf.square(y1_-y),reduction_indices=0) #均方误差
#定义训练模型
trainmodel=tf.train.GradientDescentOptimizer(0.25).minimize(loss) #梯度下降

#布置完毕，实例化会话，初始化 启动
#init=tf.initialize_all_variables()
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#训练实现，feed数据
for i in range(1000):
    sess.run(trainmodel,feed_dict={x:inputX,y:outputY})
    #if(i%100==0):
        #print(loss)

#结束，显示结果
print(weight1.eval(sess))
print("-------------------")
print(bias1.eval(sess))
print("--------结果-----------")

#由于有y1_是图中最后计算出来的结果变量  所以模型保存在y1_中
x_data=np.matrix([[1.],[2.],[3.]])
print(sess.run(y1_,feed_dict={x:x_data})) 

'''
加入两层 也是同样道理的
#第一层
weight1 = tf.Variable(np.random.rand(inputX.shape[1],4))
bias1 = tf.Variable(np.random.rand(inputX.shape[1],4))
x = tf.placeholder(tf.float64, [None, 1])
y1_ = tf.matmul(x, weight1) + bias1
#第二层
weight2 = tf.Variable(np.random.rand(4,1))
bias2 = tf.Variable(np.random.rand(inputX.shape[1],1))
y2_ = tf.matmul(y1_, weight2) + bias2

y = tf.placeholder(tf.float64, [None, 1])

loss = tf.reduce_mean(tf.square(y2_ - y))
trainmodel = tf.train.GradientDescentOptimizer(0.25).minimize(loss)  # 选择梯度下降法
'''
