# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:45:00 2018

@author: ZYH
"""
import tensorflow as tf
from numpy.random import RandomState

#定义训练数据batch 的大小
batch_size=8
#定义参数 优化系数 为变量
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#x的shape设置上 一个维度使用None可以方便使用不同的batch大小，在训练
#时把数据分割成较小的batch，但是测试时，可以一次性使用全部数据，当数据
#集较小方便测试，但是数据集很大时，可能会导致内存溢出
#占位符 用来feed
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

#定义图的结构 前向传播
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
y=tf.sigmoid(y)
#定义反向传播和损失函数
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm = RandomState(seed=1) #seed=1
dataset_size=128
X=rdm.rand(dataset_size,2)

#标签 x1+x2<1 为正样本 赋值为1 其他反之
Y=[[int(x1+x2<1)] for (x1,x2) in X]

#建立会话
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    #显示一下运行之前w1 w2的初始值
    print(sess.run(w1))
    print(sess.run(w2))
    Steps=5000
    for i in range(Steps):
        start = (i*batch_size) % dataset_size
        end = (i*batch_size) % dataset_size + batch_size
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    
    # 输出训练后的参数取值。
    print("\n")
    print(sess.run(w1))
    print(sess.run(w2))
