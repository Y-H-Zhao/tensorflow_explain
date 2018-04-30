# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:19:33 2017

@author: ZYH
"""
import tensorflow as tf

#定义变量 所有变量必须初始化才可以使用
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.constant([[0.7, 0.9]])  

#前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
sess.run(w1.initializer)  
sess.run(w2.initializer)  
print(sess.run(y))  
sess.close()

#使用placeholder ,使用tf.global_variables_initializer()来初始化所有的变量
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  
    sess.run(init_op)
    print(sess.run(y, feed_dict={x: [[0.7,0.9]]}))
    
'''
变量是一种特殊的张量，在定义变量时：tf.Variable()这是一个运行，其结果就是一个
张量，例如：w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w1是一个变量运算得到的张量。
'''
