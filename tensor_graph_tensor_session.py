# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 18:36:39 2017

@author: ZYH
"""
import tensorflow as tf

###Graph
print("-----------Graph--------------")
tf.get_default_graph() #默认计算图

#通过tf.Graph() 建立图，每个图定义一个运算过程，不相互影响
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", [1], initializer = tf.zeros_initializer()) # 设置初始值为0

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", [1], initializer = tf.ones_initializer())  # 设置初始值为1
    
with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
#计算图tf.Graph.device()来指定在哪个设备上跑
        
###Tensor
print("-----------Tensor--------------")
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print(result) 
#Tensor("add:0", shape=(2,), dtype=float32) 不会真的计算
#tf.InteractiveSession()通过设置默认会话的方式来获取张量的取值比较方便。
#直接构建默认会话。
sess = tf.InteractiveSession() #激活默认会话
print(result.eval()) #获取结果
#[ 3.  5.]
sess.close() 

###session
print("-----------Session--------------")
# 创建一个会话,方式1 不建议，需要关闭。
sess = tf.Session()

# 使用会话得到之前计算的结果。
print(sess.run(result))
# 关闭会话使得本次运行中使用到的资源可以被释放。
sess.close() 
#使用with statement 来创建会话 方式2 建议
with tf.Session() as sess:
    print(sess.run(result))
    
#指定默认会话
sess = tf.Session()
with sess.as_default():
     print(result.eval())


with tf.Session() as sess:
#下面的两个命令有相同的功能。
    print(sess.run(result))
    print(result.eval(session=sess))

#通过ConfigProto配置会话
'''
allow_soft_placement=True ：GPU 不行可以到CPU
log_device_placement=True ：日志记载在那些设备运行
'''
'''
config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
'''
