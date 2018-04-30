# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 09:28:29 2017

@author: ZYH
"""
import tensorflow as tf

input1=tf.constant(1)
print(input1)

input2=tf.Variable(2,tf.int32)
print(input2)

input2=input1 
'''
input2=6
报错，因为不能如此赋值
'''
sess=tf.Session()
print(sess.run(input2)) 

#占位符的应用 占位符平时作为空的张量进行相应的运算 
#当真正运行时，在会话的过程中不断填入数据
input1=tf.placeholder(tf.int32)
input2=tf.placeholder(tf.int32)

output=tf.add(input1,input2) 

#在运行会话，生成图之前，一切节点，边，其中的运算都是提前设计好的。
#至于其中的变量，可以使用占位符预留，真正运行时，feed填入.
sess=tf.Session()
#run方法中第一个对象是结果张量，或者说目标张量 例如output
print(sess.run(output,feed_dict={input1:[1,3],input2:[2,5]})) 
#tensor提供了基础函数，例如加减乘除 tf.add tf.sub tf.mul tf.div 
#取模tf.mod 绝对值tf.abs 取负tf.neg 等等
