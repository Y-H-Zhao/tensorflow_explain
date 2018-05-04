# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:40:40 2017

@author: ZYH
"""
import tensorflow as tf

#定义神经网络结构
input_node=784
output_node=10
layer1_node=500

'''
通过tf.get_variable函数来获取变量，在训练神经网络时会创建这些变量
在测试时通过保存的模型加载获取这些变量的取值，更是：变量加载时将滑动平均
变量重命名，可以直接通过同样的名字在训练时使用变量本身，而在测试时，使用变量的
滑动平均值。
'''
#定义获取变量，将正则化损失加入损失集合
def get_weight_variable(shape,regularizer):
    weights=tf.get_variable(
            "weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    #当给出正则化生成函数时，将当前变量的正则化损失加入losses的集合，自定义
    #集合，不在tensorflow的自动管理列表
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights)) #将张量加入集合losses
        
    return(weights)

#定义前向传播
def inference(input_tensor,regularizer):
    #第一层，前向传播
    with tf.variable_scope('layer1'):
        #训练和测试时，没有多次调用本函数，如果多次调用，除了第一次，reuse必须设置为True
        #tf.get_variable和tf.Variable没有本质区别，因为没有多次调用
        weights=get_weight_variable([input_node,layer1_node],regularizer)
        biases=tf.get_variable("biases",[layer1_node],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
        
    #第二层
    with tf.variable_scope('layer2'):
        #训练和测试时，没有多次调用本函数，如果多次调用，除了第一次，reuse必须设置为True
        #tf.get_variable和tf.Variable没有本质区别，因为没有多次调用
        weights=get_weight_variable([layer1_node,output_node],regularizer)
        biases=tf.get_variable("biases",[output_node],initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer1,weights)+biases
        
    return(layer2)
