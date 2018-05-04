# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:40:40 2017

@author: ZYH
"""
import os

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

import Mnist_Inference_1

#配置参数
batch_size=100
learning_rate_base=0.8
learning_rate_decay=0.99
regularization=0.0001
train_steps=5000
moving_average_decay=0.99

#模型保存路径和文件名
model_save_path="model/"
model_name="model.ckpt"

def train(mnist):
    #占位
    x=tf.placeholder(tf.float32,[None,Mnist_Inference_1.input_node],name="x-input")
    y_=tf.placeholder(tf.float32,[None,Mnist_Inference_1.output_node],name="y-input")
    
    regularizer=tf.contrib.layers.l2_regularizer(regularization)
    
    #前向传播
    y=Mnist_Inference_1.inference(x,regularizer) #这里大大简化了
    #下面计算avg_class存在的前向传播结果
    #定义存储训练次数的变量，此变量不适用移动平均，不进入训练trainable=False
    global_step=tf.Variable(0,trainable=False)
    
    #实例化一个移动平均类
    variable_averages=tf.train.ExponentialMovingAverage(
            moving_average_decay,global_step)
    
    #在所有的神经网络中变量上使用移动平均
    #tf.trainable_variables()返回计算图上没有指定trainable=False的
    #所有变量
    variable_averages_op=variable_averages.apply(
            tf.trainable_variables())
    #这里注意，移动平均方法会产生一个影子变量，不会改变变量的实际值，
    #需要使用移动平均值时，需要使用average函数调用其值
    #例如variable_averages.average(weights1)
    '''
    #计算使用了移动平均的前向传播结果
    average_y=inference(x,variable_averages,weights1,biases1,
                        weights2,biases2)
    '''
    #计算交叉熵作为损失函数,tf.argmax(y_,1)获取正确答案对应的编号所构成的概率分布
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
    #一个batch内所有样本的平均值
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    
    #总损失函数,在losses集合中提取出来
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    
    
    #设置指数衰减学习率 #四个参数 
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        mnist.train.num_examples / batch_size,
        learning_rate_decay,
        staircase=True) #staircase=True 学习率梯行下降
    #训练目标
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #千万不要忘了gloabal_step
    #每一次训练神经网络，既要通过反向传播来更新参数，
    #又要移动平均更新参数，为了一次完成多个操作tensorflow提供了
    #tf.control_dependencies和tf.grounp
    #train_op=tf.grounp(train_step,variable_averages_op)与下面这个等价
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')
    
    #初始化持久化类
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #在训练过程中不再验证和测试
        for i in range(train_steps+1):
            xs,ys=mnist.train.next_batch(batch_size)
            #运行三个目标[train_op,loss,global_step]
            _,loss_value,step=sess.run([train_op,loss,global_step],
                                       feed_dict={x:xs,y_:ys})
            #每1000次保存模型
            if(i % 1000 ==0):
                print("After {0} training step(s), loss on training batch is {1} ".format(step,loss_value))
                
                #保存模型，这里给出了global_step,可以让保存的模型末尾加上训练的轮数
                saver.save(
                        sess,os.path.join(model_save_path,model_name),global_step=global_step)
    
def main(argv=None):
    mnist=input_data.read_data_sets("../../../datasets/MNIST_data",one_hot=True)
    train(mnist)
    
if __name__=='__main__':
    #tf.app.run()
    main()
