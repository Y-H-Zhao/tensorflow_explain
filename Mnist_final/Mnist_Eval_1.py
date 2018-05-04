# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:40:40 2017

@author: ZYH
"""
#import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import Mnist_Inference_1
import Mnist_Train_1

#每10s加载一次最新的模型，并在测试数据上测试
#eval_interval_secs=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,Mnist_Inference_1.input_node],name="x-input")
        y_=tf.placeholder(tf.float32,[None,Mnist_Inference_1.output_node],name="y-input")
        validate_feed={x:mnist.validation.images,
                       y_:mnist.validation.labels}
        #测试时不适用正则
        y=Mnist_Inference_1.inference(x,None)
        #使用移动平均结果计算正确率（优化不使用移动平均，验证结果使用移动平均）
        #correct_prediction一个一维布尔值数组，维度是batch_size
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        #求均值 先将布尔型转为浮点型
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        '''
        通过变量重命名的方式加载模型，前向传播中就不需要调用求滑动平均的
        函数来获取平均值，可以共用Mnist_Inference_1.py中定义的前向传播
        '''
        variable_averages=tf.train.ExponentialMovingAverage(Mnist_Train_1.moving_average_decay)
        variable_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variable_to_restore) #恢复
        #每隔eval_interval_secs秒计算正确率，检测正确率的变化
        '''
        while True:
            with tf.Session() as sess:
                #tf.train.get_checkpoint_state 通过checkpoint文件
                #自动获取最新文件名
                ckpt=tf.train.get_checkpoint_state(
                    Mnist_Train_1.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #通过文件名获取轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accurucy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("After {0} training step(s),validate accurucy is {1} ".format(global_step,accurucy_score))
                else:
                    print("No checkpoint file found")
                    return()
            time.sleep(eval_interval_secs)
        '''
        with tf.Session() as sess:
                #tf.train.get_checkpoint_state 通过checkpoint文件
                #自动获取最新文件名
                ckpt=tf.train.get_checkpoint_state(
                    Mnist_Train_1.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #通过文件名获取轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accurucy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("After {0} training step(s),validate accurucy is {1} ".format(global_step,accurucy_score))
                else:
                    print("No checkpoint file found")
                    return()
def main(argv=None):
    mnist=input_data.read_data_sets("../../../datasets/MNIST_data",one_hot=True)
    evaluate(mnist)
    
if __name__ =='__main__':
    main()
