# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:46:22 2018

@author: ZYH
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
input_node=784 #输入节点数 图片像素
outout_node=10 #类别

#配置参数
layer1_node=500 #隐层节点数
batch_size=100

learning_rate_base=0.8
learning_rate_decay=0.99

regularization_rate=0.0001 #正则化系数
training_step=5000 #训练轮数
moving_average_decay=0.99 #移动平均衰减率

#定义前向传播函数，Relu激活函数，支持传入参数移动平均的类(这个应该在训练过程中想到构造)
def inference(input_tensor,avg_class,weights1,biases1,
              weights2,biases2):
    #当没有提供移动平均类时，直接使用参数当前的值
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        #计算损失函数时，一并计算softmax，所以在这里不加入softmax 
        return(tf.matmul(layer1,weights2)+biases2)
        
    else:
        #首先使用移动平均来计算变量的值
        #然后再计算前向传播结果
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+
                          avg_class.average(biases1))
        return(tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2))

#训练过程
def train(mnist):
    #实际数据占位
    x=tf.placeholder(tf.float32,[None,input_node],name='x-input')
    y_=tf.placeholder(tf.float32,[None,outout_node],name='y-intput')
    
    #定义变量
    #隐藏层
    weights1= tf.Variable(tf.truncated_normal([input_node,layer1_node],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[layer1_node]))
    
    #输出层
    weights2=tf.Variable(tf.truncated_normal([layer1_node,outout_node],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[outout_node]))
    
    #计算前向传播结果 需要四步，一步一步来
    #这里avg_class=None
    y=inference(x,None,weights1,biases1,weights2,biases2)
    
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
    
    #计算使用了移动平均的前向传播结果
    average_y=inference(x,variable_averages,weights1,biases1,
                        weights2,biases2)
    
    #计算交叉熵作为损失函数,tf.argmax(y_,1)获取正确答案对应的编号所构成的概率分布
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
    #一个batch内所有样本的平均值
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    
    #需要两步，一步一步来
    #计算正则化函数
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    #计算正则化损失
    regularization=regularizer(weights1)+regularizer(weights2)
    
    #总损失函数
    loss=cross_entropy_mean+regularization
    
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
    #使用移动平均结果计算正确率（优化不使用移动平均，验证结果使用移动平均）
    #correct_prediction一个一维布尔值数组，维度是batch_size
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #求均值 先将布尔型转为浮点型
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    #布置完毕，初始化，开始训练
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        #准备验证数据
        validate_feed = {x:mnist.validation.images,
                         y_:mnist.validation.labels}
        #准备测试数据，在真实应用中，这部分数据在训练时是不可见的
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        
        #迭代训练
        for i in range(training_step):
            #每1000轮输出一个验证集的测试结果
            if i%1000==0:
                validate_acc=session.run(accuracy,feed_dict=validate_feed)
                print("After {0} training step(s), validation accuracy, using average model is {1} ".format(i,validate_acc))
                
            #产生这一轮的训练数据 并训练
            xs,ys=mnist.train.next_batch(batch_size)
            session.run(train_op,feed_dict={x:xs,y_:ys})
            
        #训练结束，测试集的正确率
        test_acc=session.run(accuracy,feed_dict=test_feed)
        print("After {0} training step(s), test accuracy, using average model is {1} ".format(training_step,test_acc))
        
        
#主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    train(mnist)
#tensorflow提供一个主程序，调用上面的main函数
if __name__=='__main__':
    main()
