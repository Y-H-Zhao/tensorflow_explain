# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:12:50 2018

@author: ZYH
"""
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import os
#参数设置
input_size=6 #6个影响因子
output_size=1
time_step=8
#模型保存路径和文件名
model_save_path="model_test/"
model_name="model.ckpt"
#超参数设置
Option_design=input("是否自己设置超参数(y/n):")
out=False
while out==False:
    if Option_design=='N' or Option_design =='n':
        rnn_unit=18   #隐层数量
        lr_base=0.1
        lr_decay=0.99
        batch_size=100
        input_keep_prob=0.8
        output_keep_prob=0.8
        train_steps=2000
        out=True
    elif Option_design=='Y' or Option_design =='y':
        rnn_unit=int(input("设置隐藏层节点数："))
        lr_base=float(input("设置基础学习率："))
        lr_decay=float(input("设置学习率衰减系数："))
        batch_size=int(input("设置batch_size："))
        input_keep_prob=float(input("设置输入层dropout系数："))
        output_keep_prob=float(input("设置输出层dropout系数："))
        train_steps=int(input("设置训练轮数："))
        out=True
    else:
        Option_design=input("输入无效，请选择是否自己设置超参数(y/n):")
variable_sco=input("设置变量空间(例如：lstm)，可重复训练：")
'''
BasicRNNCell/BasicLSTMCell/GRUCell/RNNCell/LSTMCell
'''
Select_Function=input("请输入数字选择RNN结构(1:BasicLSTMCell -- 2:GRUCell): ")
S_out=False
while S_out==False:
    if Select_Function=='1' or Select_Function=='2':
        S_out=True
    else:
        Select_Function=input("选择无效，请输入数字选择RNN结构(1:BasicLSTMCell -- 2:GRUCell): ")
    
    
print("----------训练即将开始-------------")
#读取数据
f=open('dem_loss_sales_sig.csv') #降维后数据
df=pd.read_csv(f)     #读入股票数据
#data=df.iloc[:,:13].values  #取第1-13列
data=df.values #全部数据

'''
共24000条数据，每8个一组，共3000组，前3000*0.8=2400组作为训练集（19200条数据）
然后3000*0.1=300组作为验证集（2400条数据），最后3000*0.1=300组作为测试集（2400条数据）
'''

#构建训练，验证，测试数据
#获取训练集
#batch_size=100,time_step=8,
def get_train_data(train_begin=0,train_end=19200):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    train_num=len(normalized_train_data)
    train_size=len(normalized_train_data)//time_step
    for i in range(train_size):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_train_data[i*time_step:(i+1)*time_step,6,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)//time_step))
    return batch_index,train_x,train_y,train_num 
#batch_size=100,time_step=8,
_,train_x,_,_ =get_train_data()

#获取验证集
#time_step=8,
def get_verify_data(verify_begin=19200,verify_end=21600):
    data_verify=data[verify_begin:verify_end]
    mean=np.mean(data_verify,axis=0)
    std=np.std(data_verify,axis=0)
    normalized_verify_data=(data_verify-mean)/std  #标准化
    verify_size=len(normalized_verify_data)//time_step  #有size个sample
    verify_x,verify_y=[],[]
    for i in range(verify_size):
       x=normalized_verify_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_verify_data[i*time_step:(i+1)*time_step,6]
       verify_x.append(x.tolist())
       verify_y.extend(y)
    return mean,std,verify_x,verify_y
#time_step=8,


#获取测试集
#time_step=8,
def get_test_data(test_begin=21600):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    test_size=len(normalized_test_data)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(test_size):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_test_data[i*time_step:(i+1)*time_step,6]
       test_x.append(x.tolist())
       test_y.extend(y)
    return mean,std,test_x,test_y
#time_step=8,


def lstm(X,istrain):
    '''
    #正确的标签是一套处理方法
    正确的标签
    with tf.variable_scope('out'):
        #训练和测试时，没有多次调用本函数，如果多次调用，除了第一次，reuse必须设置为True
        #tf.get_variable和tf.Variable没有本质区别，因为没有多次调用
        w_out=tf.get_variable(
            "w_out",[rnn_unit,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_out=tf.get_variable("b_out",[1],initializer=tf.constant_initializer(0.0))
    '''
    if Select_Function=='1':
        cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    elif Select_Function=='2':
        cell=tf.nn.rnn_cell.GRUCell(rnn_unit)
    else:
        ex = Exception("选择方法有误！请重新运行程序！")
        raise ex
    if istrain==True:
        cell=tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob,output_keep_prob)
    else:
        cell=tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0,output_keep_prob=1.0)
    #init_state=cell.zero_state(batch_size,dtype=tf.float32)
    # initial_state=init_state,
    output_rnn,final_states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
    '''
    正确的标签
    output=tf.reshape(output_rnn,[-1,rnn_unit])
    predictions=tf.matmul(output,w_out)+b_out
    '''
    predictions = tf.contrib.layers.fully_connected(
            output_rnn, 1, activation_fn=None)
    #if not is_training:
        #return predictions, None, None
    return predictions
    '''
    #正确的标签
    #loss=tf.reduce_mean(tf.square(tf.reshape(predictions,[-1])-tf.reshape(y, [-1])))
    '''
#设计验证函数
def run_verify(sess, verify_x, verify_y):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices(verify_x)
    ds = ds.batch(1)
    X = ds.make_one_shot_iterator().get_next()
    
    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("{}".format(variable_sco), reuse=True):
        prediction = lstm(X,False)
    
    # 将预测结果存入一个数组。
    predictions = []
    for i in range(len(verify_x)):
        p = sess.run(prediction)
        pred=p.reshape((-1))
        predictions.extend(pred)

    #反标准化
    verify_predict=np.array(predictions)*verify_std[6]+verify_mean[6]
    verify_y=np.array(verify_y)*verify_std[6]+verify_mean[6]
    #acc=np.average(np.abs(verify_predict-verify_y[:len(verify_predict)])/verify_y[:len(verify_predict)])  #偏差程度
    acc=np.sum(np.abs(verify_predict-verify_y))/np.sum(verify_y)  #偏差程度
    #print("The accuracy of this predict:",acc)
    return acc
    
#可以开始训练了
#X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
#y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])

batch_index,train_x,train_y,train_num=get_train_data(train_begin=0,train_end=19200)
verify_mean,verify_std,verify_x,verify_y=get_verify_data(verify_begin=19200,verify_end=21600)
ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
ds = ds.repeat().shuffle(1000).batch(batch_size) #shuffle(buffersize=1000)文件流的缓存区大小
X,y = ds.make_one_shot_iterator().get_next()

with tf.variable_scope("{}".format(variable_sco)):
    predictions = lstm(X,True)
'''加入'''
#定义存储训练次数的变量，此变量不适用移动平均，不进入训练trainable=False
global_step=tf.Variable(0,trainable=False)
# 计算损失函数。
loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
#正确的标签
#loss=tf.reduce_mean(tf.square(tf.reshape(predictions,[-1])-tf.reshape(y, [-1])))

#设置学习率
learning_rate = tf.train.exponential_decay(
        lr_base,
        global_step,
        train_num/batch_size,
        lr_decay,
        staircase=False) #staircase=True 学习率梯行下降
# 创建模型优化器并得到优化步骤。
train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
'''加入''' 
saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #开始训练
    best_acc=10
    for i in range(train_steps+1):
        '''
        for step in range(len(batch_index)-1):
            _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],y:train_y[batch_index[step]:batch_index[step+1]]})
        '''
        _,loss_=sess.run([train_op,loss])
        if i % 100==0:
            print("Number of iterations:",i," loss:",loss_)
            #在这里加入验证过程，然后保存。
            acc_=run_verify(sess,verify_x,verify_y)
            if acc_<best_acc:
                print("Update best_acc：",acc_)
                best_acc=acc_
                print("Save model in: {}".format(model_save_path))
                saver.save(sess,os.path.join(model_save_path,model_name))
    print("The train has finished")
    #saver.save(sess,'model_save3\\modle.ckpt')
  
#定义测试函数
def run_test(sess, test_x, test_y):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices(test_x)
    ds = ds.batch(1)
    X = ds.make_one_shot_iterator().get_next()
    
    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("{}".format(variable_sco), reuse=True):
        prediction = lstm(X,False)
    
    # 将预测结果存入一个数组。
    predictions = []
    for i in range(len(test_x)):
        p = sess.run(prediction)
        pred=p.reshape((-1))
        predictions.extend(pred)

    #反标准化
    test_predict=np.array(predictions)*test_std[6]+test_mean[6]
    test_y=np.array(test_y)*test_std[6]+test_mean[6]
    #acc=np.average(np.abs(verify_predict-verify_y[:len(verify_predict)])/verify_y[:len(verify_predict)])  #偏差程度
    acc=np.sum(np.abs(test_predict-test_y))/np.sum(test_y)  #偏差程度
    print("The accuracy of test_dataset:",acc)
    '''
    #以折线图表示结果
    plt.figure()
    plt.plot(list(range(len(test_predict))), test_predict, color='b',)
    plt.plot(list(range(len(test_y))), test_y,  color='r')
    plt.show()
    #return acc
    '''
    
saver=tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    #参数恢复
    module_file = tf.train.latest_checkpoint(model_save_path)
    saver.restore(sess, module_file)
    test_mean,test_std,test_x,test_y=get_test_data(test_begin=21600)
    run_test(sess,test_x,test_y)

