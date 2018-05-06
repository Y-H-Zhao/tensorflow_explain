# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:31:48 2018

@author: ZYH
"""

import tensorflow as tf
import pandas as pd
#import math
import numpy as np


f=open('sales_data08.csv')
df=pd.read_csv(f)     #读入销售数据
#data=df.iloc[:,:13].values  #取第1-13列
original_features=df.iloc[:,:13].values #取第1-13列
sales=df.iloc[:,13].values  #预测数据
#标准化
mean=np.mean(original_features,axis=0)
std=np.std(original_features,axis=0)
features=(original_features-mean)/std  #标准化
#sales=(original_sales-np.mean(original_sales,axis=0))/np.std(original_sales,axis=0)
#此处可以标准化，相应在lstm预测中省去标准化过过程
n=len(features)


def batch_generator(features, batch_size=100, n_epochs=10):
    """
    Batch generator for the iris dataset
    """

    # Generate batches
    for epoch in range(n_epochs):
        start_index = 0
        while start_index != -1:
            # Calculate the end index of the batch to generate
            end_index = start_index + batch_size if start_index + batch_size < n else -1

            yield features[start_index:end_index]

            start_index = end_index


# Auto Encoder
class TF_AutoEncoder:
    def __init__(self, features,lr=0.01, steps=100, dtype=tf.float32):
        self.features = features
        self.dtype = dtype
        self.steps = steps
        self.lr=lr

        self.encoder = dict()

    def fit(self, n_dimensions):
        graph = tf.Graph()
        with graph.as_default():

            # Input variable
            X = tf.placeholder(self.dtype, shape=(None, self.features.shape[1]))

            # Network variables
            encoder_weights = tf.Variable(tf.random_normal(shape=(self.features.shape[1], n_dimensions)))
            encoder_bias = tf.Variable(tf.zeros(shape=[n_dimensions]))

            decoder_weights = tf.Variable(tf.random_normal(shape=(n_dimensions, self.features.shape[1])))
            decoder_bias = tf.Variable(tf.zeros(shape=[self.features.shape[1]]))

            # Encoder part
            encoding = tf.nn.sigmoid(tf.add(tf.matmul(X, encoder_weights), encoder_bias))
            #encoding = tf.nn.relu(tf.add(tf.matmul(X, encoder_weights), encoder_bias))

            # Decoder part
            predicted_x = tf.nn.sigmoid(tf.add(tf.matmul(encoding, decoder_weights), decoder_bias))

            # Define the cost function and optimizer to minimize squared error
            cost = tf.losses.mean_squared_error(labels=X, predictions=predicted_x)
            #cost = tf.reduce_mean(tf.pow(tf.subtract(predicted_x, X), 2))
            optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)

        with tf.Session(graph=graph) as session:
            # Initialize global variables
            session.run(tf.global_variables_initializer())

            for i in range(self.steps):
                for batch_x in batch_generator(self.features):
                    cost_, _ = session.run([cost, optimizer],feed_dict={X: batch_x})
                if i % 10==0:
                    print("均方误差为：",cost_)
                #self.encoder['weights'], self.encoder['bias'], _ = session.run([encoder_weights, encoder_bias, optimizer],feed_dict={X: batch_x})                                                           
            self.encoding_=session.run(encoding,feed_dict={X:self.features})

    def reduce(self):
        #return np.add(np.matmul(self.features, self.encoder['weights']), self.encoder['bias'])
        return self.encoding_

    



# Create an instance and encode
tf_ae = TF_AutoEncoder(features,steps=50)
tf_ae.fit(n_dimensions=6)
auto_encoded = tf_ae.reduce()
#print(auto_encoded)
#dem_loss_sales=np.insert(auto_encoded, i, values=sales, axis=1) i表示插在第i列
dem_loss_sales=np.c_[auto_encoded,sales]

dem_loss_sales_data=pd.DataFrame(dem_loss_sales)
dem_loss_sales_data.to_csv('dem_loss_sales_sig.csv',index=False)


