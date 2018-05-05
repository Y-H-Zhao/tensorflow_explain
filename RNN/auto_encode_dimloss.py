# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:31:48 2018

@author: ZYH
"""

import tensorflow as tf
import pandas as pd
#import math
from sklearn import datasets
#from sklearn.manifold import TSNE
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns

#iris_dataset = datasets.load_iris() 鸢尾花数据

f=open('sales_data08.csv')
df=pd.read_csv(f)     #读入股票数据
#data=df.iloc[:,:13].values  #取第1-13列
features=df.iloc[:,:13].values #取第1-13列
sales=df.iloc[:,13].values  #预测数据
n=len(features)
'''
# Mix the data before training
n = len(iris_dataset.data)
random_idx = np.random.permutation(n)
features= iris_dataset.data[random_idx]
'''

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
    def __init__(self, features,dtype=tf.float32):
        self.features = features
        self.dtype = dtype

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

            # Decoder part
            predicted_x = tf.nn.sigmoid(tf.add(tf.matmul(encoding, decoder_weights), decoder_bias))

            # Define the cost function and optimizer to minimize squared error
            cost = tf.reduce_mean(tf.pow(tf.subtract(predicted_x, X), 2))
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session(graph=graph) as session:
            # Initialize global variables
            session.run(tf.global_variables_initializer())

            for batch_x in batch_generator(self.features):
                self.encoder['weights'], self.encoder['bias'], _ = session.run([encoder_weights, encoder_bias, optimizer],
                                                                            feed_dict={X: batch_x})

    def reduce(self):
        return np.add(np.matmul(self.features, self.encoder['weights']), self.encoder['bias'])
    



# Create an instance and encode
tf_ae = TF_AutoEncoder(features)
tf_ae.fit(n_dimensions=6)
auto_encoded = tf_ae.reduce()
#print(auto_encoded)
#dem_loss_sales=np.insert(auto_encoded, i, values=sales, axis=1) i表示插在第i列
dem_loss_sales=np.c_[auto_encoded,sales]

dem_loss_sales_data=pd.DataFrame(dem_loss_sales)
dem_loss_sales_data.to_csv('dem_loss_sales.csv',index=False)
