'''
滑动平均方法，使变量变化的更平缓，一般为拟合的参数，通过这样的设置，使模型在测试数据上更健壮robust。
下面这段程序，来模拟一下变量的变化过程
step(num_upsteps)用于设置动态的衰减率
衰减率=min{衰减率，(1+num_upsteps)/(10+num_upsteps)}
'''
import tensorflow as tf
#定义变量及滑动平均类
v1 = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(0.99, step)
maintain_averages_op = ema.apply([v1]) 

#查看不同迭代中变量取值的变化。
with tf.Session() as sess:
    
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([v1, ema.average(v1)]))
    
    # 更新变量v1的取值
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    
    # 更新step和v1的取值
    sess.run(tf.assign(step, 10000))  
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    
    # 更新一次v1的滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))import tensorflow as tf
'''
[0.0, 0.0]
[5.0, 4.5]
[10.0, 4.5549998]
[10.0, 4.6094499]
'''
