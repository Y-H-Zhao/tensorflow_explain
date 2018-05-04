import numpy as np
#定义RRN参数
X = [1,2]
state = [0.0, 0.0]
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1
#执行前向传播过程
for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state, w_output) + b_output
    print ("before activation: ", before_activation)
    print ("state: ", state)
    print ("output: ", final_output)
#与其他结构同样道理，可以定义损失函数，进行训练，这就是最基本的RNN结构。 
