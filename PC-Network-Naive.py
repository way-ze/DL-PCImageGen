import numpy as np

# PC Network, numpy version
# Version 1.0 (2022-12-29)
# based on the the repo https://github.com/ModelDBRepository/218084

# params = {
#     'num_layers': 3,
#     'num_neurons': 10,
#     'num_epochs': 100,
#     'learning_rate': 0.2,
#     'decay_rate': 0.01,
#     'it_max': 100,
#     'beta': 0.01,
#     'var': np.ones((1, 3])),
#     'type': 'tanh'
# }

num_neurons = np.array([2, 5, 1])
num_layers = num_neurons.size
num_epochs = 500
learning_rate = 0.2
decay_rate = 0
it_max = 100
beta = 0.2 # euler integration constant
var = np.ones((1, num_layers))


class PCNet():
    def __init__(self):
        pass
        # super().__init__()
    def learn_pc(f, input, output, w, b):
        # learn the PC network
        # w, b = weights, biases
        # input = np array, input data, of dimension (input_dim, num_samples)
        # output = np array, output data, of dimension (output_dim, num_samples)
        # params = parameters of the PC network
        #        = [num_layers, num_neurons, num_epochs, learning_rate, decay_rate]
        
        iterations = len(input)
        for i in range(iterations):
            x = np.array((num_layers, 1))
            grad_b = np.array(b.shape)
            grad_w = np.array(w.shape)

            # organise data into cell arrays
            x[1] = input[:, i]
            x_out = output[:, i]
            
            # make a prediction
            for ii in range(2, num_layers):
                x[ii] = w[ii-1] @ (f(x[ii-1])) + b[ii-1]
            
            # infer
            x[num_layers] = x_out
            x, e, _ = predict_pc(x, w, b)
            
            # calculate gradients
            for ii in range(num_layers):
                grad_b[ii] = x_out * e[ii+1]
                grad_w[ii] = x_out * e[ii+1] @ f(x[ii]) - decay_rate * w[ii]
            
            # update weights
            for ii in range(num_layers):
                w[ii] = w[ii] + learning_rate * grad_w[ii]
                b[ii] = b[ii] + learning_rate * grad_b[ii]
        return w, b
    def predict_pc(x, w, b):
        # w,b - the weights and biases
        # x - Variable nodes: First cell is input layer. Last cell is output layer
        # e - Error nodes: First cell empty. Last cell is output layer
        e = np.array(num_layers)
        f_n = np.array(num_layers)
        f_p = np.array(num_layers)

        # calculate initial errors
        for ii in range(2, num_layers):
            f_n[ii-1], f_p[ii-1] = f_b(x[ii-1])
            e[ii] = (x[ii] - w[ii-1] @ f_n[ii-1] - b[ii-1]) / var[ii]
        
        for i in range(it_max):
            # update variable nodes
            for ii in range(2, num_layers-1):
                g = (w[ii].T @ e[ii+1]) * f_p[ii]
                x[ii] = x[ii] + beta * (-e[ii] + g)
            
            # calculate errors
            for ii in range(2, num_layers):
                f_n[ii-1], f_p[ii-1] = f_b(x[ii-1])
                e[ii] = (x[ii] - w[ii-1] @ f_n[ii-1] - b[ii-1]) / var[ii]
        return x, e, it_max
    
    def f_b(x):
        return 1 - np.tanh(x) ** 2
