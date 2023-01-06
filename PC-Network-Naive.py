import numpy as np

# PC Network, numpy version
# Version 1.1 (2023-01-02)
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

    def w_init():
        
        # TODO Diego: Don't really understand what this does and how to implement it.
        
        # w,b are cell arrays. Their i^th cell elements correspond the the i^th
        # weights/biases of the network.
        # return w, b

        # function [w,b] = w_init(params)
        # %w,b are cell arrays. Their i^th cell elements correspond the the i^th
        # %weights/biases of the network. For example the first cell element
        # %corresponds to the first weights/biases that convey information form input
        # %to first hidden layer of network.
        # %params is a structure of parameters
        # %type is the type of non-linearity
        # %n_layers is number of layers in network
        # %neurons is a vector with i^th element being the number of neurons in i^th
        # %layer

        # type = params.type;
        # n_layers = params.n_layers;
        # w = cell(n_layers,1);
        # b = cell(n_layers,1);
        # neurons=params.neurons;

        # for i = 1:n_layers-1
        #     norm_b = 0;
        #     switch type
        #         case 'lin'
        #             norm_w = sqrt(1/(neurons(i+1) + neurons(i))) ;
        #         case 'tanh'
        #             norm_w = sqrt(6/(neurons(i+1) + neurons(i))) ;
        #         case 'logsig'
        #             norm_w = 4 * sqrt(6/(neurons(i+1) + neurons(i))) ;
        #         case 'reclin'
        #             norm_w = sqrt(1/(neurons(i+1) + neurons(i))) ;
        #             norm_b = 0.1;
        #     end    
        #     w{i} = unifrnd(-1,1,neurons(i+1),neurons(i)) * norm_w ;
        #     b{i} = zeros(neurons(i+1),1) + norm_b * ones(neurons(i+1),1) ;  
        # end
        pass


    def rms_error(y, f):
        return np.sqrt(np.mean((y - f) ** 2))
        

    def test(input, output, w, b, f):
        iterations = len(input)
        x_output = np.zeros(output.shape)
        x = np.array(num_layers)

        for its in range(iterations):
            # make prediction
            x[1] = input[:, its]
            for ii in range(2, num_layers):
                x[ii] = w[ii-1] @ (f(x[ii-1])) + b[ii-1]
            x_output[:, its] = x[num_layers]
        # calculate errors
        rmse = rms_error(output, x_output)
        return rmse

    def main():
        # TODO: I think this is where we should put the code to run the algorithm.
        # TODO: I think we should also add a function to plot the results.
        
        # % example code

        # %Contrary to our paper, we run the network in the 'opposite direction'
        # %here as it makes the code clearer to follow.
        # %so the input layer here is layer 1, and the output layer is layer l_max
        # %i.e the first set of weights W_1, would take the input layer values and propagate
        # %it to layer 2
        

        # We train the network to solve the XOR problem
        # https://analyticsindiamag.com/xor-problem-with-neural-networks-an-explanation-for-beginners/


        # for run = 1:run_num;
        #     [w_pc, b_pc] = w_init(params); % get weights and biases parameters
        #     w_ann=w_pc;
        #     b_ann=b_pc;
        #     counter =1;
        #     [rms_error_pc(run,counter)] = test(sin,sout,w_pc,b_pc,params); %test pc
        #     [rms_error_ann(run,counter)] = test(sin,sout,w_ann,b_ann,params); %test ann 
            
        #     %learn
        #     for epoch = 1:params.epochs
        #         params.epoch_num = epoch;
        #         [w_pc,b_pc] = learn_pc(sin,sout,w_pc,b_pc,params); %train pc
        #         [w_ann,b_ann] = learn_ann(sin,sout,w_ann,b_ann,params); %train ann
                
        #         if (epoch/params.epochs)*(params.epochs/plotevery) == ceil((epoch/params.epochs)*(params.epochs/plotevery));
        #             disp(['run=',num2str(run),' it=',num2str(epoch)]);
        #             counter = counter+1;
        #             [rms_error_pc(run,counter)] = test(sin,sout,w_pc,b_pc,params); %test pc
        #             [rms_error_ann(run,counter)] = test(sin,sout,w_ann,b_ann,params); %test ann 
        #         end
        #     end
        # end
        # leg={'run1','run2','run3','run4'};

        # figure('color',[1 1 1]);
        # subplot(1,2,1);
        # plot(0:50:params.epochs,rms_error_pc')
        # xlabel('Iterations')
        # ylabel('RMSE')
        # title('Predictive coding')
        # legend(leg)
        # set(gca,'xlim',[0 params.epochs]);
        # subplot(1,2,2);
        # plot(0:50:params.epochs,rms_error_ann')
        # xlabel('Iterations')
        # ylabel('RMSE')
        # title('Artificial NN')
        # legend(leg)
        # set(gca,'xlim',[0 params.epochs]);
        pass