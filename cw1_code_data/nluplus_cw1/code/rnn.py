# coding: utf-8
from rnnmath import *
from model import Model, is_param, is_delta

class RNN(Model):
    '''
    This class implements Recurrent Neural Networks.
    
    You should implement code in the following functions:
        predict				->	predict an output sequence for a given input sequence
        acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
        acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
        acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
        acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions

    Do NOT modify any other methods!
    Do NOT change any method signatures!
    '''
    
    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        '''
        initialize the RNN with random weight matrices.
        
        DO NOT CHANGE THIS
        
        vocab_size		size of vocabulary that is being used
        hidden_dims		number of hidden units
        out_vocab_size	size of the output vocabulary
        '''

        super().__init__(vocab_size, hidden_dims, out_vocab_size)

        # matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
        with is_param():
            self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
            self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
            self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)

        # matrices to accumulate weight updates
        with is_delta():
            self.deltaU = np.zeros_like(self.U)
            self.deltaV = np.zeros_like(self.V)
            self.deltaW = np.zeros_like(self.W)

    def predict(self, x):
        '''
        predict an output sequence y for a given input sequence x
        
        x	list of words, as indices, e.g.: [0, 4, 2]
        
        returns	y,s
        y	matrix of probability vectors for each input word
        s	matrix of hidden layers for each input word
        
        '''
        
        # matrix s for hidden states, y for output states, given input x.
        # rows correspond to times t, i.e., input words
        # s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )

        s = np.zeros((len(x) + 1, self.hidden_dims))
        y = np.zeros((len(x), self.out_vocab_size))
        for t in range(len(x)):
            ##########################
            # --- your code here --- #
            # one-hot vector encoding
            x_one_hot = np.zeros(self.vocab_size , dtype = int)
            x_one_hot[x[t]] = 1
            net_in = np.dot(self.V,x_one_hot) + np.dot(self.U,s[t-1])
            s[t] = sigmoid(net_in)
            net_out = np.dot(self.W,s[t])
            y[t] = softmax(net_out)
            ##########################

        return y, s
    
    def acc_deltas(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x	list of words, as indices, e.g.: [0, 4, 2]
        d	list of words, as indices, e.g.: [4, 2, 3]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)
        
        no return values
        ''' 
        ##########################
        for t in reversed(range(len(x))):
            # Compute output error
            d_one_hot = make_onehot(d[t], self.out_vocab_size)
            delta_out = d_one_hot - y[t]

            # Update deltaW
            self.deltaW += np.outer(delta_out, s[t])

            # Compute hidden state error
            delta_in = np.dot(self.W.T, delta_out) * s[t] * (1 - s[t])

            # Compute one-hot encoding for input
            x_one_hot = make_onehot(x[t], self.vocab_size)

            self.deltaV += np.outer(delta_in, x_one_hot)
            self.deltaU += np.outer(delta_in, s[t - 1])
        ##########################

 
    def acc_deltas_np(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)
        
        no return values
        '''
        pass

        ##########################
        # --- your code here --- #
        t = len(x) - 1
        d_one_hot = make_onehot(d[0], self.vocab_size)
        x_one_hot = make_onehot(x[t], self.vocab_size)

		# the error at the output layer
        delta_out = d_one_hot - y[t]
        # computes the sigmoid derivative for backpropagation
        derivative_net_in = s[t] * (1 - s[t])
        # backpropagate the error to the hidden layer
        delta_net_in = np.dot(self.W.T,delta_out) * derivative_net_in
        #update output weights
        self.deltaW += np.outer(delta_out, s[t])
        #update input weights
        self.deltaV += np.outer(delta_net_in, x_one_hot)
        #update recurrent weights
        self.deltaU += np.outer(delta_net_in, s[t - 1])
        ##########################
        
    def acc_deltas_bptt(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x		list of words, as indices, e.g.: [0, 4, 2]
        d		list of words, as indices, e.g.: [4, 2, 3]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT
        
        no return values
        '''
        ##########################
        # --- your code here --- #
        for t in reversed(range(len(x))):
            # Compute output error
            d_one_hot = make_onehot(d[t], self.out_vocab_size)
            delta_out = d_one_hot - y[t]
            

            # Update deltaW
            self.deltaW += np.outer(delta_out, s[t])

            # Compute hidden state error
            delta_in = np.dot(self.W.T, delta_out) * s[t] * (1 - s[t])

            # Compute one-hot encoding for input
            x_one_hot = make_onehot(x[t], self.vocab_size)

            self.deltaV += np.outer(delta_in, x_one_hot)
            self.deltaU += np.outer(delta_in, s[t - 1])

            # Backpropagate error further back in time
            for back_step in range(1, steps + 1):
                if t >= back_step:
                    x_one_hot = make_onehot(x[t - back_step], self.vocab_size)

                    delta_in = np.dot(self.U.T, delta_in) * s[t - back_step] * (1 - s[t - back_step])

                    self.deltaV += np.outer(delta_in, x_one_hot)
                    self.deltaU += np.outer(delta_in, s[t - back_step - 1])
        ##########################



    def acc_deltas_bptt_np(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT
        
        no return values
        '''
        pass

        ##########################
        # --- your code here --- #
        t = len(x) - 1
        d_one_hot = make_onehot(d[0], self.vocab_size)
        x_one_hot = make_onehot(x[t], self.vocab_size)

        delta_out = d_one_hot - y[t]
        delta_in = np.dot(self.W.T, delta_out) * s[t] * (1 - s[t])

        self.deltaW += np.outer(delta_out, s[t])
        self.deltaV += np.outer(delta_in, x_one_hot)
        self.deltaU += np.outer(delta_in, s[t - 1])

        for back_step in range(1, steps + 1):
            if t >= back_step:
                x_one_hot = make_onehot(x[t - back_step], self.vocab_size)

                delta_in = np.dot(self.U.T, delta_in) * s[t - back_step] * (1 - s[t - back_step])

                self.deltaV += np.outer(delta_in, x_one_hot)
                self.deltaU += np.outer(delta_in, s[t - back_step - 1])

        ##########################