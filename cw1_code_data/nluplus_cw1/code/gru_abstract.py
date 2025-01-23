# coding: utf-8

import abc
from rnnmath import *
from model import Model
from model import is_delta
from model import is_param


class GRUAbstract(Model):
    '''
    This class implements Gated Recurrent Unit backpropagation.

    Do NOT modify anything in this class!
    '''

    def __init__(self, vocab_size, hidden_dims, out_vocab_size):

        super().__init__(vocab_size, hidden_dims, out_vocab_size)

        self.h = None
        self.z = None
        self.r = None

        # matrices to accumulate weight updates
        with is_param():
            self.Ur = np.random.randn(self.hidden_dims, self.hidden_dims) * np.sqrt(0.1)
            self.Vr = np.random.randn(self.hidden_dims, self.vocab_size) * np.sqrt(0.1)
            self.Uz = np.random.randn(self.hidden_dims, self.hidden_dims) * np.sqrt(0.1)
            self.Vz = np.random.randn(self.hidden_dims, self.vocab_size) * np.sqrt(0.1)
            self.Uh = np.random.randn(self.hidden_dims, self.hidden_dims) * np.sqrt(0.1)
            self.Vh = np.random.randn(self.hidden_dims, self.vocab_size) * np.sqrt(0.1)
            self.W = np.random.randn(self.out_vocab_size, self.hidden_dims) * np.sqrt(0.1)

        with is_delta():
            self.deltaUr = np.zeros((self.hidden_dims, self.hidden_dims))
            self.deltaVr = np.zeros((self.hidden_dims, self.vocab_size))
            self.deltaUz = np.zeros((self.hidden_dims, self.hidden_dims))
            self.deltaVz = np.zeros((self.hidden_dims, self.vocab_size))
            self.deltaUh = np.zeros((self.hidden_dims, self.hidden_dims))
            self.deltaVh = np.zeros((self.hidden_dims, self.vocab_size))
            self.deltaW = np.zeros((self.out_vocab_size, self.hidden_dims))

    def predict(self, x):
        self.r = np.zeros((len(x), self.hidden_dims))
        self.z = np.zeros((len(x), self.hidden_dims))
        self.h = np.zeros((len(x), self.hidden_dims))
        s = np.zeros((len(x) + 1, self.hidden_dims))
        y = np.zeros((len(x), self.out_vocab_size))
        for t in range(len(x)):
            y[t], s[t], self.h[t], self.z[t], self.r[t] = self.forward(x[t], s[t - 1])
        return y, s

    def __step__(self, x, t, delta_zero, s):
        delta_one = self.z[t] * delta_zero
        delta_two = s[t - 1] * delta_zero
        delta_three = self.h[t] * delta_zero
        delta_four = -1 * delta_three
        delta_five = delta_two + delta_four
        delta_six = (1 - self.z[t]) * delta_zero
        delta_seven = delta_five * (self.z[t] * (np.ones_like(self.z[t]) - self.z[t]))
        delta_eight = delta_six * (np.ones_like(self.h[t]) - np.square(self.h[t]))
        delta_ten = self.Uh.T @ delta_eight
        delta_eleven = self.Uz.T @ delta_seven
        delta_twelve = delta_ten * self.r[t]
        delta_thirteen = delta_ten * self.h[t - 1]
        delta_fourteen = delta_thirteen * (self.r[t] * (np.ones_like(self.r[t]) - self.r[t]))
        delta_fifteen = self.Ur.T @ delta_fourteen

        self.deltaUr += np.outer(delta_fourteen, s[t - 1])
        self.deltaVr += np.outer(delta_fourteen, make_onehot(x[t], self.vocab_size))
        self.deltaUz += np.outer(delta_seven, s[t - 1])
        self.deltaVz += np.outer(delta_seven, make_onehot(x[t], self.vocab_size))
        self.deltaUh += np.outer(delta_eight, (s[t - 1] * self.r[t]))
        self.deltaVh += np.outer(delta_eight, make_onehot(x[t], self.vocab_size))

        return delta_eleven + delta_twelve + delta_one + delta_fifteen

    def backward(self, x, t, s, delta_output, steps=0):
        self.deltaW += np.outer(delta_output, s[t])
        delta_zero = self.W.T @ delta_output
        delta_next = self.__step__(x, t, delta_zero, s)

        for i in range(1, min(steps + 1, t)):
            delta_next = self.__step__(x, t - i, delta_next, s)

    @abc.abstractmethod
    def forward(self, x, s):
        pass