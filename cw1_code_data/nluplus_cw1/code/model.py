import abc
import numpy as np
from collections import OrderedDict



class is_param:

    def __enter__(self):
        Model._is_param = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        Model._is_param = False

class is_delta:

    def __enter__(self):
        Model._is_delta = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        Model._is_delta = False

class Model(object):
    '''
    This class provides an abstract of implementation for both RNNs and GRUs. It ensures that the trainer runs for both
    types of models.

    DO NOT CHANGE THIS CLASS!

    '''

    _is_param = False
    _is_delta = False

    def __setattr__(self, name, value):
        if Model._is_param:
            _parameters = self.__dict__['_parameters']
            _parameters[name] = value
        elif Model._is_delta:
            _deltas = self.__dict__['_deltas']
            _deltas[name] = value
        else:
            super().__setattr__(name, value)


    def __getattr__(self, name):

        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_deltas' in self.__dict__:
            _deltas = self.__dict__['_deltas']
            if name in _deltas:
                return _deltas[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        self._parameters = OrderedDict()
        self._deltas = OrderedDict()
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self.out_vocab_size = out_vocab_size

    @abc.abstractmethod
    def predict(self, x) -> (np.ndarray, np.ndarray):
        '''
        predict an output sequence y for a given input sequence x

        x	list of words, as indices, e.g.: [0, 4, 2]

        returns	y,s
        y	matrix of probability vectors for each input word
        s	matrix of hidden layers for each input word

        '''

        pass

    @abc.abstractmethod
    def acc_deltas(self, x, d, y, s) -> None:
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
        pass

    @abc.abstractmethod
    def acc_deltas_np(self, x, d, y, s) -> None:
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

    @abc.abstractmethod
    def acc_deltas_bptt(self, x, d, y, s, steps) -> None:
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

        pass

    @abc.abstractmethod
    def acc_deltas_bptt_np(self, x, d, y, s, steps):
        '''
        accumulate updates

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


    def reset_deltas(self) -> None:
        '''
        resets delta values to zero

        no return values
        '''

        for delta in self._deltas.values():
            delta.fill(0.0)

    def scale_gradients_for_batch(self, batch_size) -> None:
        for delta in self._deltas.values():
            delta /= batch_size

    def apply_deltas(self, learning_rate) -> None:
        '''
        update the RNN's weight matrices with corrections accumulated over some training instances

        DO NOT CHANGE THIS

        learning_rate	scaling factor for update weights
        '''

        for param, delta in zip(self._parameters.values(), self._deltas.values()):
            param += learning_rate*delta

        self.reset_deltas()

    def save_params(self) -> None:

        self._best_params = OrderedDict()

        for name, parameter in self._parameters.items():
            self._best_params[name] = parameter.copy()

    def set_best_params(self) -> None:

        for name, parameter in self._best_params.items():
            self._parameters[name] = parameter