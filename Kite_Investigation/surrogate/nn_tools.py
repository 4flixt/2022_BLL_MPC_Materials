import numpy as np
import tensorflow as tf
from casadi import *

def get_keras_model(architecture, activation='relu'):
    """Function to obtain a Keras model

    Model is defined by the parameter ``architecture`` which must be a list.

    Example:

        architecture = [1, 5, 5, 1]

    Returns a NN with one input, two hidden layers (each with 5 neurons) and one output.

    All hidden layer (except for the output layer) have the same activation function.
    """
    inputs = tf.keras.Input(shape=architecture[0], name='input')
    outputs = [inputs]

    for n_l in range(1, len(architecture)-1):
        outputs.append(tf.keras.layers.Dense(architecture[n_l], 
                                 activation=activation,
                                 name='hidden_{}'.format(n_l))(outputs[n_l-1]))

    outputs.append(tf.keras.layers.Dense(architecture[-1],
                                      name='output')(outputs[-1]))

    model = tf.keras.Model(inputs=inputs, outputs=outputs[-1])
    activation_model = tf.keras.Model(inputs=inputs, outputs=outputs[-2])

    return model, activation_model

def keras2casadi(nn_model, input):
    a = [input.T]

    for layer in nn_model.layers:
        w = layer.get_weights()
        if w:
            l = a[-1]@w[0] + w[1].reshape(1,-1)
            if layer.activation == tf.keras.activations.tanh:
                a.append(tanh(l))
            if layer.activation == tf.keras.activations.linear:
                a.append(l)

    return a

class NNPredictModel:
    def __init__(self, nn, act_nn, a_train, in_scaler, out_scaler, x0, sig_e=1, sig_w=1e3):
        self.nn = nn
        self.act_nn = act_nn
        self.in_scaler = in_scaler
        self.out_scaler = out_scaler

        Sigma_W = np.eye(a_train.shape[1])*1/sig_w**2
        Sigma_E = np.eye(a_train.shape[0])*1/sig_e**2

        self.C = np.linalg.inv(a_train.T@Sigma_E@a_train+Sigma_W)

        self.reset(x0)

    def reset(self, x0):
        self._x = [x0.reshape(1,-1)]
        self._aux = []
        self._u = []
        self._v = []

    @property
    def x(self):
        return np.concatenate(self._x, axis=0)

    @x.setter
    def x(self, value):
        None

    @property
    def u(self):
        return np.concatenate(self._u, axis=0)

    @u.setter
    def u(self, value):
        None

    @property
    def v(self):
        return np.concatenate(self._v, axis=0)

    @v.setter
    def v(self, value):
        None

    @property
    def aux(self):
        return np.concatenate(self._aux, axis=0)

    @aux.setter
    def aux(self, value):
        None

    def make_step(self, u, p):
        x = self._x[-1]
        u = u.reshape(1,-1)
        p = p.reshape(1,-1)

        nn_in = np.concatenate((u,p,x), axis=1)
        nn_in_scaled = self.in_scaler.transform(nn_in)

        nn_out_scaled = self.nn(nn_in_scaled).numpy()
        a_hat = self.act_nn(nn_in_scaled)
        if not isinstance(a_hat, np.ndarray):
            a_hat = a_hat.numpy()

        v = a_hat@self.C@a_hat.T

        nn_out = (self.out_scaler.inverse_transform(nn_out_scaled)).reshape(1,-1)
        aux = nn_out[:,:2]
        x_next = nn_out[:,2:]+x


        self._v.append(v)
        self._x.append(x_next)
        self._aux.append(aux)
        self._u.append(u)
