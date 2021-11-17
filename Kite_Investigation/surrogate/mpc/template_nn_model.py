#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import do_mpc

import tensorflow as tf
from tensorflow import keras

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


def template_nn_model(symvar_type='MX', nn_model=None, nn_model_aux=None):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    a_train = nn_model_aux['a_train']
    in_scaler = nn_model_aux['in_scaler']
    out_scaler = nn_model_aux['out_scaler']
    C = nn_model_aux['C']


    # States struct (optimization variables):
    theta = model.set_variable('_x',  'theta') # zenith angle
    phi = model.set_variable('_x',  'phi') # azimuth angle
    psi = model.set_variable('_x',  'psi') # orientation kite

    # Input struct (optimization variables):
    u_tilde = model.set_variable('_u',  'u_tilde')

    # Fixed parameters:
    E_0 = model.set_variable('_p',  'E_0')
    v_0 = model.set_variable('_p', 'v_0')


    x = vertcat(theta, phi, psi)
    p = vertcat(E_0, v_0)

    nn_in = vertcat(u_tilde, p, x)
    nn_in_scaled = (nn_in-in_scaler.mean_)/in_scaler.scale_

    activations = keras2casadi(nn_model, nn_in_scaled)

    a_calc = (activations[-2]).T
    a_calc_sym = model.set_variable(var_type='_z', var_name='a_calc', shape=a_calc.shape)

    model.set_alg('a_calc', a_calc-a_calc_sym)

    # Last layer:
    w = nn_model.layers[-1].get_weights()

    nn_out_scaled = a_calc_sym.T@w[0] + w[1].reshape(1,-1)
    nn_out = nn_out_scaled.T*out_scaler.scale_+out_scaler.mean_

    T_F = nn_out[0]
    height_kite = nn_out[1]
    dx = nn_out[2:]

    x_next = x + dx

    # Differential equations
    model.set_rhs('theta', x_next[0])
    model.set_rhs('phi', x_next[1])
    model.set_rhs('psi', x_next[2])


    model.set_expression('T_F', T_F)
    model.set_expression('height_kite', height_kite)

    bll_trust = a_calc_sym.T@C@a_calc_sym

    model.set_expression('bll_trust',  bll_trust)

    # Build the model
    model.setup()

    return model
