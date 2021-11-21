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

sys.path.append('../')

import gp_tools


def template_gp_model(symvar_type='MX', gp_dict=None):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    in_scaler = gp_dict['in_scaler']
    out_scaler = gp_dict['out_scaler']
    X_train = gp_dict['X_train']
    Y_train = gp_dict['Y_train']
    l = gp_dict['l']
    alpha = gp_dict['alpha']


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

    gp_in = vertcat(u_tilde, p, x)
    gp_in_scaled = (gp_in-in_scaler.mean_)/in_scaler.scale_

    n_samples_gpr = X_train.shape[0]
    Sigma = alpha*np.eye(n_samples_gpr)
    gp = gp_tools.GP(gp_tools.RBF(l=l))
    gp.fit(X_train,Y_train, Sigma)

    gp_out_scaled, v = gp.predict(gp_in_scaled.T)
    gp_out = gp_out_scaled.T*out_scaler.scale_+out_scaler.mean_

    T_F = gp_out[0]
    height_kite = gp_out[1]
    dx = gp_out[2:]

    x_next = x + dx

    # Differential equations
    model.set_rhs('theta', x_next[0])
    model.set_rhs('phi', x_next[1])
    model.set_rhs('psi', x_next[2])

    model.set_expression('T_F', T_F)
    model.set_expression('height_kite', height_kite)

    model.set_expression('bll_trust',  v)

    # Build the model
    model.setup()

    return model
