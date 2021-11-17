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
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../../../do-mpc_fork/')
import do_mpc
from do_mpc.tools import load_pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator


""" User settings: """
show_animation = True
store_results = False
store_animation = False
trust_region_cons = False

model_type = 'NN' # 'NN' or 'GP'

"""
Get configured do-mpc modules:
"""
np.random.seed(5)

case = 0
plan = load_pickle('./validation_sampling/kite_validation_01/kite_validation_01.pkl')


w_ref = 10
E_0 = 6
h_min = 100


""" Load NN """
if model_type == 'NN':
    from tensorflow import keras
    from template_nn_model import template_nn_model

    export_name = 'S01_NN_M01'
    export_path = '../nn_models/{}/'

    nn_model_aux = do_mpc.tools.load_pickle(export_path.format(export_name) + export_name + '_aux.pkl')
    nn_model = keras.models.load_model(export_path.format(export_name))
    surrogate_model = template_nn_model('SX', nn_model, nn_model_aux)

    trust_region_ub = 0.02
elif model_type == 'GP':
    from template_gp_model import template_gp_model

    export_name = 'S01_M01'
    export_path = '../gp_models/{}/'

    gp_model = do_mpc.tools.load_pickle(export_path.format(export_name) + export_name + '_gp.pkl')
    surrogate_model = template_gp_model('SX', gp_model)

    trust_region_ub = 0.3


"""
Get configured do-mpc modules:
"""


model = template_model()
mpc = template_mpc(surrogate_model, h_min, trust_region_cons, w_ref, E_0, trust_region_ub)
simulator = template_simulator(model, w_ref, E_0)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""
# Derive initial state from bounds:
lb_theta, ub_theta = mpc.bounds['lower','_x','theta'], mpc.bounds['upper','_x','theta']
lb_phi, ub_phi = mpc.bounds['lower','_x','phi'], mpc.bounds['upper','_x','phi']
lb_psi, ub_psi = mpc.bounds['lower','_x','psi'], mpc.bounds['upper','_x','psi']
# with mean and radius:
m_theta, r_theta = (ub_theta+lb_theta)/2, (ub_theta-lb_theta)/2
m_phi, r_phi = (ub_phi+lb_phi)/2, (ub_phi-lb_phi)/2
m_psi, r_psi = (ub_psi+lb_psi)/2, (ub_psi-lb_psi)/2
# How close can the intial state be to the bounds?
# tightness=1 -> Initial state could be on the bounds.
# tightness=0 -> Initial state will be at the center of the feasible range.
tightness = 0.6
theta_0 = m_theta-tightness*r_theta+2*tightness*r_theta*np.random.rand()
phi_0 = m_phi-tightness*r_phi+2*tightness*r_phi*np.random.rand()
psi_0 = m_psi-tightness*r_psi+2*tightness*r_psi*np.random.rand()


x0 = np.array([theta_0, phi_0, psi_0]).reshape(-1,1)

x0 = plan[case]['x0']

mpc.x0 = x0
simulator.x0 =x0
estimator.x0 = x0

mpc.set_initial_guess()

"""
Setup graphic:
"""

fig = plt.figure(figsize=(16,9))
plt.ion()

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
ax2 = plt.subplot2grid((4, 2), (0, 1))
ax3 = plt.subplot2grid((4, 2), (1, 1), sharex=ax2)
ax4 = plt.subplot2grid((4, 2), (2, 1))
ax5 = plt.subplot2grid((4, 2), (3, 1))

# Add lines for constraints
ax1.axvline(mpc.bounds['lower', '_x', 'phi'].full(), linestyle='--', color='r')
ax1.axvline(mpc.bounds['upper', '_x', 'phi'].full(), linestyle='--', color='r' )
ax1.axhline(mpc.bounds['lower', '_x', 'theta'].full(),linestyle='--', color='r' )
ax1.axhline(mpc.bounds['upper', '_x', 'theta'].full(),linestyle='--', color='r' )
ax5.axhline(trust_region_ub, linestyle='--', color='r')

# Create height constraint line:
phi_height_limit = np.linspace(-1,1, 100)
L_tether = 400
theta_height_limit = np.arcsin(h_min/(np.cos(phi_height_limit)*L_tether))
ax1.plot(phi_height_limit, theta_height_limit, linestyle='--', color='r')

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

phi_pred = mpc.data.prediction(('_x', 'phi'))[0]
theta_pred = mpc.data.prediction(('_x', 'theta'))[0]
pred_lines = ax1.plot(phi_pred, theta_pred, color=color[0], linestyle='--', linewidth=1)

phi = mpc.data['_x', 'phi']
theta = mpc.data['_x', 'theta']
res_lines = ax1.plot(phi, theta, color=color[0])

# Height of kite
mpc_graphics.add_line(var_type='_aux', var_name='height_kite', axis=ax2)
mpc_graphics.add_line('_aux','T_F', axis=ax3)
mpc_graphics.add_line('_u','u_tilde',axis=ax4)
mpc_graphics.add_line('_aux','bll_trust', axis=ax5)

ax2.set_ylabel('kite height [m]')
ax3.set_ylabel('T_F')
ax4.set_ylabel('input [-]')
ax5.set_ylabel('trust region')

"""
Run MPC main loop:
"""

n_steps = 300

for k in range(n_steps):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    if show_animation:
        phi_pred = mpc.data.prediction(('_x', 'phi'))[0]
        theta_pred = mpc.data.prediction(('_x', 'theta'))[0]
        for i in range(phi_pred.shape[1]):
            pred_lines[i].set_data(phi_pred[:,i], theta_pred[:,i])
        phi = mpc.data['_x', 'phi']
        theta = mpc.data['_x', 'theta']
        res_lines[0].set_data(phi, theta)
        ax1.relim()
        ax1.autoscale()

        mpc_graphics.plot_results()
        mpc_graphics.plot_predictions()
        mpc_graphics.reset_axes()
        sim_graphics.plot_results()
        sim_graphics.reset_axes()

        plt.show()
        plt.pause(0.01)


if store_animation:
    from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
    def update(t_ind):
        phi_pred = mpc.data.prediction(('_x', 'phi'), t_ind)[0]
        theta_pred = mpc.data.prediction(('_x', 'theta'), t_ind)[0]
        for i in range(phi_pred.shape[1]):
            pred_lines[i].set_data(phi_pred[:,i], theta_pred[:,i])
        phi = mpc.data['_x', 'phi'][:t_ind]
        theta = mpc.data['_x', 'theta'][:t_ind]
        res_lines[0].set_data(phi, theta)
        ax1.relim()
        ax1.autoscale()

        mpc_graphics.plot_results(t_ind)
        mpc_graphics.plot_predictions(t_ind)
        mpc_graphics.reset_axes()
        sim_graphics.plot_results(t_ind)

    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
    gif_writer = ImageMagickWriter(fps=20)

    if trust_region_cons:
        anim_name = 'anim_nn_kite_trust_cons.gif'
    else:
        anim_name = 'anim_nn_kite.gif'
    anim.save(anim_name, writer=gif_writer, dpi=80)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([simulator], 'kite')
