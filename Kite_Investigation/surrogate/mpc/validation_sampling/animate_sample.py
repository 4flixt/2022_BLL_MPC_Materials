import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../../../../do-mpc/')
import do_mpc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time


plan_name = 'kite_validation_01'
model_name = 'S01_NN_M01'
load_name = f'./{plan_name}/{plan_name}.pkl'

plan = do_mpc.tools.load_pickle(load_name)

dh_bll = do_mpc.sampling.DataHandler(plan)
dh_bll.set_param(data_dir = f'./{plan_name}/{model_name}/with_bll_cons/')

dh_std = do_mpc.sampling.DataHandler(plan)
dh_std.set_param(data_dir = f'./{plan_name}/{model_name}/wo_bll_cons/')

for dh in [dh_bll, dh_std]:
    dh.set_post_processing('mpc_data', lambda res: res[1])
    dh.set_post_processing('sim_data', lambda res: res[0])


store_animation = False
trust_region_cons = True
trust_region_ub = 0.02

case = 1

if trust_region_cons:
    mpc_data = dh_bll[case][0]['mpc_data']
    sim_data = dh_bll[case][0]['sim_data']
else:
    mpc_data = dh_std[case][0]['mpc_data']
    sim_data = dh_std[case][0]['sim_data']


"""
Setup graphic:
"""

fig = plt.figure(figsize=(16,9))
plt.ion()

mpc_graphics = do_mpc.graphics.Graphics(mpc_data)
sim_graphics = do_mpc.graphics.Graphics(sim_data)

ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
ax2 = plt.subplot2grid((4, 2), (0, 1))
ax3 = plt.subplot2grid((4, 2), (1, 1), sharex=ax2)
ax4 = plt.subplot2grid((4, 2), (2, 1))
ax5 = plt.subplot2grid((4, 2), (3, 1))

# Add lines for constraints

ax5.axhline(trust_region_ub, linestyle='--', color='r')
phi_height_limit = np.linspace(-1,1, 100)
h_min = 100
L_tether = 400
theta_height_limit = np.arcsin(h_min/(np.cos(phi_height_limit)*L_tether))
ax1.plot(phi_height_limit, theta_height_limit, linestyle='--', color='r')

ax5.axhline(trust_region_ub, linestyle='--', color='r')
phi_height_limit = np.linspace(-1,1, 100)
h_min = 90
L_tether = 400
theta_height_limit = np.arcsin(h_min/(np.cos(phi_height_limit)*L_tether))
ax1.plot(phi_height_limit, theta_height_limit, linestyle='--', color='r')

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

phi_pred = mpc_data.prediction(('_x', 'phi'))[0]
theta_pred = mpc_data.prediction(('_x', 'theta'))[0]
pred_lines = ax1.plot(phi_pred, theta_pred, color=color[0], linestyle='--', linewidth=1)

phi = mpc_data['_x', 'phi']
theta = mpc_data['_x', 'theta']
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


""" Main loop """
n_steps = 300

for k in range(n_steps):

    phi_pred = mpc_data.prediction(('_x', 'phi'),k)[0]
    theta_pred = mpc_data.prediction(('_x', 'theta'),k)[0]
    for i in range(phi_pred.shape[1]):
        pred_lines[i].set_data(phi_pred[:,i], theta_pred[:,i])
    phi = sim_data['_x', 'phi'][:k]
    theta = sim_data['_x', 'theta'][:k]
    res_lines[0].set_data(phi, theta)
    ax1.relim()
    ax1.autoscale()

    mpc_graphics.plot_results(k)
    mpc_graphics.plot_predictions(k)
    mpc_graphics.reset_axes()
    sim_graphics.plot_results()
    sim_graphics.reset_axes()

    plt.show()
    plt.pause(0.01)
