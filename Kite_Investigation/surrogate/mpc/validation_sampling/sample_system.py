import sys
sys.path.append('../../../../../do-mpc_fork/')
import do_mpc
import tensorflow as tf
from tensorflow import keras

import time
import os

import numpy as np
import pdb
import multiprocessing as mp
from functools import partial
from do_mpc.tools import load_pickle


sys.path.append('../')
from template_nn_model import template_nn_model
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

np.random.seed(99)


""" Load NN """

export_name = 'S01_NN_M01'
export_path = '../../nn_models/{}/'

nn_model_aux = do_mpc.tools.load_pickle(export_path.format(export_name) + export_name + '_aux.pkl')
nn_model = keras.models.load_model(export_path.format(export_name))

""" Settings """

trust_region_cons = False
trust_region_ub = 0.02

w_ref = 10
E_0 = 6
h_min = 100

def info():
    print(mp.current_process())
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def sample_function(x0, seed):
    info()
    surrogate_model = template_nn_model('SX', nn_model, nn_model_aux)
    model = template_model()
    mpc = template_mpc(surrogate_model, h_min, trust_region_cons, w_ref, E_0, trust_region_ub)
    simulator = template_simulator(model, w_ref, E_0)
    estimator = do_mpc.estimator.StateFeedback(model)


    # set initial values and guess
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()

    n_steps = 300

    for k in range(n_steps):
        try:
            u0 = mpc.make_step(x0)
            if mpc.data['success'][-1] == False:
                break
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)
        except:
            break

    return [simulator.data, mpc.data]



def main():
    plan = load_pickle('./kite_validation_01/kite_validation_01.pkl')

    sampler = do_mpc.sampling.Sampler(plan)
    sampler.set_param(print_progress = False)
    sampler.set_param(overwrite = True)
    if trust_region_cons:
        sampler.data_dir = f'./kite_validation_01/{export_name}/with_bll_cons/'
    else:
        sampler.data_dir = f'./kite_validation_01/{export_name}/wo_bll_cons/'


    sampler.set_sample_function(sample_function)

    with mp.Pool(processes=8) as pool:
        p = pool.map(sampler.sample_idx, list(range(sampler.n_samples)))

    print('done')



if __name__ == '__main__':
    main()
