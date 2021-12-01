import sys
sys.path.append('../../../do-mpc/')
import do_mpc

import time
import os

import numpy as np
import pdb
import multiprocessing as mp
from functools import partial
from do_mpc.tools import load_pickle


sys.path.append('../system/')
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

def info():
    print(mp.current_process())
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def sample_function(x0, w_ref, E_0, h_min, seed):
    info()
    model = template_model()
    mpc = template_mpc(model, w_ref, E_0, h_min=h_min)
    simulator = template_simulator(model, w_ref, E_0)
    estimator = do_mpc.estimator.StateFeedback(model)


    # set initial values and guess
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()

    n_steps = 200

    for k in range(n_steps):
        try:
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)
        except:
            break

    return simulator.data



def main():
    plan = load_pickle('./kite_sampling_01/kite_sampling_01_plan.pkl')

    sampler = do_mpc.sampling.Sampler(plan)
    sampler.set_param(print_progress = False)
    sampler.data_dir = './kite_sampling_01/sample_results/'


    sampler.set_sample_function(sample_function)

    with mp.Pool(processes=8) as pool:
        p = pool.map(sampler.sample_idx, list(range(sampler.n_samples)))

    #sampler.sample_idx(0)



if __name__ == '__main__':
    main()
