import sys
sys.path.append('../../../../../do-mpc_fork/')
import do_mpc
import time
import os

import numpy as np
import pdb


def main():
    np.random.seed(99)

    sp = do_mpc.sampling.SamplingPlanner()
    sp.data_dir = './kite_validation_01/'
    sp.set_param(overwrite=True)

    def gen_x0():
        lb_theta = 0.3
        lb_phi = -0.7
        lb_psi = -1.0*np.pi

        ub_theta = 0.7
        ub_phi = 0.7
        ub_psi = 1.0*np.pi

        # with mean and radius:
        m_theta, r_theta = (ub_theta+lb_theta)/2, (ub_theta-lb_theta)/2
        m_phi, r_phi = (ub_phi+lb_phi)/2, (ub_phi-lb_phi)/2
        m_psi, r_psi = (ub_psi+lb_psi)/2, (ub_psi-lb_psi)/2
        # How close can the intial state be to the bounds?
        # tightness=1 -> Initial state could be on the bounds.
        # tightness=0 -> Initial state will be at the center of the feasible range.
        tightness = 0.8
        theta_0 = m_theta-tightness*r_theta+2*tightness*r_theta*np.random.rand()
        phi_0 = m_phi-tightness*r_phi+2*tightness*r_phi*np.random.rand()
        psi_0 = m_psi-tightness*r_psi+2*tightness*r_psi*np.random.rand()

        x0 = np.array([theta_0, phi_0, psi_0]).reshape(-1,1)

        return x0

    # Add variables
    #sp.set_sampling_var('w_ref', lambda: np.random.uniform(8,12))
    #sp.set_sampling_var('E_0', lambda: np.random.uniform(4,8))
    #sp.set_sampling_var('h_min', lambda: np.random.uniform(80,120))
    sp.set_sampling_var('seed', lambda: np.random.randint(1,1000))
    sp.set_sampling_var('x0', gen_x0)


    plan = sp.gen_sampling_plan(n_samples = 20)

    sp.export('kite_validation_01')



if __name__ == '__main__':
    main()


# pool = mp.Pool(processes=4)
# res = [pool.apply_async(sampler.sample_data, args=()) for k in range(4)]
# out = [p.get() for p in res]

# dh = do_mpc.sampling.DataHandler(plan)
#
# dh.set_post_processing('res_1', lambda x: x)
# dh.set_post_processing('res_2', lambda x: x**2)
#
#
# res = dh[:]
