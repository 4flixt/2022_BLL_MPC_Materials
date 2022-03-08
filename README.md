# Model predictive control with deep learning system model and Bayesian last layer trust regions

Accompanying repository for our work "Model predictive control with deep learning system model and Bayesian last layer trust regions".

## Abstract

Deep neural networks have proven to be an efficient and expressive black-box model 
to learn complex nonlinear relationships from large amounts of data. 
They are also increasingly popular in system identification for model predictive control.
One of the main caveats of neural networks in this setting is their lack of uncertainty quantification.
Especially for economic MPC, unreasonable predictions in regions without training data
might lead to poor and potentially dangerous control behavior. 
Bayesian neural networks (BNN) try to alleviate this problem but add significant complexity 
to both training and inference, rendering their application for MPC infeasible. 
Bayesian last layer (BLL) is a simplified BNN, representing a compromise between tractability and expressiveness. 
Most importantly, training and point estimates are unchanged in comparison to regular NN.
The BLL covariance computation is strongly related to Gaussian Processes (GP) but in contrast to them, 
BLL does not suffer from the same scaling issues. 
While BLL cannot be used for probabilistic guarantees in most cases, 
we propose to define a trust region based on the computed covariance.
We demonstrate in an empirical investigation that our economic MPC formulation with BLL trust region 
constraint leads to well behaved closed-loop trajectories, where 
the equivalent formulation without trust region leads to poor closed-loop performance.

## About this repository

Dear visitor, 

thank you for checking our the supplementary materials to our work. 
If you have questions, remarks, technical issues etc. feel free to **use the [discussions](https://github.com/4flixt/2021_BLL_MPC_Materials/discussions) page of this repository.**
We are looking forward to your feedback and the discussion. 

The purpose of this repository is to provide all the materials needed to recreate our results and to allow to experiment yourself with the code. 
Most importantly, we are providing here the [investigated system model (towing kite)](https://github.com/4flixt/2021_BLL_MPC_Materials/blob/main/Kite_Investigation/Kite_Introduction.ipynb).
All of our results are created using Python code (using Jupyter Notebooks) and the toolboxes:

- numpy
- matplotlib
- scipy
- CasADi

As well as our own toolbox [do-mpc](www.do-mpc.com). 
We heavily rely on the newly developed **do-mpc sampling tools**, for example for data generation and for statistical validation of our findings.
For a comprehensive guide to this feature, please watch this [video](https://www.youtube.com/watch?v=3ELyErkYPhE&t).

### Organization of this repository

- **BLL_Fundamentals**
    - [BLL_Fundamentals](https://github.com/4flixt/2021BLL_MPC_Materials/blob/main/BLL_Fundamentals/BLL_Fundamentals.ipynb): Used to generate the results shown in Figure 1 in our paper. 
- **Kite_Investigation**
    - [Kite Introduction](https://github.com/4flixt/2021_BLL_MPC_Materials/blob/main/Kite_Investigation/Kite_Introduction.ipynb): Statement of the investigated system and all required parameters. Showcasing an examplary MPC closed trajectory and how to obtain it. 
    - **system**
        - Python files for model, simulator, (MPC) controller used to generate closed-loop trajectories
        - ``main.py`` file (creating similar results as in the Kite Introduction Jupyter Notebook)
    - **sampling** Sample system to create training data for GP/NN system model
        - ``create_sampling_plan.py``: Define 100 cases (varying initial states, etc.)
        - ``sample_sytem.py``: Create closed-loop trajectories for the defined cases 
    - **surrogate** Train GP and NN system model
        - [train_test_val_data_prep](https://github.com/4flixt/2021_BLL_MPC_Materials/blob/main/Kite_Investigation/surrogate/train_test_val_data_prep.ipynb): Load and analyze sampled sequences. Create training, testing and validation data for GP and NN.
        - [nn_kite_modell](https://github.com/4flixt/2021_BLL_MPC_Materials/blob/main/Kite_Investigation/surrogate/nn_kite_modell.ipynb): Jupyter notebook from which the NN system model was trained. 
        - [gp_kite_modell](https://github.com/4flixt/2021_BLL_MPC_Materials/blob/main/Kite_Investigation/surrogate/gp_kite_modell.ipynb): Jupyter notebook from which the GP system model was trained. 
        - [nn_vs_gp_sys_id](https://github.com/4flixt/2021_BLL_MPC_Materials/blob/main/Kite_Investigation/surrogate/nn_vs_gp_sys_id.ipynb): Jupyter notebook to compare GP and NN system model in open-loop predictions (used for Figure 2 in our paper)
    - **mpc**: MPC with surrogate NN/GP model(s)
        - Python files for (GP / NN)model, simulator, (MPC) controller
        - **validation_sampling**: Validate MPC controller based on surrogate model
            - ``create_sampling_plan.py``: Define 20 cases
            - ``sample_sytem.py``: Create closed-loop trajectories for the defined cases 
            - [validation_evalution](https://github.com/4flixt/2021_BLL_MPC_Materials/blob/main/Kite_Investigation/surrogate/mpc/validation_sampling/validation_evalution.ipynb): Jupyter notebook to compare the obtained results (used for Figure 3 in our paper)


Please notice that we **do not include data to reproduce the results** due to space restrictions of this repository.
We do, however, provide all the tools to re-sample the exact same data that we used (random seeds are fixed). 


