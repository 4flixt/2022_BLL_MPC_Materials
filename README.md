# Model predictive control with deep learning system model and Bayesian last layer trust regions

Accompanying repository for our work "Model predictive control with deep learning system model and Bayesian last layer trust regions" submitted to L4DC 2021.

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
If you have questions, remarks, technical issues etc. feel free to **use the discussions page of this repository.**
We are looking forward to your feedback and the discussion. 

The purpose of this repository is to provide all the materials needed to recreate our results and to allow to experiment yourself with the code. 
Most importantly, we are providing here the investigated system model.
All of our results are created using Python code (using Jupyter Notebooks) and the toolboxes:

- numpy
- matplotlib
- scipy
- CasADi

As well as our own toolbox [do-mpc](www.do-mpc.com).

**To investigate the code**, we recommend to either locally install Python (and the toolboxes) and then clone the repo **or** you you can use the Binder links listed below. Binder clones the repo on their servers and installs all packages automatically. This way you can get started immediately! 
