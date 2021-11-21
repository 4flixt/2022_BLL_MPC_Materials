import numpy as np
from casadi import *



def generic_kernel(X1,X2, k_tilde):
    """Generic kernel. Wraps a Kernel function and evaluates it for all its elements."""
    if isinstance(X1,(casadi.MX, casadi.SX)) or isinstance(X2,(casadi.MX, casadi.SX)):
        K = vertcat(*[horzcat(*[k_tilde(xi, xj) for xj in vertsplit(X2)]) for xi in vertsplit(X1)])
    else:
        K = np.array([[k_tilde(xi,xj) for xj in X2] for xi in X1])
    return K

def RBF(l=0.5):
    """Radial-basis-function Kernel. Calling this function returns a function (parameterized with length-scale).
    """

    def rbf_eval(x1,x2,l=l):
        if isinstance(x1,(casadi.MX, casadi.SX)) or isinstance(x2,(casadi.MX, casadi.SX)):
            return exp(-0.5*sum2(sum1((x1-x2)**2))/l**2)
        else:
            return np.exp(-0.5*np.sum((x1-x2)**2)/l**2)

    return rbf_eval

class GP:
    """Implementation of a Gaussian Process.
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self,X,Y,Sigma):
        self.X = X
        self.Y = Y
        self.k_XX = generic_kernel(X,X,self.kernel)
        self.C = np.linalg.pinv(self.k_XX+Sigma)

    def predict(self,x, return_std=True):
        k_xX = generic_kernel(x,self.X,self.kernel)
        k_xx = generic_kernel(x,x,self.kernel)

        y = k_xX@self.C@self.Y

        if return_std:
            v = k_xx-k_xX@self.C@k_xX.T
            return y,v
        else:
            return y

class GPPredictModel:
    def __init__(self, gpr, in_scaler, out_scaler, x0):
        self.gpr = gpr
        self.in_scaler = in_scaler
        self.out_scaler = out_scaler

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

        gpr_in = np.concatenate((u,p,x), axis=1)
        gpr_in_scaled = self.in_scaler.transform(gpr_in)

        gpr_out_scaled, v = self.gpr.predict(gpr_in_scaled, return_std=True)

        gpr_out = (self.out_scaler.inverse_transform(gpr_out_scaled)).reshape(1,-1)
        aux = gpr_out[:,:2]
        x_next = gpr_out[:,2:]+x


        self._v.append(v)
        self._x.append(x_next)
        self._aux.append(aux)
        self._u.append(u)
