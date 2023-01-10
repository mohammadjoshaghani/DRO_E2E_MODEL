import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

#---------------------------------------------------------------------------------------------------
# Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta
#---------------------------------------------------------------------------------------------------
class DecisionLayer():
    def __init__(self, distance='HL', n_obs=104):
        """ this class creates decision layer.
        it can use hellinger distance or kl-divergance
        for defining ambiguity set.

        Args:
            name (str, optional): the ambiguity set diverganec metric.
                'HL' for 'Hellinger_distance' or 'KL' for 'kl_divergance'. Defaults to 'HL'.
        """
        names={"KL":'kl_divergance', "HL":'Hellinger_distance'}
        self.Declayer = eval('self.' + names[distance])(n_y=20, n_obs=n_obs)
        
    def Hellinger_distance(self, n_y=20, n_obs=104):
        """DRO layer using the Hellinger distance to define the probability ambiguity set.
        from Ben-Tal et al. (2013).
        Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta

        Inputs
        n_y: number of assets
        n_obs: Number of scenarios in the dataset
        prisk: Portfolio risk function
        
        Variables
        z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
        c_aux: Auxiliary Variable. Scalar. Allows us to p-linearize the derivation of the variance
        lambda_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
        xi_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
        beta_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.
        s_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable SOC constraint.
        mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

        Parameters
        ep: (n_obs x n_y) matrix of residuals 
        y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
        returns)
        delta: Scalar. Maximum distance between p and q.
        gamma: Scalar. Trade-off between conditional expected return and model error.

        Constraints
        Total budget is equal to 100%, sum(z) == 1
        Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)
        All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al.
        (2013).

        Objective
        Minimize xi_aux + (delta-1) * lambda_aux + (1/n_obs) * sum(beta_aux) - gamma * y_hat @ z
        """
        # Variables
        z = cp.Variable((n_y,1), nonneg=True)
        c_aux = cp.Variable()
        lambda_aux = cp.Variable(nonneg=True)
        xi_aux = cp.Variable()
        beta_aux = cp.Variable(n_obs, nonneg=True)
        tau_aux = cp.Variable(n_obs, nonneg=True)
        mu_aux = cp.Variable()

        # Parameters
        ep = cp.Parameter((n_obs, n_y))
        y_hat = cp.Parameter(n_y)
        gamma = cp.Parameter(nonneg=True)
        delta = cp.Parameter(nonneg=True)

        # Constraints
        constraints = [cp.sum(z) == 1,
                        mu_aux == y_hat @ z]
        for i in range(n_obs):
            constraints += [xi_aux + lambda_aux >= self.p_var(z, c_aux, ep[i]) + tau_aux[i]]
            constraints += [beta_aux[i] >= cp.quad_over_lin(lambda_aux, tau_aux[i])]
        
        # Objective function
        objective = cp.Minimize(xi_aux + (delta-1) * lambda_aux + (1/n_obs) * cp.sum(beta_aux) 
                                - gamma * mu_aux)

        # Construct optimization problem and differentiable layer
        problem = cp.Problem(objective, constraints)
        
        return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])  
    
    def kl_divergance(self,n_y=20, n_obs=104):

        # Variables
        z = cp.Variable((n_y,1), nonneg=True)
        c_aux = cp.Variable()
        lambda_aux = cp.Variable(nonneg=True)
        xi_aux = cp.Variable()
        zz_aux = cp.Variable(n_obs)
        mu_aux = cp.Variable()

        # Parameters
        ep = cp.Parameter((n_obs, n_y))
        y_hat = cp.Parameter(n_y)
        gamma = cp.Parameter(nonneg=True)
        delta = cp.Parameter(nonneg=True)        

        # Constraints
        constraints = [cp.sum(z) == 1,
                        mu_aux == y_hat @ z,]
        for i in range(n_obs):
            constraints += [self.f_lambd(lambda_aux,zz_aux[i]) <= xi_aux - self.p_var(z, c_aux, ep[i])]
        
        # Objective function
        objective = cp.Minimize(xi_aux + delta * lambda_aux + (1/n_obs) * cp.sum(zz_aux) 
                                - gamma * mu_aux)

        # Construct optimization problem and differentiable layer
        problem = cp.Problem(objective, constraints)
        return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])
    
    ####################################################################################################
    # Define risk functions
    ####################################################################################################    
    
    def p_var(self,z, c, x):
        """Variance
        Inputs
        z: (n x 1) vector of portfolio weights (decision variable)
        c: Scalar. Centering parameter that serves as a proxy to the expected value (auxiliary variable)
        x: (n x 1) vector of realized returns (data)

        Output: Single squared deviation. 
        Note: This function is only one component of the portfolio variance, and must be aggregated 
        over all scenarios 'x' to recover the complete variance
        """
        return cp.square(x @ z - c)
    
    ####################################################################################################
    # Define convex conjugate of adjoint of \phi divergance
    ####################################################################################################
    
    def f_lambd(self, lambda_aux,zz_aux_i):
        """this function calculates convex conjugate of adjoint of \pi divergance
        for kl-divergance distance.
        """
        return  cp.rel_entr(lambda_aux,lambda_aux+zz_aux_i)    


if __name__ == '__main__':
    DC = DecisionLayer('kl_divergance').Declayer        