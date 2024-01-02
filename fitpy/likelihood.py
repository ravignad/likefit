import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2


class LeastSquares:  


    def __init__(self, x, y, ysigma, model):  
        self.x = x
        self.y = y
        self.ysigma = ysigma
        self.model = model
            

    def cost_function(self, theta): 
        mu = self.model(self.x, theta)        
        residuals = (self.y - mu) / self.ysigma
        cost = np.sum(residuals**2)
        return cost


    def __call__(self, seed):
        self.fit_result = minimize(self.cost_function, x0=seed)

        if self.fit_result.success == False:
            print(self.fit_result)
            raise FloatingPointError("ERROR: scipy.optimize.minimize did not converge")

        return self.fit_result.status
        

    # Fit results getters
    
    def get_estimators(self):
        return self.fit_result.x
        
        
    def get_errors(self):
        covariance = self.get_covariance_matrix()
        errors = np.sqrt( np.diagonal(covariance) )
        return errors
        
        
    def get_covariance_matrix(self):
        covariance = 2 * self.fit_result.hess_inv
        return covariance
        
    
    def get_correlation_matrix(self):
        covariance = self.get_covariance_matrix()
        errors = self.get_errors()
        correlation = covariance / np.tensordot(errors, errors, axes=0)
        return correlation
    
    
    def get_deviance(self):
        return self.fit_result.fun
        
        
    def get_ndof(self):
        estimators = self.get_estimators()
        ndof = len(self.x) - len(estimators)
        return ndof
        
        
    def get_pvalue(self):
        deviance = self.get_deviance()
        ndof = self.get_ndof()
        pvalue = chi2.sf(deviance, ndof)
        return pvalue
        
        
    def get_minimize_result(self):
        return self.fit_result
        
        
    def get_yfit(self, x):
        estimators = self.get_estimators()
        return self.model(x, estimators) 
        
        
    def get_yfit_error(self, x):

        # Numerical derivative of the model wrt the parameters evaluated at the estimators
        gradient_list = []

        estimators = self.get_estimators()
        errors = self.get_errors()
        delta_theta = errors * 0.01
        ndimensions = len(delta_theta)

        for i in range(ndimensions):
            delta_theta1 = np.zeros_like(delta_theta)
            step = delta_theta[i]
            delta_theta1[i] = step
            theta_down = estimators - delta_theta1
            theta_up = estimators + delta_theta1
            model_up = self.model(x, theta_up)
            model_down = self.model(x, theta_down)
            gradient1 = (model_up - model_down) / (2*step)
            gradient_list.append(gradient1)

        gradient_array = np.array(gradient_list)
        
        # Propagate parameter errors
        covariance = self.get_covariance_matrix()
        # var_yfit = np.einsum("ki,ij,kj->k", gradient_array, covariance, gradient_array)
        var_yfit = np.einsum("ik,ij,jk->k", gradient_array, covariance, gradient_array)
        sigma_yfit = np.sqrt(var_yfit)
        
        return sigma_yfit
    
    
        
    
# class Poisson(object):  

#    def __call__(self, theta) :  
#        mu = self.fit_model(theta)
#        cost_array = 2 * (mu - self.k) - 2 * self.k * np.log(mu / self.k)
#        return cost_array.sum()




    

