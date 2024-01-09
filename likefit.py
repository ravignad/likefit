# Library to fit data with several likelihood functions

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import cm, colors


# Cost functions of the dependent variable y
def normal_cost(ydata, yfit, ydata_error):
    z_scores = (ydata - yfit) / ydata_error
    return np.sum(z_scores**2)


def poisson_cost(mu, nevents):
    """
        Piecewise-defined  function for cases ydata=0 and ydata!=0
        Note: scipy.stats.rv_discrete contains a negative log likelihood that does not work well.
        So we implement the cost function from scratch
    """

    cost = np.zeros_like(mu)

    # Select data points ydata=0
    zero_mask = (nevents == 0)
    mu1 = mu[zero_mask]
    likelihood_ratio1 = -mu1
    cost1 = -2 * likelihood_ratio1
    cost[zero_mask] += cost1

    # Select data points ydata!=0
    mu2 = mu[~zero_mask]
    nevents2 = nevents[~zero_mask]
    likelihood_ratio2 = nevents2 * np.log(mu2 / nevents2) - (mu2 - nevents2)
    cost2 = -2 * likelihood_ratio2
    cost[~zero_mask] += cost2

    return cost


def binomial_cost(proba, nsuccess, ntrials):
    """
    Piecewise-defined function for cases:
        1) nsuccess=0
        2) 0 < nsuccess < ntrials
        3) nsuccess=ntrials
    """

    cost = np.zeros_like(proba)

    # Maximum likelihood estimator of the Bernoulli probability
    proba_mle = nsuccess / ntrials

    # Case nsuccess = 0
    zero_mask = nsuccess == 0
    proba1 = proba[zero_mask]
    ntrials1 = ntrials[zero_mask]
    likelihood_ratio1 = ntrials1 * np.log(1 - proba1)
    cost1 = -2 * likelihood_ratio1.sum()
    cost[zero_mask] += cost1

    # Case 0 < nsuccess < ntrials
    intermediate_mask = np.logical_and(0 < nsuccess, nsuccess < ntrials)
    proba2 = proba[intermediate_mask]
    ntrials2 = ntrials[intermediate_mask]
    proba_mle2 = proba_mle[intermediate_mask]
    likelihood_ratio2 = ntrials2 * (proba_mle2 * np.log(proba2 / proba_mle2)
                                    + (1 - proba_mle2) * np.log((1 - proba2) / (1 - proba_mle2)))
    cost2 = -2 * likelihood_ratio2.sum()
    cost[intermediate_mask] += cost2

    # Case nsuccess = ntrials
    ntrials_mask = nsuccess == ntrials
    proba3 = proba[ntrials_mask]
    ntrials3 = ntrials[ntrials_mask]
    likelihood_ratio3 = ntrials3 * np.log(proba3)
    cost3 = -2 * likelihood_ratio3.sum()
    cost[ntrials_mask] += cost3

    return cost


# Base class of all fitters
class LikelihoodFitter(ABC):

    def __init__(self, x, model, par_names=None):
        self.xdata = x
        self.model = model
        self.fit_result = None
        self.par_names = par_names

    @abstractmethod
    def cost_function(self, par):
        pass

    # Vectorized version of the cost function useful for plotting
    def vcost_function(self, parx_index, pary_index, parx, pary):

        vcost = []
        for y in pary:
            for x in parx:
                par = self.get_estimators().copy()
                par[parx_index] = x
                par[pary_index] = y
                cost1 = self.cost_function(par)
                vcost.append(cost1)

        vcost = np.reshape(vcost, newshape=(len(pary), len(parx)))
        return vcost

    # kwargs passed to scipy.optimize.minimize to control the minimization
    def fit(self, seed, **kwargs):
        self.fit_result = minimize(self.cost_function, x0=seed, **kwargs)

        if not self.fit_result.success:
            print(self.fit_result)
            raise FloatingPointError("ERROR: scipy.optimize.minimize did not converge")

        return self.fit_result.status

    # Fit results getters
    def get_estimators(self):
        return self.fit_result.x

    def get_errors(self):
        covariance = self.get_covariance_matrix()
        errors = np.sqrt(np.diagonal(covariance))
        return errors

    def get_covariance_matrix(self):
        covariance = 2 * self.fit_result.hess_inv
        return covariance

    def get_confidence_ellipse(self, xindex, yindex, nsigma: int = 1, npoints: int = 100):

        # Get the estimators and covariance matrix for a pair of estimators
        estimators = self.get_estimators()
        estimator_pair = estimators[[xindex, yindex]]
        cova = self.get_covariance_matrix()
        cova_pair = cova[np.ix_([xindex, yindex], [xindex, yindex])]

        # Calculate the nσ ellipse for the estimator pair
        cholesky_l = np.linalg.cholesky(cova_pair)
        t = np.linspace(0, 2 * np.pi, npoints)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        ellipse = nsigma * circle @ cholesky_l.T + estimator_pair

        return ellipse.T

    def get_correlation_matrix(self):
        covariance = self.get_covariance_matrix()
        errors = self.get_errors()
        correlation = covariance / np.tensordot(errors, errors, axes=0)
        return correlation

    def get_deviance(self):
        return self.fit_result.fun

    def get_ndof(self):
        estimators = self.get_estimators()
        ndof = len(self.xdata) - len(estimators)
        return ndof

    def get_pvalue(self):
        deviance = self.get_deviance()
        ndof = self.get_ndof()
        pvalue = chi2.sf(deviance, ndof)
        return pvalue

    # def get_minimize_result(self):
    #    return self.fit_result

    def get_yfit(self, x):
        estimators = self.get_estimators()
        return self.model(x, estimators) 

    def get_yfit_error(self, x):
        gradient = self.get_gradient(x)

        # Propagate parameter errors
        covariance = self.get_covariance_matrix()
        var_yfit = np.einsum("ik,ij,jk->k", gradient, covariance, gradient)
        sigma_yfit = np.sqrt(var_yfit)
        
        return sigma_yfit

    # Return an array with the maximum likelihood estimator of each data point
    @abstractmethod
    def get_ydata(self):
        pass

    # Return an array with the error of each data point
    @abstractmethod
    def get_ydata_errors(self):
        pass

    # Return an array with the difference between the data and the fit
    def get_residuals(self):
        return self.get_ydata() - self.get_yfit(self.xdata)

    # Numerical derivative of the model wrt the parameters evaluated at the estimators
    """
    Arguments:
        xdata (np.array): values of the independent variable at which the gradient will be evaluated
    Return: 
        gradient (np.array):  2 dimensional array containing the calculated gradient. 
            The first dimension corresponds to the parameters and the second one to the values of 
            the independent variable
    """
    def get_gradient(self, x):

        estimators = self.get_estimators()
        errors = self.get_errors()

        # Setting finite difference steps to some fraction of the errors
        delta_fraction = 0.01
        steps = errors * delta_fraction
        ndimensions = len(steps)

        gradient = []
        for i in range(ndimensions):

            step1 = steps[i]

            # Change an element of the parameter vector by step
            delta_par1 = np.zeros_like(steps)
            delta_par1[i] = step1
            par_down = estimators - delta_par1
            par_up = estimators + delta_par1

            # Calculate an element of the gradient vector
            model_up = self.model(x, par_up)
            model_down = self.model(x, par_down)
            gradient1 = (model_up - model_down) / (2 * step1)
            gradient.append(gradient1)

        return gradient

    def print_results(self):
        print(f"Estimators: {self.get_estimators()}")
        print(f"Errors: {self.get_errors()}")
        print(f"Covariance matrix: {self.get_covariance_matrix()}")
        print(f"Correlation matrix: {self.get_correlation_matrix()}")
        print(f"Deviance: {self.get_deviance()}")
        print(f"Degrees of freedom: {self.get_ndof()}")
        print(f"Pvalue: {self.get_pvalue()}")

    def plot_fit(self):
        # Plot
        fig, ax = plt.subplots()
        ax.set_xlabel("xdata")
        ax.set_ylabel("ydata")

        # Plot data
        ax.errorbar(self.xdata, self.get_ydata(), self.get_ydata_errors(), ls='none', marker='o', label="Data")

        # Plot fitter
        xmin = self.xdata.min()
        xmax = self.xdata.max()
        xfit = np.linspace(start=xmin, stop=xmax, num=100)
        yfit = self.get_yfit(xfit)
        ax.plot(xfit, yfit, ls='--', label="Fit")

        # Plot error band
        yfit_error = self.get_yfit_error(xfit)
        ax.fill_between(xfit, yfit - yfit_error, yfit + yfit_error, color='tab:orange', alpha=0.2)

        plt.legend()
        plt.tight_layout()
        plt.show()


    # Plot the 1σ and 2σ confidence ellipses
    # The ellipses are calculated from the covariance matrix of the estimators
    # Two parameters must be selected
    # The first parameter is in xdata-axis and the second parameter in the ydata-axis
    def plot_confidence_ellipses(self, parx_index, pary_index, parx_name=None, pary_name=None):

        # Plot
        fig, ax = plt.subplots()
        ax.set_xlabel(parx_name)
        ax.set_ylabel(pary_name)

        estimators = self.get_estimators()
        plt.plot(estimators[parx_index], estimators[pary_index], 'o', label="Estimator")

        ellipse1 = self.get_confidence_ellipse(parx_index, pary_index, nsigma=1)
        plt.plot(*ellipse1, label=r"1σ")
        ellipse2 = self.get_confidence_ellipse(parx_index, pary_index, nsigma=2)
        plt.plot(*ellipse2, label=r"2σ")

        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plot a surface of the fit cost function
    # Two parameters must be selected
    # The first parameter is in xdata-axis and the second parameter in the ydata-axis
    # nsgima: number of nσ confidence levels to include in the plot
    def plot_cost_function(self, parx_index, pary_index, parx_name=None, pary_name=None, nsigma=2):

        # Calculate coordinates of the points to plot
        estimators = self.get_estimators()
        errors = self.get_errors()

        parx_min = estimators[parx_index] - nsigma * errors[parx_index]
        parx_max = estimators[parx_index] + nsigma * errors[parx_index]
        parx = np.linspace(parx_min, parx_max, num=50)

        pary_min = estimators[pary_index] - nsigma * errors[pary_index]
        pary_max = estimators[pary_index] + nsigma * errors[pary_index]
        pary = np.linspace(pary_min, pary_max, num=50)

        x, y = np.meshgrid(parx, pary)
        cost = self.vcost_function(parx_index, pary_index, parx, pary)
        z = cost - cost.min()

        # Plot
        fig = plt.figure(figsize=(5, 4))
        ax = fig.subplots(subplot_kw={"projection": "3d"})
        ax.set_xlabel(parx_name)
        ax.set_ylabel(pary_name)
        ax.set_zlabel(r"$-2\log(L/L_{max})$")

        # Levels of the contour lines
        sigma_levels = np.arange(0, 7)
        bounds = sigma_levels ** 2
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=128)
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, norm=norm)

        # Plot colorbar
        clb = plt.colorbar(surf, shrink=0.5, location='left')
        clb.ax.set_title(r"$\sigma^2$")

        plt.show()

    # Plot a the contour levels of the fit cost function
    # Two parameters must be selected
    # The first parameter is in xdata-axis and the second parameter in the ydata-axis
    # nsgima: number of nσ confidence levels to plot
    def plot_confidence_regions(self, parx_index, pary_index, parx_name=None, pary_name=None, nsigma=2):
        # Calculate coordinates of the points to plot
        estimators = self.get_estimators()
        errors = self.get_errors()

        parx_min = estimators[parx_index] - (nsigma+1) * errors[parx_index]
        parx_max = estimators[parx_index] + (nsigma+1) * errors[parx_index]
        parx = np.linspace(parx_min, parx_max, num=50)

        pary_min = estimators[pary_index] - (nsigma+1) * errors[pary_index]
        pary_max = estimators[pary_index] + (nsigma+1) * errors[pary_index]
        pary = np.linspace(pary_min, pary_max, num=50)

        x, y = np.meshgrid(parx, pary)
        cost = self.vcost_function(parx_index, pary_index, parx, pary)
        z = np.sqrt(cost - cost.min())

        # Plot
        fig, ax = plt.subplots()
        ax.set_xlabel(parx_name)
        ax.set_ylabel(pary_name)

        # Levels of the contour lines
        sigma_levels = np.arange(0, nsigma+1)
        contours = ax.contour(x, y, z, levels=sigma_levels, colors='black', linestyles='dashed')

        def fmt(x):
            return f"{x:.0f}σ"

        ax.clabel(contours, contours.levels, fmt=fmt, inline=True)

        plt.pcolormesh(x, y, z, shading='auto', cmap=plt.cm.viridis_r)
        clb = plt.colorbar()
        clb.ax.set_title(r"$n\sigma$")

        # contours = ax.contourf(xdata, ydata, np.sqrt(z), levels=sigma_levels, cmap='Blues_r')
        # clb = fig.colorbar(contours)
        # clb.ax.set_title(r"$n\sigma$")

        plt.tight_layout()
        plt.show()


class LinearLeastSquares(LikelihoodFitter):

    def __init__(self, xdata, ydata, ydata_error, npar, model):
        LikelihoodFitter.__init__(self, xdata, model)
        self.ydata = ydata
        self.ydata_error = ydata_error
        self.npar = npar
        self.__set_model_matrix()

    def __set_model_matrix(self):
        column_list = []

        for pari in range(self.npar):
            unit_vector = np.zeros(shape=self.npar)
            unit_vector[pari] = 1
            matrix_column = self.model(self.xdata, unit_vector)
            column_list.append(matrix_column)

        self.model_matrix = np.array(column_list)

    def cost_function(self, par):
        yfit = self.model(self.xdata, par)
        return normal_cost(self.ydata, yfit, self.ydata_error)

    def fit(self):
        inv_var_y = self.ydata_error ** (-2)
        inv_cova_par = np.einsum('ij,j,lj', self.model_matrix, inv_var_y, self.model_matrix)
        self.cova_par = np.linalg.inv(inv_cova_par)
        matrix_b = np.einsum('ij,jk,k -> ik', self.cova_par, self.model_matrix, inv_var_y)
        self.estimators = np.einsum('ij,j', matrix_b, self.ydata)

    def get_covariance_matrix(self):
        return self.cova_par

    def get_deviance(self):
        residuals = self.ydata - self.get_yfit(self.xdata)
        z_array = residuals/self.ydata_error
        chi2_min = np.sum(z_array**2)
        return chi2_min

    def get_estimators(self):
        return self.estimators

    def get_ydata(self):
        return self.ydata

    def get_ydata_errors(self):
        return self.ydata_error


class NonLinearLeastSquares(LikelihoodFitter):

    def __init__(self, xdata, ydata, ydata_error, model):
        LikelihoodFitter.__init__(self, xdata, model)
        self.ydata = ydata
        self.ydata_error = ydata_error

    def cost_function(self, par):
        yfit = self.model(self.xdata, par)
        return normal_cost(self.ydata, yfit, self.ydata_error)

    def get_ydata(self):
        return self.ydata

    def get_ydata_errors(self):
        return self.ydata_error


class Poisson(LikelihoodFitter):

    def __init__(self, xdata, nevents, model):
        LikelihoodFitter.__init__(self, xdata, model)
        self.nevents = nevents

    # Poisson cost function
    def cost_function(self, par):
        yfit = self.model(self.xdata, par)
        data_point_costs = poisson_cost(yfit, self.nevents)
        return data_point_costs.sum()

    def get_ydata(self):
        return self.nevents

    # Approximated Poisson errors
    def get_ydata_errors(self):
        return np.sqrt(self.nevents)


class Binomial(LikelihoodFitter):

    def __init__(self, x, ntrials, nsuccess, model):
        LikelihoodFitter.__init__(self, x, model)
        self.ntrials = ntrials
        self.nsuccess = nsuccess

    # Binomial cost function
    def cost_function(self, par):
        # Bernoulli probability for each xdata according to the model
        yfit = self.model(self.xdata, par)
        data_point_costs = binomial_cost(yfit, self.nsuccess, self.ntrials)
        return data_point_costs.sum()

    def get_ydata(self):
        return self.nsuccess / self.ntrials

    # Approximated binomial errors, not valid when nsuccess~0 or nsuccess~ntrials
    def get_ydata_errors(self):
        proba_mle = self.get_ydata()
        ydata_variance = proba_mle * (1-proba_mle) / self.ntrials
        return np.sqrt(ydata_variance)
