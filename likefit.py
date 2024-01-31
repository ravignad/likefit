# Library to fit data with several likelihood functions

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import cm, colors


# Cost functions of the dependent variable y
def normal_cost(mu: np.ndarray, ydata: np.ndarray, ydata_error: np.ndarray) -> np.ndarray:
    """
    Calculate the squared z-scores for a normal distribution.

    Parameters
    ----------
    mu : numpy.ndarray
        Array of the means of the normal distribution.
    ydata : numpy.ndarray
        Array of observed data points.
    ydata_error : numpy.ndarray
        Array of error or uncertainty associated with each data point.

    Returns
    -------
    numpy.ndarray
        Array of squared z-scores for each data point.
    """
    
    z_scores = (ydata - mu) / ydata_error
    return z_scores**2


def poisson_cost(mu: np.ndarray, nevents: np.ndarray) -> np.ndarray:
    """
    Calculate the Poisson costs for each data point.

    Parameters
    ----------
    mu : numpy.ndarray
        The mean parameter of the Poisson distribution for each data point.
    nevents : float or numpy.ndarray
        The number of observed events for each data point.

    Returns
    -------
    numpy.ndarray
        Cost of each data point.
    """

    cost = np.zeros_like(mu)
    # If the number of events is a float fill an array with them
    nevents_array = np.asarray(nevents, like=mu)

    """
        Piecewise-defined  function for cases ydata=0 and ydata!=0
        Note: scipy.stats.rv_discrete contains a negative log likelihood that does not work well.
        So we implement the cost function from scratch
    """
    
    # Select data points ydata=0
    zero_mask = (nevents_array == 0)
    mu1 = mu[zero_mask]
    likelihood_ratio1 = -mu1
    cost1 = -2 * likelihood_ratio1
    cost[zero_mask] += cost1

    # Select data points ydata!=0
    mu2 = mu[~zero_mask]
    nevents2 = nevents_array[~zero_mask]
    likelihood_ratio2 = nevents2 * np.log(mu2 / nevents2) - (mu2 - nevents2)
    cost2 = -2 * likelihood_ratio2
    cost[~zero_mask] += cost2

    return cost


def binomial_cost(proba: np.ndarray, nsuccess: np.ndarray, ntrials: np.ndarray) -> np.ndarray:
    """
    Calculate the binomial costs for each data point.

    Parameters
    ----------
    proba : np.ndarray
        Array of success probabilities for each data point.
    nsuccess : np.ndarray
        Array of the number of successes for each data point.
    ntrials : np.ndarray
        Array of the number of trials for each data point.

    Returns
    -------
    np.ndarray
        Binomial costs for each data point.
    """

    """
    The function is defined piecewise for the following cases:
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
    """
    Base class for likelihood-based fitters.

    Parameters
    ----------
    x : array_like
        The independent variable data.
    model : callable
        The model function to fit the data.


    Attributes
    ----------
    xdata : array_like
        The independent variable data.
    model : callable
        The model function to fit the data.
    fit_result : scipy.optimize.OptimizeResult, optional
        Result of the fitting process.
    """

    def __init__(self, x, model):
        self.xdata = x
        self.model = model
        self.fit_result = None

    @abstractmethod
    def cost_function(self, par):
        """
        Abstract method to define the cost function for the optimization process.

        Parameters
        ----------
        par : array_like
            The parameter values.

        Returns
        -------
        float
            The value of the cost function for the given parameters.
        """
        pass

    def vcost_function(self, parx_index, pary_index, parx, pary):
        """
        Vectorized version of the cost function useful for plotting.

        Parameters
        ----------
        parx_index : int
            Index of the x-axis parameter.
        pary_index : int
            Index of the y-axis parameter.
        parx : array_like
            Values of the x-axis parameter.
        pary : array_like
            Values of the y-axis parameter.

        Returns
        -------
        np.ndarray
            2D array of the cost values.
        """

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

    def fit(self, seed, **kwargs):
        """
        Fit the model to the data.

        Parameters
        ----------
        seed : array_like
            Initial guess for the parameters.
        **kwargs
            Additional keyword arguments passed to scipy.optimize.minimize.

        Returns
        -------
        int
            Status code of the optimization process.
        """
        
        self.fit_result = minimize(self.cost_function, x0=seed, **kwargs)

        if not self.fit_result.success:
            print(self.fit_result)
            raise FloatingPointError("ERROR: scipy.optimize.minimize did not converge")

        return self.fit_result.status

    # Fit results getters
    def get_estimators(self):
        """
        Get the maximum likelihood estimators of the fit parameters.

        Returns
        -------
        np.ndarray
            Array containing the maximum likelihood estimators.
        """
        
        return self.fit_result.x

    def get_errors(self):
        """
        Get the errors of the fit parameters.

        Returns
        -------
        np.ndarray
            Array containing the errors of the fit parameters.
        """
        
        covariance = self.get_covariance_matrix()
        errors = np.sqrt(np.diagonal(covariance))
        return errors

    def get_covariance_matrix(self):
        """
        Get the covariance matrix of the fitted parameters.

        Returns
        -------
        np.ndarray
            Covariance matrix of the fitted parameters.
        """    
    
        covariance = 2 * self.fit_result.hess_inv
        return covariance

    def get_confidence_ellipse(self, xindex, yindex, nsigma: int = 1, npoints: int = 100):
        """
        Get the confidence ellipse for a pair of parameters.

        Parameters
        ----------
        xindex : int
            Index of the x-axis parameter.
        yindex : int
            Index of the y-axis parameter.
        nsigma : int, optional
            Number of standard deviations for the confidence ellipse. Default is 1.
        npoints : int, optional
            Number of points to use for the ellipse. Default is 100.

        Returns
        -------
        np.ndarray
            Array containing the coordinates of the confidence ellipse.
        """

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
        """
        Get the correlation matrix of the fitted parameters.

        Returns
        -------
        np.ndarray
            Correlation matrix of the fitted parameters.
        """
        
        covariance = self.get_covariance_matrix()
        errors = self.get_errors()
        correlation = covariance / np.tensordot(errors, errors, axes=0)
        return correlation

    def get_chi_square(self):
        """
        Get the chi-square statistic of the fit.

        Returns
        -------
        float
            The chi-square statistic of the fit.
        """
        
        return self.fit_result.fun

    def get_ndof(self):
        """
        Get the number of degrees of freedom.

        Returns
        -------
        int
            Number of degrees of freedom.
        """
        
        estimators = self.get_estimators()
        ndof = len(self.xdata) - len(estimators)
        return ndof

    def get_pvalue(self):
        """
        Get the p-value of the fit.

        Returns
        -------
        float
            The p-value of the fit.
        """
        
        deviance = self.get_chi_square()
        ndof = self.get_ndof()
        pvalue = chi2.sf(deviance, ndof)
        return pvalue

    def get_yfit(self, x):
        """
        Get the model predictions for a given set of independent variables.

        Parameters
        ----------
        x : np.ndarray
            Values of the independent variable.

        Returns
        -------
        np.ndarray
            Model predictions for the given independent variable values.
        """
        
        estimators = self.get_estimators()
        return self.model(x, estimators) 

    def get_yfit_error(self, x):
        """
        Get the errors of the model predictions for a given set of independent variables.

        Parameters
        ----------
        x : np.ndarray
            Values of the independent variable.

        Returns
        -------
        np.ndarray
            Errors of the model predictions for the given independent variable values.
        """

        gradient = self.get_gradient(x)

        # Propagate parameter errors
        covariance = self.get_covariance_matrix()
        var_yfit = np.einsum("ik,ij,jk->k", gradient, covariance, gradient)
        sigma_yfit = np.sqrt(var_yfit)
        
        return sigma_yfit

    @abstractmethod
    def get_ydata(self):
        """
        Abstract method to get the maximum likelihood estimator of each data point.

        Returns
        -------
        np.ndarray
            Array containing the maximum likelihood estimator of each data point.
        """
        
        pass

    @abstractmethod
    def get_ydata_errors(self):
        """
        Abstract method to get the errors of each data point.

        Returns
        -------
        np.ndarray
            Array containing the errors of each data point.
        """
        pass

    def get_residuals(self):
        """
        Get the residuals calculated as the data minus the fit at each data point.

        Returns
        -------
        np.ndarray
            Array containing the residuals.
        """
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
        """
        Get the numerical derivative of the model with respect to the parameters.

        Parameters
        ----------
        x : np.ndarray
            Values of the independent variable at which the gradient will be evaluated.

        Returns
        -------
        np.ndarray
            2D array containing the calculated gradient.
            
       Notes
       -----
            The first dimension of the returned array corresponds to the fit parameters and the second 
            one to the values of the independent variable
        """

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
        """
        Print a summary of the fit results including estimators, errors, covariance matrix, and more.
        """
        
        print("Fit summary")
        print(f"Estimators: {self.get_estimators()}")
        print(f"Errors: {self.get_errors()}")
        print(f"Covariance matrix: {self.get_covariance_matrix()}")
        print(f"Correlation matrix: {self.get_correlation_matrix()}")
        print(f"Deviance: {self.get_chi_square()}")
        print(f"Degrees of freedom: {self.get_ndof()}")
        print(f"Pvalue: {self.get_pvalue()}")

    def plot_fit(self, xlabel="x", ylabel="y"):
        """
        Plot the data, the fit, and the error band.

        Parameters
        ----------
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        """
        
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

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

    def plot_confidence_ellipses(self, parx_index, pary_index, xlabel=None, ylabel=None):
        """
        Plot the 1σ and 2σ confidence ellipses for a pair of parameters.

        Parameters
        ----------
        parx_index : int
            Index of the x-axis parameter.
        pary_index : int
            Index of the y-axis parameter.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        """

        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        estimators = self.get_estimators()
        plt.plot(estimators[parx_index], estimators[pary_index], 'o', label="Estimator")

        ellipse1 = self.get_confidence_ellipse(parx_index, pary_index, nsigma=1)
        plt.plot(*ellipse1, label=r"1σ")
        ellipse2 = self.get_confidence_ellipse(parx_index, pary_index, nsigma=2)
        plt.plot(*ellipse2, label=r"2σ")

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_cost_function(self, parx_index, pary_index, xlabel=None, ylabel=None, nsigma=2):
        """
        Plot the surface of the fit cost function for a pair of parameters.

        Parameters
        ----------
        parx_index : int
            Index of the x-axis parameter.
        pary_index : int
            Index of the y-axis parameter.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        nsigma : int, optional
            Number of sigma confidence levels to include in the plot.
        """

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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
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

    def plot_confidence_regions(self, parx_index, pary_index, xlabel=None, ylabel=None, nsigma=2):
        """
        Plot the confidence regions for a pair of parameters.

        Parameters
        ----------
        parx_index : int
            Index of the x-axis parameter.
        pary_index : int
            Index of the y-axis parameter.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        nsigma : int, optional
            Number of sigma confidence levels to plot.

        Returns
        -------
        None
        """
        
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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Levels of the contour lines
        sigma_levels = np.arange(0, nsigma+1)
        contours = ax.contour(x, y, z, levels=sigma_levels, colors='black', linestyles='dashed')

        def fmt(nsigma_label):
            return f"{nsigma_label:.0f}σ"

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
    """
    Linear least squares fitter based on the LikelihoodFitter base class.

    Parameters
    ----------
    xdata : array_like
        The independent variable data.
    ydata : array_like
        The dependent variable data.
    ydata_error : array_like
        Errors associated with the dependent variable data.
    model : callable
        The linear model function to fit the data.

    Attributes
    ----------
    xdata : array_like
        The independent variable data.
    ydata : array_like
        The dependent variable data.
    ydata_error : array_like
        Errors associated with the dependent variable data.
    model : callable
        The linear model function to fit the data.
    model_matrix : np.ndarray
        Matrix generated by the linear model.
    estimators : np.ndarray
        Maximum likelihood estimators of the fit parameters.
    cova_par : np.ndarray
        Covariance matrix of the fit parameters.
    """

    def __init__(self, xdata, ydata, ydata_error, model):
        """
        Initialize the LinearLeastSquares instance.

        Parameters
        ----------
        xdata : array_like
            The independent variable data.
        ydata : array_like
            The dependent variable data.
        ydata_error : array_like
            Errors associated with the dependent variable data.
        model : callable
            The linear model function to fit the data.
        """
        
        LikelihoodFitter.__init__(self, xdata, model)
        self.ydata = ydata
        self.ydata_error = ydata_error
        npar = self.__count_parameters()
        self.__set_model_matrix(npar)

    def __count_parameters(self):
        """
        Count the number of the fit parameters 

        Returns
        -------
        int
            The number of fit parameters.
        
        Notes
        -----
        Dirty way to count the number of the fit parameters by forcing an IndexError exception until reaching
        the number of parameters
        """
        
        npar = 0
        par = np.array([1])
        
        while True:
            try:
                self.model(self.xdata, par)
            except IndexError:
                npar += 1
                par = np.zeros(npar)
                par[-1] = 1
            else:
                break
        
        return npar

    def __set_model_matrix(self, npar):
        """
        Set the model matrix based on the linear model and the number of fit parameters.

        Parameters
        ----------
        npar : int
            The number of fit parameters.
        """
        
        column_list = []
        for pari in range(npar):
            unit_vector = np.zeros(shape=npar)
            unit_vector[pari] = 1
            matrix_column = self.model(self.xdata, unit_vector)
            column_list.append(matrix_column)

        self.model_matrix = np.array(column_list)

    def cost_function(self, par):
        """
        Calculate the cost function for the linear least squares fit.

        Parameters
        ----------
        par : np.ndarray
            The parameters to evaluate the cost function.

        Returns
        -------
        float
            The value of the cost function.
        """
        
        yfit = self.model(self.xdata, par)
        data_point_costs = normal_cost(yfit, self.ydata, self.ydata_error)
        return data_point_costs.sum()

    def fit(self):
        """
        Perform the linear least squares fit and compute estimators and covariance matrix.
        """
        
        inv_var_y = self.ydata_error ** (-2)
        inv_cova_par = np.einsum('ij,j,lj', self.model_matrix, inv_var_y, self.model_matrix)
        self.cova_par = np.linalg.inv(inv_cova_par)
        matrix_b = np.einsum('ij,jk,k -> ik', self.cova_par, self.model_matrix, inv_var_y)
        self.estimators = np.einsum('ij,j', matrix_b, self.ydata)
        return 0

    def get_covariance_matrix(self):
        """
        Get the covariance matrix of the fit parameters.

        Returns
        -------
        np.ndarray
            Covariance matrix of the fit parameters.
        """
        
        return self.cova_par

    def get_chi_square(self):
        """
        Get the deviance of the linear least squares fit.

        Returns
        -------
        float
            The deviance of the fit.
        """
        
        residuals = self.ydata - self.get_yfit(self.xdata)
        z_array = residuals/self.ydata_error
        chi2_min = np.sum(z_array**2)
        return chi2_min

    def get_estimators(self):
        """
        Get the maximum likelihood estimators for the fitted parameters.

        Returns
        -------
        np.ndarray
            Array containing the maximum likelihood estimators.
        """
        
        return self.estimators

    def get_ydata(self):
        """
        Get the dependent variable data.

        Returns
        -------
        np.ndarray
            Array containing the dependent variable data.
        """
    
        return self.ydata

    def get_ydata_errors(self):
        """
        Get the errors of the dependent variable data.

        Returns
        -------
        np.ndarray
            Array containing the errors of the dependent variable data.
        """
        
        return self.ydata_error


class LeastSquares(LikelihoodFitter):
    """
    Non-linear least squares fitter based on the LikelihoodFitter base class.

    Parameters
    ----------
    xdata : array_like
        The independent variable data.
    ydata : array_like
        The dependent variable data.
    ydata_error : array_like
        Errors associated with the dependent variable data.
    model : callable
        The non-linear model function to fit the data.

    Attributes
    ----------
    xdata : array_like
        The independent variable data.
    ydata : array_like
        The dependent variable data.
    ydata_error : array_like
        Errors associated with the dependent variable data.
    model : callable
        The non-linear model function to fit the data.
    """
    
    def __init__(self, xdata, ydata, ydata_error, model):
        """
        Initialize the LeastSquares instance.

        Parameters
        ----------
        xdata : array_like
            The independent variable data.
        ydata : array_like
            The dependent variable data.
        ydata_error : array_like
            Errors associated with the dependent variable data.
        model : callable
            The non-linear model function to fit the data.
        """
        
        LikelihoodFitter.__init__(self, xdata, model)
        self.ydata = ydata
        self.ydata_error = ydata_error

    def cost_function(self, par):
        """
        Calculate the cost function for the non-linear least squares fit.

        Parameters
        ----------
        par : np.ndarray
            The parameters to evaluate the cost function.

        Returns
        -------
        float
            The value of the cost function.
        """
        
        yfit = self.model(self.xdata, par)
        data_point_costs = normal_cost(yfit, self.ydata, self.ydata_error)
        return data_point_costs.sum()

    def get_ydata(self):
        """
        Get the dependent variable data.

        Returns
        -------
        np.ndarray
            Array containing the dependent variable data.
        """
        return self.ydata

    def get_ydata_errors(self):
        """
        Get the errors associated with the dependent variable data.

        Returns
        -------
        np.ndarray
            Array containing the errors associated with the dependent variable data.
        """
        return self.ydata_error


class Poisson(LikelihoodFitter):
    """
    Poisson fitter based on the LikelihoodFitter base class.

    Parameters
    ----------
    xdata : array_like
        The independent variable data.
    nevents : array_like
        The number of observed events data.
    model : callable
        The model function to fit the data.

    Attributes
    ----------
    xdata : array_like
        The independent variable data.
    nevents : array_like
        The number of observed events data.
    model : callable
        The model function to fit the data.
    """

    def __init__(self, xdata, nevents, model):
        """
        Initialize the Poisson fitter instance.

        Parameters
        ----------
        xdata : array_like
            The independent variable data.
        nevents : array_like
            The number of observed events data.
        model : callable
            The model function to fit the data.
        """
        LikelihoodFitter.__init__(self, xdata, model)
        self.nevents = nevents

    def cost_function(self, par):
        """
        Calculate the Poisson cost function for the fit.

        Parameters
        ----------
        par : np.ndarray
            The parameters to evaluate the cost function.

        Returns
        -------
        float
            The value of the Poisson cost function.
        """
        yfit = self.model(self.xdata, par)
        data_point_costs = poisson_cost(yfit, self.nevents)
        return data_point_costs.sum()

    def get_ydata(self):
        """
        Get the number of observed events data.

        Returns
        -------
        np.ndarray
            Array containing the number of observed events data.
        """
        
        return self.nevents

    def get_ydata_errors(self):
        """
        Get the approximated Poisson errors associated with the number of observed events.

        Returns
        -------
        np.ndarray
            Array containing the approximated Poisson errors.
            
        Notes
        -----
        These approximated errors are invalid when the number of observed events is zero
        """
        
        return np.sqrt(self.nevents)


class Binomial(LikelihoodFitter):
    """
    Binomial fitter based on the LikelihoodFitter base class.

    Parameters
    ----------
    ntrials : array_like
        The number of trials data.
    nsuccess : array_like
        The number of successful trials data.
    model : callable
        The Binomial model function to fit the data.

    Attributes
    ----------
    ntrials : array_like
        The number of trials data.
    nsuccess : array_like
        The number of successful trials data.
    model : callable
        The model function to fit the data.
    """

    def __init__(self, x, ntrials, nsuccess, model):
        """
        Initialize the Binomial fitter instance.

        Parameters
        ----------
        ntrials : array_like
            The number of trials data.
        nsuccess : array_like
            The number of successful trials data.
        model : callable
            The model function to fit the data.
        """
        
        LikelihoodFitter.__init__(self, x, model)
        self.ntrials = ntrials
        self.nsuccess = nsuccess

    def cost_function(self, par):
        """
        Calculate the Binomial cost function for the fit.

        Parameters
        ----------
        par : np.ndarray
            The parameters to evaluate the cost function.

        Returns
        -------
        float
            The value of the Binomial cost function.
        """
        
        # Bernoulli probability for each xdata according to the model
        yfit = self.model(self.xdata, par)
        data_point_costs = binomial_cost(yfit, self.nsuccess, self.ntrials)
        return data_point_costs.sum()

    def get_ydata(self):
        """
        Get the probability of success for each trial data.

        Returns
        -------
        np.ndarray
            Array containing the probability of success for each trial data.
        """
        
        return self.nsuccess / self.ntrials

    def get_ydata_errors(self):
        """
        Get the approximated Binomial errors associated with the probability of success.

        Returns
        -------
        np.ndarray
            Array containing the approximated Binomial errors.
            
        Notes
        -----
        These approximated errors are invalid in the limits of the number of successes is equal to zero or
        to the number of trials.
        """
        
        proba_mle = self.get_ydata()
        ydata_variance = proba_mle * (1-proba_mle) / self.ntrials
        return np.sqrt(ydata_variance)
