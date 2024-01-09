# Example of fitting data with the least squares method

import numpy as np
from scipy.stats import norm

import likefit

xdata = np.linspace(start=-2.9, stop=2.9, num=30)
nevents = np.array([0, 2, 5, 8, 7, 18, 15, 27, 34, 51, 55, 63, 67, 75, 90, 78, 73, 70, 62, 51, 33, 26, 30, 17, 15, 14,
                    5, 4, 1, 0])


# fit_model vectorized in xdata
def fit_model(x, par):
    return par[0] * norm.pdf(x, loc=par[1], scale=par[2])


fitter = likefit.Poisson(xdata, nevents, fit_model)
seed = np.array([1, 0, 1])
fitter.fit(seed)
fitter.print_results()

# Plot data and fit
fitter.plot_fit()
