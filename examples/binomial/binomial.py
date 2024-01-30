# Example of fitting data with the least squares method

import numpy as np
import likefit

xdata = np.arange(start=0.05, stop=1.05, step=0.05)
ntrials = np.full(xdata.shape, fill_value=30)
nsuccess = np.array([0, 0, 0, 3, 3, 2, 8, 5, 4, 11, 18, 15, 19, 20, 26, 24, 26, 29, 30, 30])


# fit_model is sigmoid function vectorized in xdata
def fit_model(x, par):
    return 1 / (1+np.exp(-(x-par[0])/par[1]))


fitter = likefit.Binomial(xdata, ntrials, nsuccess, fit_model)
seed = np.array([0.5, 1])
fitter.fit(seed, tol=1e-4)
fitter.print_results()

# Plot data and fit
fitter.plot_fit()
