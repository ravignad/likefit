# Example of fitting data with a linear least squares method

import numpy as np
import likefit

xdata = np.array([1.02, 1.06, 1.1, 1.14, 1.18, 1.22, 1.26, 1.3, 1.34])
ydata = np.array([2.243, 2.217, 2.201, 2.175, 2.132, 2.116, 2.083, 2.016, 2.004])
ysigma = np.array([0.008, 0.008, 0.01, 0.009, 0.011, 0.016, 0.018, 0.021, 0.017])
npar = 2


# Model linear in the parameters
def fit_model(x, par):
    return par[0] + par[1] * (x-1.2)


fitter = likefit.LinearLeastSquares(xdata, ydata, ysigma, npar, fit_model)
fitter.fit()
fitter.print_results()

# Plot data and fit
fitter.plot_fit()
