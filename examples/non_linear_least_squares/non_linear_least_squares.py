# Example of fitting data with the least squares method

import numpy as np
import likefit

xdata = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
ydata = np.array([0.92, 0.884, 0.626, 0.504, 0.481, 0.417, 0.288, 0.302, 0.177, 0.13, 0.158])
ysigma = np.array([0.1, 0.082, 0.067, 0.055, 0.045, 0.037, 0.03, 0.025, 0.02, 0.017, 0.014])


# fit_model vectorized in x
def fit_model(x, par):
    return par[0] * np.exp(par[1] * x)


fitter = likefit.NonLinearLeastSquares(xdata, ydata, ysigma, fit_model)
seed = np.array([0, 0])
fitter.fit(seed)
fitter.print_results()

# Plot data and fit
fitter.plot_fit()

# Plot the 1σ and 2σ confidence ellipses
# The first parameter is in x-axis and the second parameter in the y-axis
fitter.plot_confidence_ellipses(parx_index=0, pary_index=1)

# plt.savefig("non_linear_least_squares.png")
