# Fit the cosmic ray spectrum with a non-linear least squares fit

import numpy as np
import likefit

# Data binned in $x = log_{10}(E/eV)$, y = number of cosmic rays in the energy bin
xdata = np.linspace(18.45, 20.45, 21)
ydata = np.array([13023, 7711, 4478, 3159, 2162, 1483, 1052, 699, 451, 323, 200, 110, 43, 28, 23, 5, 2, 0, 1, 0, 0])
ysigma = np.sqrt(ydata)

# Fit model
def fit_model(x, theta):
    return np.power(10, theta[0]-theta[1]*(x-19))


# Select the bins to fit
xfit = xdata[3:11]
yfit = ydata[3:11]
yfit_sigma = ysigma[3:11]

fitter = likefit.NonLinearLeastSquares(xfit, yfit, yfit_sigma, fit_model)
seed = np.array([3, 2])
fitter.fit(seed, tol=1e-3)
fitter.print_results()

# Plot data and fit
fitter.plot_fit()

# Plot the 1σ and 2σ confidence ellipses
fitter.plot_confidence_ellipses(parx_index=0, pary_index=1)










