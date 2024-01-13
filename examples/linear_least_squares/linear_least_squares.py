# Example of fitting data with a linear least squares method

import numpy as np
import likefit

xdata = np.array([-0.18, -0.14, -0.1, -0.06, -0.02,  0.02,  0.06,  0.1,  0.14])
ydata = np.array([2.243, 2.217, 2.201, 2.175, 2.132, 2.116, 2.083, 2.016, 2.004])
ysigma = np.array([0.008, 0.008, 0.01, 0.009, 0.011, 0.016, 0.018, 0.021, 0.017])


# Model linear in the parameters
def fit_model(x, par):
    return par[0] + par[1] * x


# Create the fitter
fitter = likefit.LinearLeastSquares(xdata, ydata, ysigma, fit_model)

# Fit the data
fitter.fit()

# Output the fit results
fitter.print_results()

# Plot data and fit
fitter.plot_fit()

# Plot the confidence ellipses
fitter.plot_confidence_ellipses(parx_index=0, pary_index=1, xlabel="y0", ylabel="m")

# Plot the cost function
fitter.plot_cost_function(parx_index=0, pary_index=1, xlabel="y0", ylabel="m")
