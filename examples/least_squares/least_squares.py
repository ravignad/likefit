# Example of fitting data with the least squares method

import numpy as np

from likelihood import LeastSquares


# fit_model vectorized in x
def fit_model(x, par):
    return par[0] * np.exp(par[1] * x)


xdata = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
ydata = np.array([0.92, 0.884, 0.626, 0.504, 0.481, 0.417, 0.288, 0.302, 0.177, 0.13, 0.158])
ysigma = np.array([0.1, 0.082, 0.067, 0.055, 0.045, 0.037, 0.03, 0.025, 0.02, 0.017, 0.014])

fit = LeastSquares(xdata, ydata, ysigma, fit_model)
seed = np.array([0, 0])
fit(seed)

print(f"Estimators: {fit.get_estimators()}")
print(f"Errors: {fit.get_errors()}")
print(f"Covariance matrix: {fit.get_covariance_matrix()}")
print(f"Correlation matrix: {fit.get_correlation_matrix()}")
print(f"Deviance: {fit.get_deviance()}")
print(f"Degrees of freedom: {fit.get_ndof()}")
print(f"Pvalue: {fit.get_pvalue()}")
