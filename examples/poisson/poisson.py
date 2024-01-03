# Example of fitting data with the least squares method

import numpy as np
from scipy.stats import norm

from likelihood import Poisson


# fit_model vectorized in x
def fit_model(x, par):
    return par[0] * norm.pdf(x, loc=par[1], scale=par[2])


xdata = np.linspace(start=-2.9, stop=2.9, num=30)
ydata = np.array([0, 2, 5, 8, 7, 18, 15, 27, 34, 51, 55, 63, 67, 75, 90, 78, 73, 70, 62, 51, 33, 26, 30, 17, 15, 14, 5, 4, 1, 0])

fit = Poisson(xdata, ydata, fit_model)
seed = np.array([1, 0, 1])
fit(seed)

print(f"Estimators: {fit.get_estimators()}")
print(f"Errors: {fit.get_errors()}")
print(f"Covariance matrix: {fit.get_covariance_matrix()}")
print(f"Correlation matrix: {fit.get_correlation_matrix()}")
print(f"Deviance: {fit.get_deviance()}")
print(f"Degrees of freedom: {fit.get_ndof()}")
print(f"Pvalue: {fit.get_pvalue()}")
