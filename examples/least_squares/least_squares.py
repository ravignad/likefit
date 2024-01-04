# Example of fitting data with the least squares method

import numpy as np
import matplotlib.pyplot as plt

from fitpy import LeastSquares


# fit_model vectorized in x
def fit_model(x, par):
    return par[0] * np.exp(par[1] * x)


xdata = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
ydata = np.array([0.92, 0.884, 0.626, 0.504, 0.481, 0.417, 0.288, 0.302, 0.177, 0.13, 0.158])
ysigma = np.array([0.1, 0.082, 0.067, 0.055, 0.045, 0.037, 0.03, 0.025, 0.02, 0.017, 0.014])

fitter = LeastSquares(xdata, ydata, ysigma, fit_model)
seed = np.array([0, 0])
fitter.fit(seed)

print(f"Estimators: {fitter.get_estimators()}")
print(f"Errors: {fitter.get_errors()}")
print(f"Covariance matrix: {fitter.get_covariance_matrix()}")
print(f"Correlation matrix: {fitter.get_correlation_matrix()}")
print(f"Deviance: {fitter.get_deviance()}")
print(f"Degrees of freedom: {fitter.get_ndof()}")
print(f"Pvalue: {fitter.get_pvalue()}")

# Plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y")

# Plot data
ax.errorbar(fitter.x, fitter.y, fitter.ysigma, ls='none', marker='o', label="Data")

# Plot fitter
xmin = xdata.min()
xmax = xdata.max()
xfit = np.linspace(start=xmin, stop=xmax, num=100)
yfit = fitter.get_yfit(xfit)
ax.plot(xfit, yfit, ls='--', label="Fit")

# Plot error band
yfit_error = fitter.get_yfit_error(xfit)
ax.fill_between(xfit, yfit - yfit_error, yfit + yfit_error, color='tab:orange', alpha=0.2)

plt.legend()
plt.tight_layout()
plt.show()

# plt.savefig("least_squares.png")
