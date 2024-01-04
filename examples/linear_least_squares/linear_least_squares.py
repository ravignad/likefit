# Example of fitting data with a linear least squares method

import numpy as np
import matplotlib.pyplot as plt

from fitpy import LinearLeastSquares


# fit_model vectorized in x
def fit_model(x, par):
    return par[0] + par[1] * (x-1.2)


xdata = np.array([1.02, 1.06, 1.1 , 1.14, 1.18, 1.22, 1.26, 1.3 , 1.34])
ydata = np.array([2.243, 2.217, 2.201, 2.175, 2.132, 2.116, 2.083, 2.016, 2.004])
ysigma = np.array([0.008, 0.008, 0.01 , 0.009, 0.011, 0.016, 0.018, 0.021, 0.017])
npar = 2

fitter = LinearLeastSquares(xdata, ydata, ysigma, npar, fit_model)

fitter.fit()

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
