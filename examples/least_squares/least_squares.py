# Example of a least squares fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from likelihood import LeastSquares


input_file = "least_squares.csv"
data = pd.read_csv(input_file)
 
# fit_model vectorized in x 
def fit_model(x, theta):
     return theta[0] + theta[1] * x + theta[2] * x**2 

fit = LeastSquares(data["x"], data["beta"], data["dbeta"], fit_model)

seed = np.array([1, 1, 1])

fit_result = fit(seed)

print(f"Estimators: {fit.get_estimators()}")
print(f"Errors: {fit.get_errors()}")
print(f"Covariance matrix: {fit.get_covariance_matrix()}")
print(f"Correlation matrix: {fit.get_correlation_matrix()}")
print(f"Deviance: {fit.get_deviance()}")
print(f"Degrees of freedom: {fit.get_ndof()}")
print(f"Pvalue: {fit.get_pvalue()}")

# Plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel(r"$\beta$")

ax.errorbar(data["x"], data["beta"], data["dbeta"], ls='none', marker='o', label="Data")

xmin = 1
xmax = 1.35
xfit = np.linspace(start=xmin, stop=xmax, num=100)

yfit = fit.get_yfit(xfit)
ax.plot(xfit, yfit, ls='--', label="Fit")

yfit_error = fit.get_yfit_error(xfit)
ax.fill_between(xfit, yfit-yfit_error, yfit+yfit_error, color='tab:orange', alpha=0.2)

plt.legend()
plt.tight_layout()

plt.show()
