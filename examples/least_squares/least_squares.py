# Example of fitting data with the least squares method

import numpy as np
import pandas as pd

from likelihood import LeastSquares


# fit_model vectorized in x
def fit_model(x, theta):
    return theta[0] * np.exp(theta[1]*x)


input_file = "least_squares.csv"
data = pd.read_csv(input_file)

fit = LeastSquares(data["x"], data["y"], data["dy"], fit_model)
seed = np.array([0, 0])
fit(seed)

print(f"Estimators: {fit.get_estimators()}")
print(f"Errors: {fit.get_errors()}")
print(f"Covariance matrix: {fit.get_covariance_matrix()}")
print(f"Correlation matrix: {fit.get_correlation_matrix()}")
print(f"Deviance: {fit.get_deviance()}")
print(f"Degrees of freedom: {fit.get_ndof()}")
print(f"Pvalue: {fit.get_pvalue()}")
