# Example of a least squares fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

from likelihood import LeastSquares


def main():
     input_file = "least_squares.csv"
     data = pd.read_csv(input_file)
     fit = fit_data(data)
     plot_cost_function(fit)
     plot_confidence_ellipses(fit)
     plot_fit(fit)


# fit_model vectorized in x
def fit_model(x, theta):
     return theta[0] + theta[1] * (x-1.2) + theta[2] * (x-1.2) ** 2


def fit_data(data):

     fit = LeastSquares(data["x"], data["y"], data["dy"], fit_model)

     seed = np.array([1, 1, 1])
     fit(seed)

     print(f"Estimators: {fit.get_estimators()}")
     print(f"Errors: {fit.get_errors()}")
     print(f"Covariance matrix: {fit.get_covariance_matrix()}")
     print(f"Correlation matrix: {fit.get_correlation_matrix()}")
     print(f"Deviance: {fit.get_deviance()}")
     print(f"Degrees of freedom: {fit.get_ndof()}")
     print(f"Pvalue: {fit.get_pvalue()}")

     return fit


# Plot confidence ellipses
def plot_confidence_ellipses(fit):

     parx_index = 0
     pary_index = 1

     estimators = fit.get_estimators()

     fig, ax = plt.subplots()
     ax.set_xlabel(f"Parameter {parx_index}")
     ax.set_ylabel(f"Parameter {pary_index}")
     plt.plot(estimators[parx_index], estimators[pary_index], 'o', label="Estimator")
     ellipse1_x, ellipse1_y = fit.get_confidence_ellipse(parx_index, pary_index, nsigma=1)
     plt.plot(ellipse1_x, ellipse1_y, label=r"1σ")
     ellipse2_x, ellipse2_y = fit.get_confidence_ellipse(parx_index, pary_index, nsigma=2)
     plt.plot(ellipse2_x, ellipse2_y, label=r"2σ")
     plt.legend()
     plt.tight_layout()
     plt.show()


def plot_cost_function(fit):

     parx_index = 0
     pary_index = 1
     nsigma = 2

     estimators = fit.get_estimators()
     errors = fit.get_errors()

     parx_min = estimators[parx_index] - nsigma*errors[parx_index]
     parx_max = estimators[parx_index] + nsigma*errors[parx_index]
     parx = np.linspace(parx_min, parx_max, num=50)

     pary_min = estimators[pary_index] - nsigma*errors[pary_index]
     pary_max = estimators[pary_index] + nsigma*errors[pary_index]
     pary = np.linspace(pary_min, pary_max, num=50)

     x, y = np.meshgrid(parx, pary)

     z = fit.vcost_function(parx_index, pary_index, parx, pary)

     fig = plt.figure(figsize=(5, 4))
     ax = fig.subplots(subplot_kw={"projection": "3d"})
     ax.set_xlabel(f"Parameter {parx_index}")
     ax.set_ylabel(f"Parameter {pary_index}")
     ax.set_zlabel(r"$-2\log(L/L_{max})$")

     # Levels of the countour lines
     sigma_levels = np.arange(0, nsigma+2)
     bounds = z.min() + sigma_levels**2
     norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
     ax.plot_surface(x, y, z, cmap=cm.coolwarm, norm=norm, rstride=1, cstride=1, linewidth=0)

     plt.show()


# Plot fit
def plot_fit(fit):

     fig, ax = plt.subplots()
     ax.set_xlabel("x")
     ax.set_ylabel("y")
     ax.errorbar(fit.x, fit.y, fit.ysigma, ls='none', marker='o', label="Data")
     xmin = 1
     xmax = 1.35
     xfit = np.linspace(start=xmin, stop=xmax, num=100)
     yfit = fit.get_yfit(xfit)
     ax.plot(xfit, yfit, ls='--', label="Fit")
     yfit_error = fit.get_yfit_error(xfit)
     ax.fill_between(xfit, yfit - yfit_error, yfit + yfit_error, color='tab:orange', alpha=0.2)
     plt.legend()
     plt.tight_layout()
     plt.show()


if __name__ == '__main__':
     main()
