# Example of a least squares fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from likelihood import LeastSquares


def main():
     input_file = "least_squares.csv"
     data = pd.read_csv(input_file)
     fit = fit_data(data)
     plot_confidence_ellipses(fit)
     plot_fit(fit)


def fit_data(data):

     # fit_model vectorized in x
     def fit_model(x, theta):
          return theta[0] + theta[1] * x + theta[2] * x ** 2

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


# parx_min = estimators[0] - nsigma*errors[0]
# parx_max = estimators[0] + nsigma*errors[0]
# parx = np.linspace(parx_min, parx_max)

# pary_min = estimators[1] - nsigma*errors[1]
# pary_max = estimators[1] + nsigma*errors[1]
# pary = np.linspace(pary_min, pary_max)

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
