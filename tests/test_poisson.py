import pytest
import numpy as np
from scipy.stats import norm
import likefit


# fit_model vectorized in xdata
def fit_model(x, par):
    return par[0] * norm.pdf(x, loc=par[1], scale=par[2])


# Sample data for testing
xdata = np.linspace(start=-2.9, stop=2.9, num=30)
nevents = np.array([0, 2, 5, 8, 7, 18, 15, 27, 34, 51, 55, 63, 67, 75, 90, 78, 73, 70, 62, 51, 33, 26, 30, 17, 15, 14,
                    5, 4, 1, 0])


@pytest.fixture
def poisson_fitter():
    return likefit.Poisson(xdata, nevents, fit_model)


def test_initialization(poisson_fitter):
    assert np.array_equal(poisson_fitter.xdata, xdata)
    assert np.array_equal(poisson_fitter.nevents, nevents)
    assert poisson_fitter.model is fit_model


def test_fit(poisson_fitter):
    seed = np.array([1, 0, 1])
    assert poisson_fitter.fit(seed) == 0
