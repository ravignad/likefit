import pytest
import numpy as np
import likefit


# fit_model is sigmoid function vectorized in xdata
def fit_model(x, par):
    return 1 / (1+np.exp(-(x-par[0])/par[1]))


# Sample data for testing
xdata = np.arange(start=0.05, stop=1.05, step=0.05)
ntrials = np.full(xdata.shape, fill_value=30)
nsuccess = np.array([0, 0, 0, 3, 3, 2, 8, 5, 4, 11, 18, 15, 19, 20, 26, 24, 26, 29, 30, 30])


@pytest.fixture
def binomial_fitter():
    return likefit.Binomial(xdata, ntrials, nsuccess, fit_model)


def test_initialization(binomial_fitter):
    assert np.array_equal(binomial_fitter.xdata, xdata)
    assert np.array_equal(binomial_fitter.ntrials, ntrials)
    assert np.array_equal(binomial_fitter.nsuccess, nsuccess)
    assert binomial_fitter.model is fit_model


def test_fit(binomial_fitter):
    seed = np.array([0.5, 1])
    assert binomial_fitter.fit(seed, tol=1e-4) == 0


def test_cost_function(binomial_fitter):
    # Test the cost function with some dummy parameters
    par = np.array([0.5, 1])
    cost = binomial_fitter.cost_function(par)
    assert cost == 2619.431668583625


def test_get_ydata(binomial_fitter):
    ydata = binomial_fitter.get_ydata()
    assert np.array_equal(ydata, nsuccess/ntrials)
