import os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

import pytest
import numpy as np
from clustering import Sk, InitCentroids, step1


@pytest.mark.parametrize(
    "r, Omega, D, lower, upper, init, X",
    [(Sk(X=np.ones(shape=(10, 20)), Omega=np.ones(shape=(100, 20))), np.ones(shape=(100, 20)), 20, -10, 10, InitCentroids.UNIFORM, np.ones(shape=(10, 20)))]
)
def test_step1_output(r: np.ndarray[complex], Omega: np.ndarray[float], D: int, lower: float | int, upper: float | int, init: InitCentroids, X: np.ndarray[float]) -> None:
    """
    """
    output = step1(r=r, Omega=Omega, D=D, lower=lower, upper=upper, init=init, X=X)
    assert output.shape[0] == D and type(output) == np.ndarray and np.issubdtype(output.dtype, np.floating)

@pytest.fixture
def r() -> np.ndarray[complex]:
    """
    """
    return Sk(X=np.ones(shape=(10, 20)), Omega=np.ones(shape=(100, 20)))


def test_step1_exception(r) -> None:
    """
    """
    # Assert on r type.
    with pytest.raises(AssertionError):   step1(r=None, Omega=np.ones(shape=(100, 20)), D=20, lower=-10, upper=10, init=InitCentroids.UNIFORM, X=np.ones(shape=(10, 20)))

    # Assert on Omega type.
    with pytest.raises(AssertionError):   step1(r=r, Omega=None, D=20, lower=-10, upper=10, init=InitCentroids.UNIFORM, X=np.ones(shape=(10, 20)))

    # Assert on D type.
    with pytest.raises(AssertionError):   step1(r=r, Omega=np.ones(shape=(100, 20)), D=None, lower=-10, upper=10, init=InitCentroids.UNIFORM, X=np.ones(shape=(10, 20)))

    # Assert on lower type.
    with pytest.raises(AssertionError):   step1(r=r, Omega=np.ones(shape=(100, 20)), D=20, lower=None, upper=10, init=InitCentroids.UNIFORM, X=np.ones(shape=(10, 20)))

    # Assert on upper type.
    with pytest.raises(AssertionError):   step1(r=r, Omega=np.ones(shape=(100, 20)), D=20, lower=-10, upper=None, init=InitCentroids.UNIFORM, X=np.ones(shape=(10, 20)))

    # Assert on init type.
    with pytest.raises(AssertionError):   step1(r=r, Omega=np.ones(shape=(100, 20)), D=20, lower=-10, upper=10, init=None, X=np.ones(shape=(10, 20)))

    # Assert on X type.
    with pytest.raises(AssertionError):   step1(r=r, Omega=np.ones(shape=(100, 20)), D=20, lower=-10, upper=10, init=InitCentroids.UNIFORM, X=None)

    # Assert on D different from X and Omega shapes.
    with pytest.raises(AssertionError):   step1(r=r, Omega=np.ones(shape=(100, 20)), D=21, lower=-10, upper=10, init=InitCentroids.UNIFORM, X=np.ones(shape=(10, 20)))

    # Assert on X and Omega shapes.
    with pytest.raises(AssertionError):   step1(r=r, Omega=np.ones(shape=(100, 20)), D=21, lower=-10, upper=10, init=InitCentroids.UNIFORM, X=np.ones(shape=(10, 21)))

    # Assert on lower and upper comparison.
    with pytest.raises(AssertionError):   step1(r=r, Omega=np.ones(shape=(100, 20)), D=20, lower=11, upper=10, init=InitCentroids.UNIFORM, X=np.ones(shape=(10, 20)))