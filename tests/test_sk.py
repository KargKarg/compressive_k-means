import os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

import pytest
import numpy as np
from clustering import Sk

@pytest.mark.parametrize(
    "X,Omega,expected",
    [(np.zeros(shape=(10, 20)), np.zeros(shape=(100, 20)), np.ones(shape=(100,)))]
)
def test_sk_value(X: np.ndarray[float], Omega: np.ndarray[float], expected: np.ndarray[complex]) -> None:
    """
    """
    # Assert on the operation.
    assert np.all(Sk(X=X, Omega=Omega) == expected)


@pytest.mark.parametrize(
    "X,Omega",
    [(np.zeros(shape=(10, 20)), np.zeros(shape=(100, 20)))]
)
def test_sk_output(X: np.ndarray[float], Omega: np.ndarray[float]) -> None:
    """
    """
    # Assert on the output.
    output = Sk(X=X, Omega=Omega)
    assert type(output) == np.ndarray and np.issubdtype(output.dtype, np.complexfloating)


def test_sk_exception() -> None:
    """
    """
    # Assert on X type.
    with pytest.raises(AssertionError):   Sk(X=None, Omega=np.random.random(size=(100, 20)))

    # Assert on Omega type.
    with pytest.raises(AssertionError):   Sk(X=np.random.random(size=(10, 20)), Omega=None)

    # Assert on Beta type.
    with pytest.raises(AssertionError):   Sk(X=np.random.random(size=(10, 20)), Omega=np.random.random(size=(100, 20)), Beta=1)

    # Assert on X and Omega shapes.
    with pytest.raises(AssertionError):   Sk(X=np.random.random(size=(10, 21)), Omega=np.random.random(size=(100, 20)))

    # Assert on X's values types.
    with pytest.raises(AssertionError):   Sk(X=(-1j)*np.random.random(size=(10, 21)), Omega=np.random.random(size=(100, 20)))

    # Assert on Omega's values types.
    with pytest.raises(AssertionError):   Sk(X=np.random.random(size=(10, 21)), Omega=(-1j)*np.random.random(size=(100, 20)))