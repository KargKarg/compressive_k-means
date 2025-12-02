import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.optimize._optimize import OptimizeResult
from tqdm import tqdm
from enum import Enum
from typing import Callable
import inspect
from sklearn.metrics import silhouette_score


class History(Enum):
    """
    Defines the tracking of all centroids at each epochs.

    Enumeration:
        - (0) OFF: int = All the centroids will not be tracked along the epochs.
        - (1) ON: int = All the centroids will be tracked along the epochs.
    """
    OFF: int = 0
    ON: int = 1


class InitCentroids(Enum):
    """
    Defines the method used for the initialisation of the centroids.

    Enumeration:
        - (0) UNIFORM: int = The centroids will be drawn from a uniform distribution starting by "lower", ending by "upper".
        - (1) MEAN: int = The centroids will be computed as the mean of all points.
        - (2) SAMPLE: int = The centroids will drawn from the data distribution uniformely.
    """
    UNIFORM: int = 0
    MEAN: int = 1
    SAMPLE: int = 2


class InitOmega(Enum):
    """
    Defines the method used for the initialisation of the omegas.

    Enumeration:
        - (0) UNIFORM: int = The omegas will be drawn from a uniform distribution starting by "lower", ending by "upper".
        - (1) NORMAL: int = The omegas will be drawn from a normal distribution with µ=0 and s²=1.
    """
    UNIFORM: int = 0
    NORMAL: int = 1


def Sk(X: np.ndarray[float], Omega: np.ndarray[float], Beta: np.ndarray[float] | None = None) -> np.ndarray[complex]:
    """
    Computes the compressive sketch of a dataset using weighted complex random Fourier features.
    This function implements the sketching operator.

    Parameters:
        - (required) X: np.ndarray[float] = Data matrix.
        - (required) Omega: np.ndarray[float] = Frequency matrix.
        - (optional) Beta: np.ndarray[float] | None = Weight vector of each sample.

    Returns:
        - (always) S: np.ndarray[complex] = The complex sketch vector of the data.

    Dimensions:
        - dim(X) = (N, D)
        - dim(Omega) = (m, D)
        - dim(Beta) = (N,)
        - dim(S) = (m,)
    """
    assert isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating) and X.size > 0, ":param X: has to be a non-empty-array of floating numbers."
    assert isinstance(Omega, np.ndarray) and np.issubdtype(Omega.dtype, np.floating) and Omega.size > 0, ":param Omega: has to be a non-empty-array of floating numbers."
    assert (isinstance(Beta, np.ndarray) and np.issubdtype(Beta.dtype, np.floating) and Beta.size > 0) or Beta == None, ":param Beta: has to be a non-empty-array of floating numbers or None."
    assert X.shape[1] == Omega.shape[1], ":param X: and :param Omega: must have the same number of columns."    
    if Beta is not None:    assert Beta.shape[0] == X.shape[0], ":param Beta:'s dimension must match the number of :param X:'s line."

    # Uniform distribution if Beta is not defined.
    if Beta is None:    Beta = np.full(shape=(X.shape[0], ), fill_value=1/X.shape[0])
    return np.array([
        (
            Beta*np.exp(
                (-1j)*np.dot(X, Omega[j])
                )
        ).sum(axis=0)
        for j in range(Omega.shape[0])]) #S [j] = Σ_k β_k exp(−i ⟨X[k], Omega[j]⟩).


def step1(r: np.ndarray[complex], Omega: np.ndarray[float], D: int, lower: float | int, upper: float | int, init: InitCentroids, X: np.ndarray[float], max_iters: int = 3) -> np.ndarray[float]:
    """
    Computes the gradient descent of the negative function step1.

    Parameters:
        - (required) r: np.ndarray[complex] = Residual vector.
        - (required) Omega: np.ndarray[float] = Frequency matrix.
        - (required) D: int = Dimension of the data.
        - (required) lower: float | int = The lower bound of the restricted centroid range.
        - (required) upper: float | int = The upper bound of the restricted centroid range.
        - (required) init: InitCentroids = The initialisation method for the centroids.
        - (required) X: np.ndarray[float] = Data matrix.
        - (optional) max_iters: int = The number of attempt to reach a better minimum.

    Returns:
        - (always) best_c: np.ndarray[float] = The best centroid which minizes the function.

    Dimensions:
        - dim(r) = (m,)
        - dim(Omega) = (m, D)
        - dim(D) = (1,)
        - dim(lower) = (1,)
        - dim(upper) = (1,)
        - dim(X) = (N, D)
        - dim(max_iters) = (1,)
    """
    assert isinstance(r, np.ndarray) and np.issubdtype(r.dtype, np.complexfloating) and r.size > 0, ":param r: has to be a non-empty-array of complex numbers."
    assert isinstance(Omega, np.ndarray) and np.issubdtype(Omega.dtype, np.floating) and Omega.size > 0, ":param Omega: has to be a non-empty-array of floating numbers."
    assert isinstance(D, int) and D > 0, ":param D: must be an integer greater than 0."
    assert (isinstance(lower, (int, float))) and (isinstance(upper, (int, float))) and lower < upper, ":params lower, upper: must be either a float or a integer with :param lower: < :param upper:."
    assert isinstance(init, InitCentroids), ":param init: must be an enumeration of InitCentroids."
    assert isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating) and X.size > 0, ":param X: has to be a non-empty-array of floating numbers."
    assert isinstance(max_iters, int) and max_iters > 0, ":param max_iters: must be an integer greater than 0."

    assert r.shape[0] == Omega.shape[0], ":params r, Omega: must have the same size on the axis=0."
    assert Omega.shape[1] == X.shape[1] == D, ":params Omega, X: must have the size of their first axis == :param D:."

    if init.value == InitCentroids.UNIFORM.value:    init_c: np.ndarray[float] = np.random.uniform(low=lower, high=upper, size=(D,))
    if init.value == InitCentroids.MEAN.value:    init_c: np.ndarray[float] = np.mean(X, axis=0)
    if init.value == InitCentroids.SAMPLE.value:    init_c: np.ndarray[float] = X[np.random.randint(low=0, high=X.shape[0]-1), :]

    # Function of the first step, we minimize it so -f.
    f: Callable[[np.ndarray[float]], np.ndarray[float]] = lambda c : - np.real(np.vdot(np.exp(-1j * Omega @ c), r))

    # Gradient of the aformentionned function with respect to C.
    grad: Callable[[np.ndarray[float]], np.ndarray[float]] = lambda c : - np.real(-1j * (np.conj(r) * np.exp(-1j * Omega @ c)) @ Omega) # dim(grad(C)) = (C.shape[0],)
    
    # Same bounds on each dimensions.
    bounds: list[tuple[float, float]] = [(lower, upper)] * D

    best_c: np.ndarray = np.empty(shape=(D,))
    best_score: float = float("-inf")

    for _ in range(max_iters):

        solver: OptimizeResult = minimize(
            fun = f,
            x0 = init_c,
            jac = grad,
            bounds = bounds,
            method = "L-BFGS-B")
        
        if -solver.fun > best_score:
            best_c = solver.x
            best_score = -solver.fun

    return best_c

def step2(C: np.ndarray[float], Omega: np.ndarray[float], c_new: np.ndarray[float], A_c: np.ndarray[complex]) -> tuple[np.ndarray[float], np.ndarray[complex]]:
    """
    Add a cluster in the group of candidates and update their complex sketch.

    Parameters:
        - (required) C: np.ndarray[float] = Centroids matrix (are stacked).
        - (required) Omega: np.ndarray[float] = Frequency matrix.
        - (required) c_new: np.ndarray[float] = New centroid vector (will be stacked).
        - (required) A_c: np.ndarray[complex] = The complex sketch of each centroids.

    Returns:
        - (always) (C*, A_c*): tuple[np.ndarray[float], np.ndarray[complex]] = The tuple of newly stacked matrix.

    Dimensions:
        - dim(C) = (0...K-1, D) # Since the centroids are added through the algorithm.
        - dim(Omega) = (m, D)
        - dim(c_new) = (D,)
        - dim(A_c) = (0...K-1, m) # Since the centroids are added through the algorithm.
        - dim(C*) = (dim(C)[0] + 1, D)
        - dim(A_c*) = (dim(A_c)[0] + 1, D)
    """
    assert isinstance(C, np.ndarray) and np.issubdtype(C.dtype, np.floating), ":param C: has to be an array of floating numbers."
    assert isinstance(Omega, np.ndarray) and np.issubdtype(Omega.dtype, np.floating) and Omega.size > 0, ":param Omega: has to be a non-empty-array of floating numbers."
    assert isinstance(c_new, np.ndarray) and np.issubdtype(c_new.dtype, np.floating) and c_new.size > 0, ":param c_new: has to be a non-empty-array of floating numbers."
    assert isinstance(A_c, np.ndarray) and np.issubdtype(A_c.dtype, np.complexfloating), ":param A_c: has to be an array of complex numbers."

    assert C.shape[1] == Omega.shape[1] == c_new.shape[0], ":params C, Omega, c_new: must have the same size on the axis=1, axis=1 and axis=0 respectively."
    assert Omega.shape[0] == A_c.shape[1], ":params Omega, A_c: must have the same size on the axis=0 and axis=1 respectively."
    assert A_c.shape[0] == C.shape[0], ":params A_c, C: must have the same size on the axis=0."

    return np.vstack((C, c_new)), np.vstack((A_c, (np.exp(-1j * (c_new @ Omega.T)))))


def step3(zhat: np.ndarray[complex], A_c: np.ndarray[complex], C: np.ndarray[float], K: int) -> tuple[np.ndarray[float], np.ndarray[complex]]:
    """
    Computes the minimization of the function step3 only if we have K+1 centroids.

    It is a Least Square problem on a positive support, but we choose K centroids.

    Parameters:
        - (required) zhat: np.ndarray[complex] = The complex sketch vector of the data.
        - (required) A_c: np.ndarray[complex] = The complex sketch of each centroids.
        - (required) C: np.ndarray[float] = Centroids matrix.
        - (required) K: int = The maximum numbers of centroid.

    Returns:
        - (always) (C*, A_c*): tuple[np.ndarray[float], np.ndarray[complex]] = The tuple of the at most K best centroids and their sketch respectively.

    Dimensions:
        - dim(zhat) = (m,)
        - dim(A_c) = (K+1, m)
        - dim(C) = (K+1, D)
        - dim(K) = (1,)
        - dim(C*) = (K, D)
        - dim(A_c*) = (K, D)
    """
    assert isinstance(zhat, np.ndarray) and np.issubdtype(zhat.dtype, np.complexfloating) and zhat.size > 0, ":param zhat: has to be a non-empty-array of complex numbers."
    assert isinstance(A_c, np.ndarray) and np.issubdtype(A_c.dtype, np.complexfloating), ":param A_c: has to be an array of complex numbers."
    assert isinstance(C, np.ndarray) and np.issubdtype(C.dtype, np.floating), ":param C: has to be an array of floating numbers."
    assert isinstance(K, int) and K > 0, ":param K: must be an integer greater than 0."

    assert A_c.shape[0] == C.shape[0], ":params A_c, C: must have their axis=0 of the same size."
    assert zhat.shape[0] == A_c.shape[1], ":params zhat, A_c: must have the same size on the axis=0 and axis=1 respectively."

    # Transform a complex problem onto a real one, by stacking the value of their real and imaging part.
    b: np.ndarray[float] = np.hstack([zhat.real, zhat.imag]) 
    A: np.ndarray[float] = np.hstack([A_c.real, A_c.imag])

    solver: OptimizeResult = lsq_linear(A=A.T, b=b, bounds=(0, float("inf")))
    Beta: np.ndarray[float] = solver.x

    # Keep at most the best K values.
    mask: np.ndarray[int] = np.argsort(-Beta)[:K]
    return C[mask, :], A_c[mask, :]


def step4(zhat: np.ndarray[complex], A_c: np.ndarray[complex]) -> np.ndarray[float]:
    """
    Computes the minimization of the function step4.

    It is a Least Square problem on a [0; 1] support, the sum must be equal to 1.

    Parameters:
        - (required) zhat: np.ndarray[complex] = The complex sketch vector of the data.
        - (required) A_c: np.ndarray[complex] = The complex sketch of each centroids.

    Returns:
        - (always) Alpha: np.ndarray[float] = The weights vector.

    Dimensions:
        - dim(zhat) = (m,)
        - dim(A_c) = (0...K, m)
        - dim(Alpha) = (dim(A_c), m)
    """
    assert isinstance(zhat, np.ndarray) and np.issubdtype(zhat.dtype, np.complexfloating) and zhat.size > 0, ":param zhat: has to be a non-empty-array of complex numbers."
    assert isinstance(A_c, np.ndarray) and np.issubdtype(A_c.dtype, np.complexfloating), ":param A_c: has to be an array of complex numbers."

    assert zhat.shape[0] == A_c.shape[1], ":params zhat, A_c: must have the same size on the axis=0 and axis=1 respectively."

    # Transform a complex problem onto a real one, by stacking the value of their real and imaging part.
    b: np.ndarray[float] = np.hstack([zhat.real, zhat.imag]) 
    A: np.ndarray[float] = np.hstack([A_c.real, A_c.imag])

    solver: OptimizeResult = minimize(
        fun = lambda z: np.linalg.norm(x=(b - (A.T@z)))**2,
        x0 = np.zeros(shape=(A.shape[0],)),
        constraints = {"type": "eq",
                       "fun": lambda z: np.sum(z) - 1})
    Alpha: np.ndarray[float] = solver.x
    return Alpha


def step5(zhat: np.ndarray[complex], Omega: np.ndarray[float], C: np.ndarray[float], A_c: np.ndarray[complex], Alpha: np.ndarray[float], r: np.ndarray[complex], lower: float | int, upper: float | int, epochs: int = 1000, eta: float | int = 1e-3) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Computes the joint gradient descent of the function step5.

    Parameters:
        - (required) zhat: np.ndarray[complex] = The complex sketch vector of the data.
        - (required) Omega: np.ndarray[float] = Frequency matrix.
        - (required) C: np.ndarray[float] = Centroids matrix.
        - (required) A_c: np.ndarray[complex] = The complex sketch of each centroids.
        - (required) Alpha: np.ndarray[float] = The weights vector.
        - (required) r: np.ndarray[complex] = Residual vector.
        - (required) lower: float | int = The lower bound of the restricted centroid range.
        - (required) upper: float | int = The upper bound of the restricted centroid range.
        - (optional) epochs: int = The number of epochs.
        - (optional) eta: float | int = The parameter eta.
    
    Returns:
        - (always) (C*, Alpha*): tuple[np.ndarray[float], np.ndarray[float]] = The best centroids and Alpha after the joint gradient descent.

    Dimensions:
        - dim(zhat) = (m,)
        - dim(Omega) = (m, D)
        - dim(C) = (K+1, D)
        - dim(A_c) = (dim(K)[0], m)
        - dim(Alpha) = (dim(A_c)[0], m)
        - dim(r) = (m,)
        - dim(lower) = (1,)
        - dim(upper) = (1,)
        - dim(epochs) = (1,)
        - dim(eta) = (1,)
    """
    assert isinstance(zhat, np.ndarray) and np.issubdtype(zhat.dtype, np.complexfloating) and zhat.size > 0, ":param zhat: has to be a non-empty-array of complex numbers."
    assert isinstance(Omega, np.ndarray) and np.issubdtype(Omega.dtype, np.floating) and Omega.size > 0, ":param Omega: has to be a non-empty-array of floating numbers."
    assert isinstance(C, np.ndarray) and np.issubdtype(C.dtype, np.floating), ":param C: has to be an array of floating numbers."
    assert isinstance(A_c, np.ndarray) and np.issubdtype(A_c.dtype, np.complexfloating), ":param A_c: has to be an array of complex numbers."
    assert isinstance(Alpha, np.ndarray) and np.issubdtype(Alpha.dtype, np.floating) and Alpha.size > 0, ":param Alpha: has to be a non-empty-array of floating numbers."
    assert isinstance(r, np.ndarray) and np.issubdtype(r.dtype, np.complexfloating) and r.size > 0, ":param r: has to be a non-empty-array of complex numbers."
    assert isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower < upper, ":params lower, upper: must be either a float or a integer with :param lower: < :param upper:."
    assert isinstance(epochs, int) and epochs > 0, ":param epochs: must be an integer greater than 0."
    assert isinstance(eta, (int, float)) and eta > 0, ":param eta: must be an integer or a float greater than 0."

    assert C.shape[0] == Alpha.shape[0] == A_c.shape[0], ":params C, Alpha, A_c: must have the same size on the axis=0."
    assert zhat.shape[0] == Omega.shape[0] == A_c.shape[1] == r.shape[0], ":params zhat, Omega, A_c, r: must have the same size on the axis=0, axis=0, axis=1 and axis=0 respectively."

    for _ in range(epochs):
        
        # Gradient of the aformentionned function with respect to C.
        grad: np.ndarray[float] = np.real(1j * Alpha[:, None] * ((np.conj(r)[None, :] * A_c) @ Omega))  # dim(grad(C)) = (C.shape[0],)

        C -= eta*grad
        C = np.clip(C, lower, upper)

        # Compute the new matrix.
        A_c = np.exp(-1j * (C @ Omega.T))

        # Compute the Alpha given the new A_c
        Alpha = step4(zhat=zhat, A_c=A_c)

        # Update the residual
        r = zhat - A_c.T @ Alpha

    return C, Alpha



class CKM:
    """
    Class for the Compressive-K-Means Algorithm.

    Property:
        - m: int = The dimension of the sketch.
        - seed: int | float = The seed for the runs.
        - history: History = The history parameter.
        - init_centroids: InitCentroids = The initialisation of centroids parameter.
        - Omega: np.ndarray[float] = The frequency matrix.
        - centroids: np.ndarray[float] = The centroids matrix.
        - all_centroids: list[np.ndarray[float]] | None = The list of all centroids matrices through the optimisation.

    Methods:
        - fit: Callable[[np.ndarray[float]], None] = Learn the position of each cluster on a data-set X.
        - predict: Callable[[np.ndarray[float]], np.ndarray[int]] = Assign at each data point its cluster.
    """

    def __init__(self, K: int, m: int, lower: float, upper: float, seed: int | float = 12, iters: int = 1, metric: Callable[[np.ndarray[float], np.ndarray[int]], float] = silhouette_score, history: History = History.OFF, init_centroids: InitCentroids = InitCentroids.UNIFORM) -> None:
        """
        Defines the properties of the algorithm.

        Parameters:
            - (required) K : int = The number of clusters.
            - (required) m : int = The dimension of the sketch.
            - (required) lower: float | int = The lower bound for the Algorithm.
            - (required) upper: float | int = The upper bound for the Algorithm.
            - (optional) seed: float | int = The seed for the randomness.
            - (optional) iters: int = The number of runs to find the best clustering.
            - (optional) metric: Callable[[np.ndarray[float], np.ndarray[int]], float] = Metric to maximize, will be used to assess the best clustering.
            - (optional) history: History = The parameter for the clinitialisation.
            - (optional) init_centroids: InitCentroids = The parameter for the clusters initialisation.            

        Returns:
            - (always) None.

        Dimensions:
            - dim(K) = (1,)
            - dim(m) = (1,)
            - dim(lower) = (1,)
            - dim(upper) = (1,)
            - dim(seed) = (1,)       
        """
        self._centroids: np.ndarray[float] | None = None

        self._K: int = K
        self.m: int = m
        self._lower: float = lower
        self._upper: float = upper
        self.seed: int = seed
        self._iters: int = iters
        self._metric: Callable[[np.ndarray[float], np.ndarray[int]], float] = metric
        self.history: History = history
        self.init_centroids: InitCentroids = init_centroids
        
        return self.__post_init__()
    
    
    def __post_init__(self) -> None:
        """
        Runs the asserts after the initialisation.
        """

        assert (self._K > 0) and (isinstance(self._K, int)), ":param K: must be an integer greater than zero."
        assert (self.m > 0) and (isinstance(self.m, int)), ":param m: must be an integer greater than zero."
        assert (isinstance(self._lower, (int, float))) and (isinstance(self._upper, (int, float))) and self._lower < self._upper, ":params lower, upper: must be either a float or a integer with :param lower: < :param upper:."
        assert isinstance(self.seed, int), ":param seed: must be either a float or a integer."
        assert isinstance(self._iters, int) and self._iters > 0, ":param iters: must be an integer greater than zero."
        assert isinstance(self._metric, Callable) and len(inspect.signature(self._metric).parameters) >= 2, ":param metric: must be a callable of at least 2 parameters."
        assert isinstance(self.history, History), ":param history: must be an enumeration of History."
        assert isinstance(self.init_centroids, InitCentroids), ":param init_centroids: must be an enumeration of InitCentroids."

        if self.history.value == History.ON.value:  self._all_centroids: list[np.ndarray[float]] = []

        return None
    

    def draw_Omega(self, X: np.ndarray[float], block: int, frac_m: int = 10, frac_n: int = 10, n_iters: int = 5) -> np.ndarray[float]:
        """
        Property which defines the frequency matrix.

        Parameters:
            - (required) None.

        Returns:
            - (always) Omega: np.ndarray[float] = Frequency matrix.

        Dimensions:
            - dim(Omega) = (m, D)
        """
        np.random.seed(seed=self.seed)

        N, D = X.shape
        
        m0: int = self.m // frac_m
        n0: int = N // frac_n

        # Initialization
        sigma = 1

        for _ in range(n_iters):
            
            # Draw some frequencies adapted to the current σ^2
            Omega: np.ndarray[float] = np.random.multivariate_normal(np.zeros(D), np.linalg.inv(sigma * np.identity(D)), size=m0)

            # Compute the radius
            rad: np.ndarray[float] = np.linalg.norm(Omega, axis=1)**2

            # Sort the omegas according to their frequencies
            Omega: np.ndarray[float] = Omega[(idx := np.argsort(rad))]
            rad: np.ndarray[float] = rad[idx]

            # Small empirical sketch
            z: np.ndarray[complex] = Sk(Omega, X[np.random.choice(np.arange(N), size=(n0,), replace=False)])

            # Find maximum peak in each block
            J: np.ndarray[complex] = np.zeros(block)
            R: np.ndarray[float] = np.zeros(block)
            
            for q in range(block):

                idx: slice = slice(q * ((m0) // block), (q + 1) * ((m0) // block))

                block_z: np.ndarray[complex] = z[idx]
                block_r: np.ndarray[float] = rad[idx]

                kmax: slice = np.argmax(np.abs(block_z))

                # Construction of ê
                J[q] = np.abs(block_z[kmax])

                # Construction of R_jq
                R[q] = block_r[kmax]
            
            # Find the optimal σ^2 by a lsq problem
            sigma: float = minimize(
                fun = lambda sigma: np.mean((J - np.exp(-0.5 * R * sigma))**2),
                x0 = sigma,
                bounds = [(0, None)],
                method = "L-BFGS-B").x[0]
        
        Omega: np.ndarray[float] = np.random.multivariate_normal(np.zeros(D), np.linalg.inv(sigma * np.identity(D)), size=self.m)

        return Omega

    @property
    def cluster_centers_(self) -> np.ndarray[float]:
        """
        Property which defines the centroids matrix.

        Parameters:
            - (required) None.

        Returns:
            - (always) self._centroids: np.ndarray[float] = Centroids matrix.

        Dimensions:
            - dim(self._centroids) = (0...K, D)
        """
        if self._centroids is None: self._centroids = np.empty(shape=(0, self._D))
        return self._centroids
    

    @cluster_centers_.setter
    def cluster_centers_(self, value: np.ndarray[float]) -> None:
        """
        Setter which defines the replacement of the "self._centroids"'s value.

        Parameters:
            - (required) value: np.ndarray[float] = The new value for the centroids.

        Returns:
            - (always) None.

        Dimensions:
            - dim(value) = (0...K, D)
        """
        assert isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating) and value.size > 0, "The centroids have to be a non-empty-array of floating numbers."
        self._centroids = value
        return None
    

    @property
    def all_centroids(self) -> list[np.ndarray[float]] | None:
        """
        Property which defines the list of all centroids matrix through the optimisation.

        Parameters:
            - (required) None.

        Returns:
            - (if History.ON) self._all_centroids: list[np.ndarray[float]] = List of all centroids matrix.
            - (else) None.

        Dimensions:
            - dim(self._all_centroids) = [steps, 0...K, D]
        """
        if self.history.value == History.ON.value:  return self._all_centroids

        else:
            print("History was turned off.")
            return None
    

    def fit(self, X: np.ndarray[float]) -> None:
        """
        Learning function on the data X.

        Parameters:
            - (required) X: np.ndarray[float] = The data matrix.

        Returns:
            - (always) None.
        
        Dimensions:
            - dim(X) = (N, D)
        """
        assert isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating) and X.size > 0, ":param X: has to be a non-empty-array of floating numbers."

        np.random.seed(seed=self.seed)
        self._D: int = X.shape[1]

        Omega: np.ndarray[float] = self.draw_Omega(X=X, block=self._K)

        zhat: np.ndarray[complex] = Sk(X=X, Omega=Omega)

        best_c: np.ndarray[float] = np.empty(shape=(self._K, self._D))
        best_metric: float = float("-inf")
        best_history: list[np.ndarray[float]] = []

        for _ in tqdm(range(self._iters), desc="ITERS"):

            r: np.ndarray[complex] = zhat.copy()

            A_c: np.ndarray[complex] = np.exp(-1j * (self.cluster_centers_ @ Omega.T))

            for _ in range(1, 2*self._K+1):

                c_new: np.ndarray[float] = step1(r=r, Omega=Omega, D=self._D, lower=self._lower, upper=self._upper, init=self.init_centroids, X=X)

                self.cluster_centers_, A_c = step2(C=self.cluster_centers_, Omega=Omega, c_new=c_new, A_c=A_c)

                if self.cluster_centers_.shape[0] > self._K:
                    self.cluster_centers_, A_c = step3(zhat=zhat, A_c=A_c, C=self.cluster_centers_, K=self._K)

                Alpha: np.ndarray[float] = step4(zhat=zhat, A_c=A_c)

                self.cluster_centers_, Alpha = step5(zhat=zhat, Omega=Omega, C=self.cluster_centers_, A_c=A_c, Alpha=Alpha, r=zhat - A_c.T @ Alpha, lower=self._lower, upper=self._upper, epochs=100, eta=1e-2)

                if self.history.value == History.ON.value:  self._all_centroids.append(self.cluster_centers_)

                r = zhat - A_c.T @ Alpha

            pred: np.ndarray[int] = self.predict(X=X)
            metric_value: float = self._metric(X, pred)

            if metric_value > best_metric:  best_c, best_history = np.copy(self.cluster_centers_), self._all_centroids.copy() if self.history.value == History.ON.value else []

            self._centroids = np.empty(shape=(0, self._D))

            self.seed += 1

        self.cluster_centers_ = np.copy(best_c)
        self.seed -= (self._iters - 1)
        self._all_centroids = best_history.copy() if self.history.value == History.ON.value else None

        return None
    
    
    def predict(self, X: np.ndarray[float]) -> np.ndarray[int]:
        """
        Predicting function on the data X, each point has a int assigned which defines the nearest cluster.

        Parameters:
            - (required) X: np.ndarray[float] = The data matrix.

        Returns:
            - (always) np.ndarray[int] = The predictions vector.

        Dimensions:
            - dim(X) = (N, D)
            - dim(pred) = (N,)
        """
        assert isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating) and X.size > 0, ":param X: has to be a non-empty-array of floating numbers."
        return np.argmin(a=np.linalg.norm(x=(X[:, None, :] - self.cluster_centers_), axis=2), axis=1)
    

__all__ = ["Sk",
           "step1",
           "step2",
           "step3",
           "step4",
           "step5",
           "CKM",
           "History",
           "InitCentroids",
           "InitOmega",]