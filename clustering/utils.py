import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def match_labels(label_true: np.ndarray[int], label_pred: np.ndarray[int]) -> np.ndarray[int]:
    """
    
    """

    cm: np.ndarray[int] = confusion_matrix(label_true, label_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping: dict[int: int] = {col: row for row, col in zip(row_ind, col_ind)}

    label_pred_matched: np.ndarray[int] = np.vectorize(lambda x: mapping.get(x, x))(label_pred)

    return label_pred_matched