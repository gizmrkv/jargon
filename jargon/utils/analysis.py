from typing import Callable, Hashable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from rapidfuzz.distance import (
    OSA,
    DamerauLevenshtein,
    Hamming,
    Indel,
    Jaro,
    JaroWinkler,
    LCSseq,
    Levenshtein,
    Postfix,
    Prefix,
)
from rapidfuzz.process import cdist
from scipy.stats import kendalltau, pearsonr, spearmanr

str2distance = {
    "DamerauLevenshtein": DamerauLevenshtein,
    "Levenshtein": Levenshtein,
    "Hamming": Hamming,
    "Indel": Indel,
    "Jaro": Jaro,
    "JaroWinkler": JaroWinkler,
    "LCSseq": LCSseq,
    "OSA": OSA,
    "Postfix": Postfix,
    "Prefix": Prefix,
}


def topographic_similarity(
    x: NDArray[np.int32],
    y: NDArray[np.int32],
    x_dist: str = "Hamming",
    y_dist: str = "Levenshtein",
    correlation: Literal["spearman", "kendall", "pearson"] = "spearman",
    x_processor: Callable[[NDArray[np.int32]], Sequence[Hashable]] | None = None,
    y_processor: Callable[[NDArray[np.int32]], Sequence[Hashable]] | None = None,
    normalized: bool = True,
    workers: int = 1,
) -> float:
    """Implement the topographic similarity metric. arXiv:2004.09124

    Calculate the topographic similarity between two sets of integer sequences.
    Quantitatively evaluate the compositionality when x is regarded as meaning and y as expression.
    If all values of x are the same or all values of y are the same, the calculation will not be correct.

    Parameters
    ----------
    x : NDArray[np.int32]
        The first set of integer sequences.
    y : NDArray[np.int32]
        The second set of integer sequences.
    x_dist : str, optional
        The distance metric for x, by default "Hamming"
    y_dist : str, optional
        The distance metric for y, by default "Levenshtein"
    correlation : Literal["spearman", "kendall", "pearson"], optional
        The method for evaluating the correlation of calculated similarities, by default "spearman"
    x_processor : Callable[[NDArray[np.int32]], Sequence[Hashable]] | None, optional
        Custom processor function for x, by default None
    y_processor : Callable[[NDArray[np.int32]], Sequence[Hashable]] | None, optional
        Custom processor function for y, by default None
    normalized : bool, optional
        Whether to use normalized distances, by default True
    workers : int, optional
        The number of parallel workers, by default 1

    Returns
    -------
    float
        The topographic similarity between x and y.

    Examples
    --------
    >>> x = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])
    >>> y = np.array([[1, 2, 3], [0, 2, 3], [1, 2, 2], [0, 2, 3]])
    >>> topographic_similarity(x, y)
    0.33541019662496846
    """
    x_dist_type = str2distance[x_dist]
    y_dist_type = str2distance[y_dist]

    if normalized:
        x_dist_scorer = x_dist_type.normalized_distance
        y_dist_scorer = y_dist_type.normalized_distance
    else:
        x_dist_scorer = x_dist_type.distance
        y_dist_scorer = y_dist_type.distance

    x_dmat = cdist(x, x, scorer=x_dist_scorer, processor=x_processor, workers=workers)
    y_dmat = cdist(y, y, scorer=y_dist_scorer, processor=y_processor, workers=workers)

    x_pdist = x_dmat[np.triu_indices(n=x_dmat.shape[0], k=1)]
    y_pdist = y_dmat[np.triu_indices(n=y_dmat.shape[0], k=1)]

    corrs = {
        "spearman": spearmanr,
        "kendall": kendalltau,
        "pearson": pearsonr,
    }
    corr: float = corrs[correlation](x_pdist, y_pdist).correlation
    return corr


"""
Calculate the language similarity between two sets of integer sequences.

Args:
    x (NDArray[np.int32]): The first set of integer sequences.
    y (NDArray[np.int32]): The second set of integer sequences.
    dist (str, optional): The distance metric. Defaults to "Levenshtein".
    processor (Callable[[NDArray[np.int32]], NDArray[np.int32]] | None, optional):
        Custom processor function. Defaults to None.
    normalized (bool, optional): Whether to use normalized similarities. Defaults to True.

Returns:
    float: The mean language similarity between x and y.
"""


def language_similarity(
    x: NDArray[np.int32],
    y: NDArray[np.int32],
    dist: str = "Levenshtein",
    processor: Callable[[NDArray[np.int32]], NDArray[np.int32]] | None = None,
    normalized: bool = True,
) -> float:
    """Calculate the language similarity between two sets of integer sequences.

    Parameters
    ----------
    x : NDArray[np.int32]
        The first set of integer sequences.
    y : NDArray[np.int32]
        The second set of integer sequences.
    dist : str, optional
        The distance metric, by default "Levenshtein"
    processor : Callable[[NDArray[np.int32]], NDArray[np.int32]] | None, optional
        Custom processor function, by default None
    normalized : bool, optional
        Whether to use normalized similarities, by default True

    Returns
    -------
    float
        The mean language similarity between x and y.

    Examples
    --------
    >>> x = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])
    >>> y = np.array([[2, 2], [1, 3], [1, 1], [2, 3]])
    >>> language_similarity(x, y)
    0.625
    """
    dist_type = str2distance[dist]
    if normalized:
        sim = dist_type.normalized_similarity
    else:
        sim = dist_type.similarity

    mean_sim = 0
    for xx, yy in zip(x, y):
        mean_sim += sim(xx, yy, processor=processor)

    return mean_sim / len(x)
