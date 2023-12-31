from typing import Callable, Hashable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import distance
from rapidfuzz.process import cdist
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr


def topographic_similarity(
    x: NDArray[np.int32],
    y: NDArray[np.int32],
    x_dist: str = "Hamming",
    y_dist: str = "Levenshtein",
    correlation: str = "spearmanr",
    x_processor: Callable[[NDArray[np.int32]], Sequence[Hashable]] | None = None,
    y_processor: Callable[[NDArray[np.int32]], Sequence[Hashable]] | None = None,
    normalized: bool = True,
    workers: int = 1,
) -> float:
    """Implement the topographic similarity metric. arXiv:2004.09124"""
    x_dist_type = getattr(distance, x_dist)
    y_dist_type = getattr(distance, y_dist)

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

    corr = getattr(stats, correlation)(x_pdist, y_pdist).correlation
    return corr


def language_similarity(
    x: NDArray[np.int32],
    y: NDArray[np.int32],
    dist: str = "Levenshtein",
    processor: Callable[[NDArray[np.int32]], NDArray[np.int32]] | None = None,
    normalized: bool = True,
) -> float:
    """Calculate the language similarity between two sets of integer sequences."""
    dist_type = getattr(distance, dist)
    if normalized:
        sim = dist_type.normalized_similarity
    else:
        sim = dist_type.similarity

    mean_sim = 0
    for xx, yy in zip(x, y):
        mean_sim += sim(xx, yy, processor=processor)

    return mean_sim / len(x)
