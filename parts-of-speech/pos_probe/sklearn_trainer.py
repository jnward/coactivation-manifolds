"""Scikit-learn training utilities for linear POS probes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .constants import POS_TAGS
from .probe import LinearProbe


@dataclass
class SklearnTrainResult:
    estimator: LogisticRegression
    probe: LinearProbe


def train_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 200,
    random_state: int | None = 42,
    class_weight: str | dict | None = None,
) -> LogisticRegression:
    """Train a multinomial logistic regression probe."""

    clf = LogisticRegression(
        penalty="l2",
        C=C,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=max_iter,
        n_jobs=None,
        verbose=0,
        random_state=random_state,
        class_weight=class_weight,
    )
    clf.fit(features, labels)
    return clf


def train_one_vs_rest_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 200,
    random_state: int | None = 42,
    class_weight: str | dict | None = "balanced",
    hidden_size: int,
) -> LinearProbe:
    """Train separate binary classifiers per tag and combine into a LinearProbe."""

    num_labels = len(POS_TAGS)
    weight_rows = []
    bias_rows = []

    for tag_idx in range(num_labels):
        binary = (labels == tag_idx).astype(np.int32)
        positive = binary.sum()
        if positive == 0:
            raise ValueError(f"No positive examples found for tag {POS_TAGS[tag_idx]}.")

        clf = LogisticRegression(
            penalty="l2",
            C=C,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=max_iter,
            n_jobs=None,
            verbose=0,
            random_state=None if random_state is None else random_state + tag_idx,
            class_weight=class_weight,
        )
        clf.fit(features, binary)
        weight_rows.append(clf.coef_[0])
        bias_rows.append(clf.intercept_[0])

    weight = np.stack(weight_rows, axis=0)
    bias = np.asarray(bias_rows)

    probe = LinearProbe(hidden_size, num_labels)
    with torch.no_grad():
        probe.linear.weight.copy_(torch.from_numpy(weight).to(torch.float32))
        probe.linear.bias.copy_(torch.from_numpy(bias).to(torch.float32))
    return probe


def to_linear_probe(
    clf: LogisticRegression,
    *,
    hidden_size: int,
) -> LinearProbe:
    """Convert a trained sklearn model to our torch LinearProbe."""

    num_labels = len(POS_TAGS)
    classes = clf.classes_
    expected = np.arange(num_labels)
    if len(classes) != num_labels or not np.array_equal(classes, expected):
        raise ValueError(
            f"LogisticRegression classes {classes} do not match expected {expected}."
        )

    probe = LinearProbe(hidden_size, num_labels)
    with torch.no_grad():
        probe.linear.weight.copy_(torch.from_numpy(clf.coef_).to(torch.float32))
        probe.linear.bias.copy_(torch.from_numpy(clf.intercept_).to(torch.float32))
    return probe
