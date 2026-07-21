import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

#### KNN Classification with cross-validation for robust result

def knn_classification_accuracy(
    emb1,
    emb2,
    n_neighbors=5,
    n_splits=5,
    random_state=42,
):
    """
    Evaluate KNN classification accuracy between two embedding sets.

    Parameters
    ----------
    emb1 : np.ndarray or torch.Tensor
        Embeddings for class 0, shape (N1, D).
    emb2 : np.ndarray or torch.Tensor
        Embeddings for class 1, shape (N2, D).
    n_neighbors : int
        Number of neighbors for KNN.
    n_splits : int
        Number of folds for cross-validation.
    random_state : int
        Random seed.

    Returns
    -------
    mean_acc : float
        Mean cross-validation accuracy.
    std_acc : float
        Standard deviation of accuracy.
    """

    # Convert PyTorch tensors to NumPy
    if hasattr(emb1, "detach"):
        emb1 = emb1.detach().cpu().numpy()
    if hasattr(emb2, "detach"):
        emb2 = emb2.detach().cpu().numpy()

    X = np.concatenate([emb1, emb2], axis=0)
    y = np.concatenate([
        np.zeros(len(emb1), dtype=int),
        np.ones(len(emb2), dtype=int)
    ])

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")

    print(f"KNN ({n_neighbors} neighbors)")
    print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores.mean(), scores.std()



### Gradient boost classifier

def gradboost_classification_accuracy(
    emb1,
    emb2,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    n_splits=5,
    random_state=42,
):
    """
    Evaluate embedding quality using a Gradient Boosting classifier.

    Parameters
    ----------
    emb1 : np.ndarray or torch.Tensor
        Embeddings for class 0, shape (N1, D).
    emb2 : np.ndarray or torch.Tensor
        Embeddings for class 1, shape (N2, D).

    Returns
    -------
    mean_acc : float
    std_acc : float
    """

    if hasattr(emb1, "detach"):
        emb1 = emb1.detach().cpu().numpy()
    if hasattr(emb2, "detach"):
        emb2 = emb2.detach().cpu().numpy()

    X = np.concatenate([emb1, emb2], axis=0)
    y = np.concatenate([
        np.zeros(len(emb1), dtype=int),
        np.ones(len(emb2), dtype=int)
    ])

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    print("Gradient Boosting")
    print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores.mean(), scores.std()


### XGBoost

def xgboost_classification_accuracy(
    emb1,
    emb2,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_splits=5,
    random_state=42,
):
    """
    Evaluate embedding quality using an XGBoost classifier.

    Parameters
    ----------
    emb1 : np.ndarray or torch.Tensor
        Embeddings for class 0.
    emb2 : np.ndarray or torch.Tensor
        Embeddings for class 1.

    Returns
    -------
    mean_acc : float
    std_acc : float
    """

    if hasattr(emb1, "detach"):
        emb1 = emb1.detach().cpu().numpy()
    if hasattr(emb2, "detach"):
        emb2 = emb2.detach().cpu().numpy()

    X = np.concatenate([emb1, emb2], axis=0)
    y = np.concatenate([
        np.zeros(len(emb1), dtype=int),
        np.ones(len(emb2), dtype=int)
    ])

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    print("XGBoost")
    print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores.mean(), scores.std()






###============================== Classification as downstream task pipeline===================================

"""
downstream_embedding_classifier.py

Reusable downstream classification utilities for two sets of embedding vectors.

Assumption
----------
The two arrays correspond to two classes. By default, 80% of each class is
used for training and 20% for testing.

Supported classifiers
---------------------
- kNN
- Logistic regression
- Gradient boosting
- Random forest
- MLP

Example
-------
from downstream_embedding_classifier import evaluate_two_embedding_sets

result = evaluate_two_embedding_sets(
    embeddings_class0=ttbar_embeddings,
    embeddings_class1=ggf_embeddings,
    classifier="knn",
    train_fraction=0.8,
    classifier_kwargs={"n_neighbors": 15},
    random_state=42,
)

print(result.metrics)
print(result.confusion_matrix)
"""



from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union
import json

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ClassifierName = Literal[
    "knn",
    "logistic_regression",
    "gradient_boosting",
    "random_forest",
    "mlp",
]


@dataclass
class DownstreamResult:
    """Container holding the fitted model and downstream evaluation outputs."""

    model: BaseEstimator
    metrics: Dict[str, float]
    confusion_matrix: NDArray[np.int64]
    classification_report: Dict[str, Any]
    train_embeddings: NDArray[np.float64]
    test_embeddings: NDArray[np.float64]
    train_labels: NDArray[np.int64]
    test_labels: NDArray[np.int64]
    predictions: NDArray[np.int64]
    probabilities: Optional[NDArray[np.float64]]
    train_indices_class0: NDArray[np.int64]
    test_indices_class0: NDArray[np.int64]
    train_indices_class1: NDArray[np.int64]
    test_indices_class1: NDArray[np.int64]

    def save(self, output_dir: Union[str, Path]) -> None:
        """
        Save the trained pipeline, metrics, predictions, and split indices.

        Parameters
        ----------
        output_dir:
            Directory in which the outputs will be written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_dir / "classifier.joblib")

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(self.metrics, file, indent=2)

        with open(
            output_dir / "classification_report.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.classification_report, file, indent=2)

        np.save(output_dir / "confusion_matrix.npy", self.confusion_matrix)

        np.savez_compressed(
            output_dir / "predictions.npz",
            test_embeddings=self.test_embeddings,
            test_labels=self.test_labels,
            predictions=self.predictions,
            probabilities=self.probabilities,
        )

        np.savez_compressed(
            output_dir / "split_indices.npz",
            train_indices_class0=self.train_indices_class0,
            test_indices_class0=self.test_indices_class0,
            train_indices_class1=self.train_indices_class1,
            test_indices_class1=self.test_indices_class1,
        )


def _validate_embeddings(
    embeddings: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    """Convert an embedding array to a finite two-dimensional NumPy array."""
    array = np.asarray(embeddings, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must have shape (n_samples, embedding_dim), "
            f"but received shape {array.shape}."
        )

    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples.")

    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return array


def _split_one_class(
    embeddings: NDArray[np.float64],
    train_fraction: float,
    rng: np.random.Generator,
    shuffle: bool,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Split one class while guaranteeing nonempty train and test partitions."""
    n_samples = embeddings.shape[0]

    n_train = int(np.floor(train_fraction * n_samples))
    n_train = min(max(n_train, 1), n_samples - 1)

    indices = np.arange(n_samples)

    if shuffle:
        indices = rng.permutation(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    return (
        embeddings[train_indices],
        embeddings[test_indices],
        train_indices.astype(np.int64),
        test_indices.astype(np.int64),
    )


def split_two_embedding_sets(
    embeddings_class0: ArrayLike,
    embeddings_class1: ArrayLike,
    train_fraction: float = 0.8,
    random_state: int = 42,
    shuffle_within_class: bool = True,
    shuffle_combined_sets: bool = True,
) -> Dict[str, NDArray]:
    """
    Split two embedding sets independently and combine their partitions.

    Splitting each class independently ensures that both train and test sets
    preserve the requested class composition even when the class sizes differ.

    Returns
    -------
    dict
        Contains training/testing embeddings, labels, and class-local indices.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")

    class0 = _validate_embeddings(embeddings_class0, "embeddings_class0")
    class1 = _validate_embeddings(embeddings_class1, "embeddings_class1")

    if class0.shape[1] != class1.shape[1]:
        raise ValueError(
            "Both embedding sets must have the same embedding dimension. "
            f"Received {class0.shape[1]} and {class1.shape[1]}."
        )

    rng = np.random.default_rng(random_state)

    x0_train, x0_test, idx0_train, idx0_test = _split_one_class(
        class0,
        train_fraction,
        rng,
        shuffle_within_class,
    )
    x1_train, x1_test, idx1_train, idx1_test = _split_one_class(
        class1,
        train_fraction,
        rng,
        shuffle_within_class,
    )

    x_train = np.concatenate([x0_train, x1_train], axis=0)
    y_train = np.concatenate(
        [
            np.zeros(len(x0_train), dtype=np.int64),
            np.ones(len(x1_train), dtype=np.int64),
        ]
    )

    x_test = np.concatenate([x0_test, x1_test], axis=0)
    y_test = np.concatenate(
        [
            np.zeros(len(x0_test), dtype=np.int64),
            np.ones(len(x1_test), dtype=np.int64),
        ]
    )

    if shuffle_combined_sets:
        train_order = rng.permutation(len(x_train))
        test_order = rng.permutation(len(x_test))

        x_train = x_train[train_order]
        y_train = y_train[train_order]
        x_test = x_test[test_order]
        y_test = y_test[test_order]

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_indices_class0": idx0_train,
        "test_indices_class0": idx0_test,
        "train_indices_class1": idx1_train,
        "test_indices_class1": idx1_test,
    }


def build_classifier(
    classifier: ClassifierName = "knn",
    scale_embeddings: bool = True,
    random_state: int = 42,
    classifier_kwargs: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Construct a scikit-learn classification pipeline.

    Parameters
    ----------
    classifier:
        Name of the downstream classifier.

    scale_embeddings:
        Apply StandardScaler before fitting. This is strongly recommended for
        kNN, logistic regression, and MLP.

    random_state:
        Seed used by stochastic classifiers.

    classifier_kwargs:
        Keyword arguments overriding the classifier defaults.
    """
    kwargs = dict(classifier_kwargs or {})

    if classifier == "knn":
        defaults: Dict[str, Any] = {
            "n_neighbors": 15,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
        }
        defaults.update(kwargs)
        estimator: BaseEstimator = KNeighborsClassifier(**defaults)

    elif classifier == "logistic_regression":
        defaults = {
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        defaults.update(kwargs)
        estimator = LogisticRegression(**defaults)

    elif classifier == "gradient_boosting":
        defaults = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": random_state,
        }
        defaults.update(kwargs)
        estimator = GradientBoostingClassifier(**defaults)

    elif classifier == "random_forest":
        defaults = {
            "n_estimators": 300,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": random_state,
        }
        defaults.update(kwargs)
        estimator = RandomForestClassifier(**defaults)

    elif classifier == "mlp":
        defaults = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "early_stopping": True,
            "max_iter": 500,
            "random_state": random_state,
        }
        defaults.update(kwargs)
        estimator = MLPClassifier(**defaults)

    else:
        raise ValueError(
            f"Unknown classifier '{classifier}'. Supported values are: "
            "'knn', 'logistic_regression', 'gradient_boosting', "
            "'random_forest', and 'mlp'."
        )

    steps = []
    if scale_embeddings:
        steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", estimator))
    return Pipeline(steps)


def evaluate_two_embedding_sets(
    embeddings_class0: ArrayLike,
    embeddings_class1: ArrayLike,
    classifier: ClassifierName = "knn",
    train_fraction: float = 0.8,
    random_state: int = 42,
    scale_embeddings: bool = True,
    shuffle_within_class: bool = True,
    shuffle_combined_sets: bool = True,
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    class_names: Tuple[str, str] = ("class_0", "class_1"),
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> DownstreamResult:
    """
    Train and evaluate a classifier on two sets of embedding vectors.

    Parameters
    ----------
    embeddings_class0, embeddings_class1:
        Arrays with shapes (N0, D) and (N1, D).

    classifier:
        One of: "knn", "logistic_regression", "gradient_boosting",
        "random_forest", or "mlp".

    train_fraction:
        Fraction of each class assigned to the training set.

    random_state:
        Random seed controlling data splitting and stochastic estimators.

    scale_embeddings:
        Whether to standardize embedding coordinates using training statistics.

    shuffle_within_class:
        Shuffle each class before taking the training fraction. Keep this True
        unless the original ordering has a deliberate meaning.

    shuffle_combined_sets:
        Shuffle the combined class-0/class-1 train and test arrays.

    classifier_kwargs:
        Optional keyword arguments passed to the chosen classifier.

    class_names:
        Human-readable names for class 0 and class 1.

    output_dir:
        Optional directory in which to save the model and results.

    verbose:
        Print a compact evaluation summary.

    Returns
    -------
    DownstreamResult
        Fitted model, metrics, predictions, and the generated split.
    """
    split = split_two_embedding_sets(
        embeddings_class0=embeddings_class0,
        embeddings_class1=embeddings_class1,
        train_fraction=train_fraction,
        random_state=random_state,
        shuffle_within_class=shuffle_within_class,
        shuffle_combined_sets=shuffle_combined_sets,
    )

    model = build_classifier(
        classifier=classifier,
        scale_embeddings=scale_embeddings,
        random_state=random_state,
        classifier_kwargs=classifier_kwargs,
    )

    model.fit(split["x_train"], split["y_train"])
    predictions = model.predict(split["x_test"]).astype(np.int64)

    probabilities: Optional[NDArray[np.float64]] = None
    positive_scores: Optional[NDArray[np.float64]] = None

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(split["x_test"]),
            dtype=np.float64,
        )
        positive_scores = probabilities[:, 1]

    elif hasattr(model, "decision_function"):
        positive_scores = np.asarray(
            model.decision_function(split["x_test"]),
            dtype=np.float64,
        )

    metrics = {
        "accuracy": float(
            accuracy_score(split["y_test"], predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(split["y_test"], predictions)
        ),
        "precision": float(
            precision_score(
                split["y_test"],
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                split["y_test"],
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                split["y_test"],
                predictions,
                zero_division=0,
            )
        ),
    }

    if positive_scores is not None and len(np.unique(split["y_test"])) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(split["y_test"], positive_scores)
        )

    matrix = confusion_matrix(
        split["y_test"],
        predictions,
        labels=[0, 1],
    ).astype(np.int64)

    report = classification_report(
        split["y_test"],
        predictions,
        labels=[0, 1],
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    result = DownstreamResult(
        model=model,
        metrics=metrics,
        confusion_matrix=matrix,
        classification_report=report,
        train_embeddings=split["x_train"],
        test_embeddings=split["x_test"],
        train_labels=split["y_train"],
        test_labels=split["y_test"],
        predictions=predictions,
        probabilities=probabilities,
        train_indices_class0=split["train_indices_class0"],
        test_indices_class0=split["test_indices_class0"],
        train_indices_class1=split["train_indices_class1"],
        test_indices_class1=split["test_indices_class1"],
    )

    if output_dir is not None:
        result.save(output_dir)

    if verbose:
        print(f"Classifier:        {classifier}")
        print(f"Training samples:  {len(split['y_train'])}")
        print(f"Testing samples:   {len(split['y_test'])}")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"F1 score:          {metrics['f1']:.4f}")
        if "roc_auc" in metrics:
            print(f"ROC AUC:           {metrics['roc_auc']:.4f}")
        print("Confusion matrix:")
        print(matrix)

    return result


if __name__ == "__main__":
    # Minimal self-test using two synthetic embedding clouds.
    rng = np.random.default_rng(42)

    embeddings_0 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(400, 128),
    )
    embeddings_1 = rng.normal(
        loc=0.2,
        scale=1.0,
        size=(400, 128),
    )

    result = evaluate_two_embedding_sets(
        embeddings_class0=embeddings_0,
        embeddings_class1=embeddings_1,
        classifier="knn",
        train_fraction=0.8,
        random_state=42,
        classifier_kwargs={"n_neighbors": 15},
        class_names=("ttbar", "ggf"),
        output_dir=None,
    )