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