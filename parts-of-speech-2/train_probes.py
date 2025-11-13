import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

from utils import tag_name_to_upos

# TRAIN_PATH = "activations_train.npz"
TRAIN_PATH = "activations_test_12.npz"
VAL_PATH = "activations_validation_12.npz"
MODEL_PATH = "probe_test.joblib"
EXCLUDED_LABELS = ["_", "SYM", "INTJ"]  # e.g., ["_", "NOUN", "X"]
# EXCLUDED_LABELS = []
C = 1.0
MAX_ITER = 500


def load_split(path: str, excluded_labels: list[int]):
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    if excluded_labels:
        keep_mask = ~np.isin(y, excluded_labels)
        X = X[keep_mask]
        y = y[keep_mask]
    if X.size == 0:
        raise ValueError(f"No data left after filtering for {path}")
    return X, y


def main():
    excluded_label_ids = [tag_name_to_upos(label) for label in EXCLUDED_LABELS]
    X_train, y_train = load_split(TRAIN_PATH, excluded_label_ids)
    X_val, y_val = load_split(VAL_PATH, excluded_label_ids)

    clf = LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = (y_pred == y_val).mean()
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, digits=4))
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved probe to {MODEL_PATH}")


if __name__ == "__main__":
    main()
