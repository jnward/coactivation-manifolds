from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

BASE_DIR = Path("spacy_retagged")
TRAIN_PATH = BASE_DIR / "spacy_activations_train_12.npz"
VAL_PATH = BASE_DIR / "spacy_activations_validation_12.npz"
MODEL_PATH = BASE_DIR / "spacy_probe.joblib"
EXCLUDED_TAGS = ["_", "SYM", "INTJ"]
C = 1.0
MAX_ITER = 500


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    tag_names = data["tag_names"].tolist()
    return X, y, tag_names


def filter_data(X, y, excluded_ids):
    if not excluded_ids:
        return X, y
    mask = ~np.isin(y, excluded_ids)
    return X[mask], y[mask]


def main():
    X_train, y_train, tag_names = load_dataset(TRAIN_PATH)
    X_val, y_val, tag_names_val = load_dataset(VAL_PATH)

    if tag_names != tag_names_val:
        raise ValueError("Train/val tag name lists do not match.")

    tag_to_id = {tag: idx for idx, tag in enumerate(tag_names)}
    excluded_ids = [tag_to_id[tag] for tag in EXCLUDED_TAGS if tag in tag_to_id]

    X_train, y_train = filter_data(X_train, y_train, excluded_ids)
    X_val, y_val = filter_data(X_val, y_val, excluded_ids)

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

    included_labels = [idx for idx in range(len(tag_names)) if idx not in excluded_ids]
    target_names = [tag_names[idx] for idx in included_labels]
    print(classification_report(y_val, y_pred, labels=included_labels, target_names=target_names, digits=4))

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "tag_names": tag_names}, MODEL_PATH)
    print(f"Saved probe to {MODEL_PATH}")


if __name__ == "__main__":
    main()
