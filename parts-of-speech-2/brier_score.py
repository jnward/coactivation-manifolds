# %% imports and constants
import numpy as np
import joblib
import matplotlib.pyplot as plt

from utils import upos_to_tag_name

PROBE_PATH = "probe.joblib"
VAL_PATH = "activations_validation_12.npz"
CONF_THRESH = None  # set to None to use all samples
TEMP_SETTINGS = [0.5, 2.0]

# %% load probe and data
clf = joblib.load(PROBE_PATH)
val_data = np.load(VAL_PATH)
X_val = val_data["X"]
y_val = val_data["y"]
mask = np.isin(y_val, clf.classes_)
X_val = X_val[mask]
y_val = y_val[mask]
probs = clf.predict_proba(X_val)
max_conf = probs.max(axis=1)

if CONF_THRESH is not None:
    conf_mask = max_conf <= CONF_THRESH
    X_val = X_val[conf_mask]
    y_val = y_val[conf_mask]
    probs = probs[conf_mask]
    max_conf = max_conf[conf_mask]
    print(f"Filtered to {len(y_val)} samples with confidence <= {CONF_THRESH}")
else:
    print(f"Using all {len(y_val)} samples")

if len(y_val) == 0:
    raise ValueError("No samples left after applying filters for Brier computation")

num_classes = len(clf.classes_)
one_hot = np.zeros((len(y_val), num_classes), dtype=np.float32)
class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
for row, label in enumerate(y_val):
    one_hot[row, class_to_idx[label]] = 1.0

def brier_from_probs(pred_probs):
    return np.mean(np.sum((one_hot - pred_probs) ** 2, axis=1))

# Soft predictions
brier_soft = brier_from_probs(probs)

# Hard predictions
pred_labels = probs.argmax(axis=1)
pred_one_hot = np.zeros_like(probs)
pred_one_hot[np.arange(len(pred_labels)), pred_labels] = 1.0
brier_argmax = brier_from_probs(pred_one_hot)

# Temperature scaling
brier_temps = []
for temp in TEMP_SETTINGS:
    scaled = probs ** (1.0 / temp)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    brier_temps.append((temp, brier_from_probs(scaled)))

# Dataset prior baseline
mapped_labels = np.array([class_to_idx[label] for label in y_val])
class_counts = np.bincount(mapped_labels, minlength=num_classes)
prior_probs = class_counts / class_counts.sum()
prior_pred = np.broadcast_to(prior_probs, probs.shape)
brier_prior = brier_from_probs(prior_pred)

# Top-prob fixed, residual uniform across others
residual_probs = np.full_like(probs, fill_value=0.0)
top_indices = probs.argmax(axis=1)
top_vals = probs[np.arange(len(probs)), top_indices]
residual = 1.0 - top_vals
residual_share = residual / (num_classes - 1)
residual_probs += residual_share[:, None]
residual_probs[np.arange(len(probs)), top_indices] = top_vals
brier_residual_uniform = brier_from_probs(residual_probs)

print(f"Brier scores on filtered set (confidence <= {CONF_THRESH}):")
print(f"1. Soft predictions (probe):   {brier_soft:.4f}")
print(f"2. Hard predictions (argmax):  {brier_argmax:.4f}")
idx = 3
for temp, score in brier_temps:
    label = "sharp" if temp < 1 else "smooth"
    print(f"{idx}. Temperature={temp:.1f} ({label}): {score:.4f}")
    idx += 1
print(f"{idx}. Dataset prior baseline: {brier_prior:.4f}")
print(f"{idx+1}. Top fixed, residual uniform: {brier_residual_uniform:.4f}")

# %% per-class Brier scores (soft predictions)
class_names = [upos_to_tag_name(c) for c in clf.classes_]
per_class_scores = []
for label, name in zip(clf.classes_, class_names):
    mask = y_val == label
    if not np.any(mask):
        per_class_scores.append((name, np.nan))
        continue
    class_one_hot = one_hot[mask]
    class_probs = probs[mask]
    score = np.mean(np.sum((class_one_hot - class_probs) ** 2, axis=1))
    per_class_scores.append((name, score))

print("Per-class Brier scores:")
for name, score in per_class_scores:
    if np.isnan(score):
        print(f"  {name}: N/A (no samples)")
    else:
        print(f"  {name}: {score:.5f}")

# %% bar chart of per-class scores
valid_scores = [(name, score) for name, score in per_class_scores if not np.isnan(score)]
names, scores = zip(*valid_scores)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(names, scores, color="steelblue")
ax.set_ylabel("Brier score")
ax.set_title("Per-class Brier Scores")
ax.set_xticklabels(names, rotation=90)
fig.tight_layout()
plt.show()

# %%
