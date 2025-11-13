# %% imports and constants
import numpy as np
import joblib
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from utils import upos_to_tag_name

PROBE_PATH = "probe.joblib"
VAL_PATH = "activations_validation_12.npz"
UMAP_SEED = 0

# %% load probe
clf = joblib.load(PROBE_PATH)
weights = clf.coef_
class_ids = clf.classes_
class_names = [upos_to_tag_name(label_id) for label_id in class_ids]
print(f"Loaded probe with {len(class_names)} classes and {weights.shape[1]}-d directions")

# %% load validation activations
val_data = np.load(VAL_PATH)
X_val = val_data["X"]
y_val = val_data["y"]
mask = np.isin(y_val, class_ids)
X_val = X_val[mask]
y_val = y_val[mask]

# %% cosine similarity matrix
norms = np.linalg.norm(weights, axis=1, keepdims=True)
unit_weights = weights / norms
cos_sim = unit_weights @ unit_weights.T

num_classes = len(class_names)
pair_sims = []
for i in range(num_classes):
    for j in range(i + 1, num_classes):
        pair_sims.append((cos_sim[i, j], class_names[i], class_names[j]))
pair_sims.sort(reverse=True, key=lambda x: x[0])
print("Top cosine pairs:")
for sim, a, b in pair_sims[:8]:
    print(f"  {a} - {b}: {sim:.3f}")
print("Bottom cosine pairs:")
for sim, a, b in pair_sims[-8:]:
    print(f"  {a} - {b}: {sim:.3f}")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=90)
ax.set_yticklabels(class_names)
ax.set_title("Probe Direction Cosine Similarity")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

# %% PCA of probe directions
pca = PCA()
pca.fit(weights)
cum_var = np.cumsum(pca.explained_variance_ratio_)
print("PCA cumulative variance:")
for i, val in enumerate(cum_var, 1):
    print(f"  {i}: {val:.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
ax.set_xlabel("Components")
ax.set_ylabel("Cumulative explained variance")
ax.set_ylim(0, 1.01)
ax.set_title("PCA of Probe Directions")
fig.tight_layout()
plt.show()

UMAP_SAMPLES = 5000
logits = clf.decision_function(X_val)
if logits.shape[0] > UMAP_SAMPLES:
    rng = np.random.default_rng(UMAP_SEED)
    idx = rng.choice(logits.shape[0], UMAP_SAMPLES, replace=False)
    logits_subset = logits[idx]
    labels_subset = y_val[idx]
else:
    logits_subset = logits
    labels_subset = y_val

# %% 2D PCA of probe logits
pca_logits = PCA(n_components=2)
pca_logits_2d = pca_logits.fit_transform(logits_subset)

colors = plt.cm.get_cmap("tab20", len(class_names))
fig, ax = plt.subplots(figsize=(6, 5))
for color_idx, (label_id, name) in enumerate(zip(class_ids, class_names)):
    mask = labels_subset == label_id
    if not np.any(mask):
        continue
    ax.scatter(
        pca_logits_2d[mask, 0],
        pca_logits_2d[mask, 1],
        color=colors(color_idx),
        label=name,
        s=6,
        alpha=0.6,
    )
ax.set_title("2D PCA of Probe Logits")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=1)
fig.tight_layout()
plt.show()

# %% UMAP of probe logits
reducer = umap.UMAP(n_components=2, metric="cosine", random_state=UMAP_SEED)
embedding = reducer.fit_transform(logits_subset)

fig, ax = plt.subplots(figsize=(6, 5))
for color_idx, (label_id, name) in enumerate(zip(class_ids, class_names)):
    mask = labels_subset == label_id
    if not np.any(mask):
        continue
    ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        color=colors(color_idx),
        label=name,
        s=6,
        alpha=0.6,
    )
ax.set_title("UMAP of Probe Logits (Validation)")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=1)
fig.tight_layout()
plt.show()

# %% 3D UMAP of probe logits
label_name_map = {cid: name for cid, name in zip(class_ids, class_names)}
labels_subset_names = [label_name_map[label] for label in labels_subset]
reducer_3d = umap.UMAP(n_components=3, metric="cosine", random_state=UMAP_SEED)
embedding_3d = reducer_3d.fit_transform(logits_subset)

fig3d = px.scatter_3d(
    x=embedding_3d[:, 0],
    y=embedding_3d[:, 1],
    z=embedding_3d[:, 2],
    color=labels_subset_names,
    opacity=0.6,
    title="3D UMAP of Probe Logits (Validation)",
)
fig3d.update_traces(marker=dict(size=3))
fig3d.update_layout(margin=dict(l=0, r=0, t=40, b=0))
fig3d.show()

# %% 3D PCA of probe logits
pca_logits_3d = PCA(n_components=3).fit_transform(logits_subset)
color_palette = px.colors.qualitative.Bold
color_map = {
    name: color_palette[i % len(color_palette)]
    for i, name in enumerate(class_names)
}
labels_subset_names = np.array(labels_subset_names)
fig3d_pca = px.scatter_3d(
    x=pca_logits_3d[:, 0],
    y=pca_logits_3d[:, 1],
    z=pca_logits_3d[:, 2],
    color=labels_subset_names,
    opacity=0.1,
    color_discrete_map=color_map,
    title="3D PCA of Probe Logits",
)
fig3d_pca.update_traces(marker=dict(size=3))
centroids = []
for name in class_names:
    mask = labels_subset_names == name
    if np.any(mask):
        centroids.append((name, pca_logits_3d[mask].mean(axis=0)))
for name, coord in centroids:
    fig3d_pca.add_trace(
        go.Scatter3d(
            x=[coord[0]],
            y=[coord[1]],
            z=[coord[2]],
            mode="markers+text",
            marker=dict(size=6, color=color_map[name]),
            text=[name],
            textposition="top center",
            name=f"{name} centroid",
            opacity=1.0,
            showlegend=False,
        )
    )
fig3d_pca.update_layout(margin=dict(l=0, r=0, t=40, b=0))
fig3d_pca.show()

# %% 3D PCA of normalized probe logits
logits_norms = np.linalg.norm(logits_subset, axis=1, keepdims=True)
logits_unit = np.divide(
    logits_subset,
    logits_norms,
    out=np.zeros_like(logits_subset),
    where=logits_norms != 0,
)
pca_logits_unit_3d = PCA(n_components=3).fit_transform(logits_unit)
fig3d_pca_unit = px.scatter_3d(
    x=pca_logits_unit_3d[:, 0],
    y=pca_logits_unit_3d[:, 1],
    z=pca_logits_unit_3d[:, 2],
    color=labels_subset_names,
    opacity=0.1,
    color_discrete_map=color_map,
    title="3D PCA of Unit-Norm Probe Logits",
)
fig3d_pca_unit.update_traces(marker=dict(size=3))
centroids_unit = []
for name in class_names:
    mask = labels_subset_names == name
    if np.any(mask):
        centroids_unit.append((name, pca_logits_unit_3d[mask].mean(axis=0)))
for name, coord in centroids_unit:
    fig3d_pca_unit.add_trace(
        go.Scatter3d(
            x=[coord[0]],
            y=[coord[1]],
            z=[coord[2]],
            mode="markers+text",
            marker=dict(size=6, color=color_map[name]),
            text=[name],
            textposition="top center",
            name=f"{name} centroid (unit)",
            opacity=1.0,
            showlegend=False,
        )
    )
fig3d_pca_unit.update_layout(margin=dict(l=0, r=0, t=40, b=0))
fig3d_pca_unit.show()

# %% 3D PCA of normalized probe logits (selected classes)
FOCUS_CLASSES = {"PROPN", "NOUN", "ADJ", "ADV", "ADP", "SCONJ"}
focus_mask = np.array([label in FOCUS_CLASSES for label in labels_subset_names])
logits_focus = logits_unit[focus_mask]
labels_focus = labels_subset_names[focus_mask]

pca_focus = PCA(n_components=3)
focus_embedding = pca_focus.fit_transform(logits_focus)
print("Explained variance (focus classes):", pca_focus.explained_variance_ratio_)

fig3d_focus = px.scatter_3d(
    x=focus_embedding[:, 0],
    y=focus_embedding[:, 1],
    z=focus_embedding[:, 2],
    color=labels_focus,
    opacity=0.1,
    color_discrete_map=color_map,
    title="3D PCA (Unit Logits, Focus Classes)",
)
fig3d_focus.update_traces(marker=dict(size=3))
centroids_focus = []
for name in FOCUS_CLASSES:
    mask = labels_focus == name
    if np.any(mask):
        centroids_focus.append((name, focus_embedding[mask].mean(axis=0)))
for name, coord in centroids_focus:
    fig3d_focus.add_trace(
        go.Scatter3d(
            x=[coord[0]],
            y=[coord[1]],
            z=[coord[2]],
            mode="markers+text",
            marker=dict(size=6, color=color_map[name]),
            text=[name],
            textposition="top center",
            name=f"{name} centroid (focus)",
            opacity=1.0,
            showlegend=False,
        )
    )
fig3d_focus.update_layout(margin=dict(l=0, r=0, t=40, b=0))
fig3d_focus.show()

# %% confusion matrix
y_pred = clf.predict(X_val)
probs_val = clf.predict_proba(X_val)

cm = confusion_matrix(y_val, y_pred, labels=class_ids)
row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0,
)

confusions = []
for i, true_name in enumerate(class_names):
    row_total = row_sums[i, 0]
    for j, pred_name in enumerate(class_names):
        if i == j or cm[i, j] == 0:
            continue
        frac = cm[i, j] / row_total if row_total else 0.0
        confusions.append((cm[i, j], frac, true_name, pred_name))
confusions.sort(key=lambda x: (x[1], x[0]), reverse=True)
print("Top confusions:")
for count, frac, true_name, pred_name in confusions[:24]:
    print(f"  true {true_name} -> pred {pred_name}: {count} ({frac:.1%})")

# %% class entropy
entropies = []
for class_id, class_name in zip(class_ids, class_names):
    mask_class = y_val == class_id
    if not np.any(mask_class):
        continue
    class_probs = probs_val[mask_class]
    entropy = -np.sum(class_probs * np.log(class_probs + 1e-12), axis=1).mean()
    entropies.append((entropy, class_name))
entropies.sort(reverse=True)
print("Highest entropy classes:")
for entropy, class_name in entropies[:8]:
    print(f"  {class_name}: {entropy:.3f}")

# %% plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=90)
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Validation Confusion Matrix (Normalized)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

# %% prediction confidence histograms
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(low_conf, bins=20, color="steelblue", alpha=0.8)
ax.set_xlabel("Top-class probability")
ax.set_ylabel("Count")
ax.set_yscale("log")
ax.set_title("Prediction Confidence (<99.5%)")
fig.tight_layout()
plt.show()

n_classes = len(class_ids)
ncols = 3
nrows = (n_classes + ncols - 1) // ncols
# per-class histograms (low confidence only)
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), sharex=True, sharey=True)
axes = list(axes.flat)
mask_low_conf = max_conf < 0.995
for ax, class_id, class_name in zip(axes, class_ids, class_names):
    class_mask = (y_val == class_id) & mask_low_conf
    if not np.any(class_mask):
        ax.set_visible(False)
        continue
    class_conf = probs_val[class_mask].max(axis=1)
    ax.hist(class_conf, bins=20, color="steelblue", alpha=0.8)
    ax.set_title(class_name)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Top prob")
    ax.set_ylabel("Count")
for ax in axes[len(class_ids):]:
    ax.set_visible(False)
fig.suptitle("Prediction Confidence per Class", y=0.92)
fig.tight_layout()
plt.show()

# %% prediction confidence quantiles
max_conf = probs_val.max(axis=1)
low_conf = max_conf[max_conf < 0.995]
quantiles = np.quantile(low_conf, np.linspace(0, 1, 11))
bin_indices = np.digitize(low_conf, quantiles[1:-1], right=True)
counts = np.bincount(bin_indices, minlength=10)
print("Confidence decile ranges (top prob):")
for i in range(10):
    lo = quantiles[i]
    hi = quantiles[i + 1]
    print(f"  Bin {i+1}: [{lo:.3f}, {hi:.3f}) -> {counts[i]} examples")

# %% quantile visualization
fig, ax = plt.subplots(figsize=(8, 1.5))
ax.hlines(0, low_conf.min(), low_conf.max(), colors="gray", linestyles="dashed")
for i in range(11):
    x = quantiles[i]
    ax.vlines(x, -0.2, 0.2, colors="black", linewidth=1)
    if i in {0, 10}:
        ax.text(x, 0.25, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
for i in range(10):
    mid = 0.5 * (quantiles[i] + quantiles[i + 1])
    ax.text(mid, -0.25, f"{counts[i]}", ha="center", va="top", fontsize=8)
ax.set_ylim(-0.4, 0.4)
ax.set_yticks([])
ax.set_xlabel("Top-class probability (low-confidence deciles)")
ax.set_title("Confidence Quantiles (<99.5%)")
fig.tight_layout()
plt.show()

# %%
