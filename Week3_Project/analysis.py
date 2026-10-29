"""Week 3: Unsupervised Learning & Model Evaluation on the Wine dataset."""
import json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, adjusted_rand_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score, accuracy_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold,
                                     GridSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RS = 42
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
R = {}

# ---------------- 1. Data ----------------
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
names = list(wine.target_names)
R["n_samples"], R["n_features"] = X.shape
R["class_counts"] = {names[i]: int((y == i).sum()) for i in range(3)}
Xs = StandardScaler().fit_transform(X)

# ---------------- 2. PCA ----------------
pca_full = PCA().fit(Xs)
cum = np.cumsum(pca_full.explained_variance_ratio_)
R["pca_var_2"] = float(cum[1])
R["pca_comp_90"] = int(np.argmax(cum >= 0.90) + 1)
R["pca_comp_95"] = int(np.argmax(cum >= 0.95) + 1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, 14), pca_full.explained_variance_ratio_, alpha=.6, label="Individual")
ax.step(range(1, 14), cum, where="mid", color="crimson", label="Cumulative")
ax.axhline(0.9, ls="--", c="grey")
ax.set(xlabel="Principal component", ylabel="Explained variance ratio", title="PCA explained variance")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/01_pca_variance.png", dpi=150); plt.close(fig)
X2 = PCA(n_components=2, random_state=RS).fit_transform(Xs)

# ---------------- 3. K-Means ----------------
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(Xs)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xs, km.labels_))
R["kmeans_silhouette_by_k"] = {int(k): round(float(s), 4) for k, s in zip(ks, sil)}
best_k = int(list(ks)[int(np.argmax(sil))])
R["best_k_by_silhouette"] = best_k
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set(xlabel="k", ylabel="Inertia (WCSS)", title="Elbow method")
ax[1].plot(list(ks), sil, "o-", c="darkorange"); ax[1].set(xlabel="k", ylabel="Silhouette score", title="Silhouette analysis")
fig.tight_layout(); fig.savefig(f"{OUT}/02_kmeans_elbow_silhouette.png", dpi=150); plt.close(fig)

km = KMeans(n_clusters=3, n_init=10, random_state=RS).fit(Xs)
km_labels = km.labels_
R["kmeans_k3_silhouette"] = float(silhouette_score(Xs, km_labels))
R["kmeans_k3_ARI"] = float(adjusted_rand_score(y, km_labels))
R["kmeans_k3_inertia"] = float(km.inertia_)

# ---------------- 4. Hierarchical ----------------
Z = linkage(Xs, method="ward")
fig, ax = plt.subplots(figsize=(10, 4.5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, ax=ax,
           color_threshold=0.7 * max(Z[:, 2]))
ax.set(title="Hierarchical clustering dendrogram (Ward, truncated)", xlabel="Cluster / sample count", ylabel="Distance")
fig.tight_layout(); fig.savefig(f"{OUT}/03_dendrogram.png", dpi=150); plt.close(fig)
hc_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Xs)
R["hier_silhouette"] = float(silhouette_score(Xs, hc_labels))
R["hier_ARI"] = float(adjusted_rand_score(y, hc_labels))
R["linkage_comparison"] = {}
for l in ["ward", "complete", "average", "single"]:
    lab = AgglomerativeClustering(n_clusters=3, linkage=l).fit_predict(Xs)
    R["linkage_comparison"][l] = {"silhouette": round(float(silhouette_score(Xs, lab)), 4),
                                  "ARI": round(float(adjusted_rand_score(y, lab)), 4)}

# ---------------- 5. Cluster plots + crosstabs ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, lab, t in zip(ax, [y, km_labels, hc_labels], ["True classes", "K-Means (k=3)", "Hierarchical (Ward, k=3)"]):
    sc = a.scatter(X2[:, 0], X2[:, 1], c=lab, cmap="viridis", s=35, edgecolor="k", linewidth=.3)
    a.set(title=t, xlabel="PC1", ylabel="PC2")
fig.tight_layout(); fig.savefig(f"{OUT}/04_cluster_pca_plots.png", dpi=150); plt.close(fig)
R["crosstab_kmeans"] = pd.crosstab(pd.Series(y, name="True").map(dict(enumerate(names))),
                                   pd.Series(km_labels, name="Cluster")).to_dict()
R["crosstab_hier"] = pd.crosstab(pd.Series(y, name="True").map(dict(enumerate(names))),
                                 pd.Series(hc_labels, name="Cluster")).to_dict()

# ---------------- 6. Classification & evaluation ----------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RS)
from sklearn.pipeline import Pipeline
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RS),
}
R["cv_baseline"] = {}
for n, m in models.items():
    s = cross_val_score(m, X_tr, y_tr, cv=skf, scoring="accuracy")
    R["cv_baseline"][n] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
                           "folds": [round(float(v), 4) for v in s]}

# Hyperparameter tuning
grid_rf = GridSearchCV(RandomForestClassifier(random_state=RS),
                       {"n_estimators": [50, 100, 200], "max_depth": [None, 3, 5, 10],
                        "min_samples_split": [2, 5]}, cv=skf, scoring="f1_macro", n_jobs=-1).fit(X_tr, y_tr)
grid_lr = GridSearchCV(Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
                       {"clf__C": [0.01, 0.1, 1, 10, 100]},
                       cv=skf, scoring="f1_macro", n_jobs=-1).fit(X_tr, y_tr)
R["grid_rf"] = {"best_params": grid_rf.best_params_, "best_cv_f1_macro": round(float(grid_rf.best_score_), 4)}
R["grid_lr"] = {"best_params": {k: (v if isinstance(v, str) else float(v)) for k, v in grid_lr.best_params_.items()},
                "best_cv_f1_macro": round(float(grid_lr.best_score_), 4)}

final = {"Logistic Regression (tuned)": grid_lr.best_estimator_, "Random Forest (tuned)": grid_rf.best_estimator_}
R["test_metrics"] = {}
yb = label_binarize(y_te, classes=[0, 1, 2])
probas = {}
for n, m in final.items():
    pred = m.predict(X_te); pr = m.predict_proba(X_te); probas[n] = pr
    R["test_metrics"][n] = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision_macro": round(float(precision_score(y_te, pred, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_te, pred, average="macro")), 4),
        "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
        "roc_auc_ovr": round(float(roc_auc_score(y_te, pr, multi_class="ovr")), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "report": classification_report(y_te, pred, target_names=names, output_dict=True),
    }
best_name = max(R["test_metrics"], key=lambda k: (R["test_metrics"][k]["f1_macro"], R["test_metrics"][k]["roc_auc_ovr"]))
R["best_model"] = best_name

# Confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (n, m) in zip(ax, final.items()):
    ConfusionMatrixDisplay.from_estimator(m, X_te, y_te, display_labels=names, cmap="Blues", ax=a, colorbar=False)
    a.set_title(n)
fig.tight_layout(); fig.savefig(f"{OUT}/05_confusion_matrices.png", dpi=150); plt.close(fig)

# ROC curves (best model, one-vs-rest)
fig, ax = plt.subplots(figsize=(6, 5))
pr = probas[best_name]
for i in range(3):
    fpr, tpr, _ = roc_curve(yb[:, i], pr[:, i])
    ax.plot(fpr, tpr, label=f"{names[i]} (AUC={roc_auc_score(yb[:, i], pr[:, i]):.3f})")
ax.plot([0, 1], [0, 1], "k--"); ax.set(xlabel="False positive rate", ylabel="True positive rate",
                                       title=f"ROC curves (OvR) - {best_name}")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(f"{OUT}/06_roc_curves.png", dpi=150); plt.close(fig)

# Feature importance
rf = grid_rf.best_estimator_
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 5))
imp.tail(10).plot.barh(ax=ax, color="teal"); ax.set(title="Top 10 feature importances (Random Forest)")
fig.tight_layout(); fig.savefig(f"{OUT}/07_feature_importance.png", dpi=150); plt.close(fig)
R["top_features"] = [str(i) for i in imp.tail(5).index[::-1]]

# Final 5-fold CV on full data with tuned models (scoring variety)
R["final_cv_full_data"] = {}
for n, m in final.items():
    R["final_cv_full_data"][n] = {sc: round(float(cross_val_score(m, X, y, cv=skf, scoring=sc).mean()), 4)
                                  for sc in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]}

json.dump(R, open("results.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
