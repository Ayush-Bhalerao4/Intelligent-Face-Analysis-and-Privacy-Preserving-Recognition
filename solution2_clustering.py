"""
SOLUTION 2 : Probabilistic modelling (GMM-EM) and cluster analysis
Lab exercises covered: 2 (probabilistic foundations), 4 (cluster analysis), 1 (visualisation)
Fully unsupervised: true labels are used ONLY to evaluate the clusters afterwards.
"""
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, silhouette_samples, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score)

from common import load_faces, save_fig, OUT_DIR, RANDOM_STATE

warnings.filterwarnings("ignore")
import os


def purity_score(y_true, labels):
    total = 0
    for c in np.unique(labels):
        total += np.bincount(y_true[labels == c]).max()
    return total / len(y_true)


def evaluate(labels, Z, y):
    """Internal (silhouette, Davies-Bouldin) + external (ARI, NMI, purity) validity indices.
    Noise points (-1, DBSCAN) are excluded from the internal indices."""
    mask = labels != -1
    n_cl = len(np.unique(labels[mask]))
    if n_cl >= 2 and mask.sum() > n_cl:
        sil = silhouette_score(Z[mask], labels[mask])
        db = davies_bouldin_score(Z[mask], labels[mask])
    else:
        sil, db = np.nan, np.nan
    return {"Clusters": n_cl, "Noise pts": int((~mask).sum()), "Silhouette": sil,
            "Davies-Bouldin": db, "ARI": adjusted_rand_score(y, labels),
            "NMI": normalized_mutual_info_score(y, labels), "Purity": purity_score(y, labels)}


def main():
    print("=== SOLUTION 2: Probabilistic modelling & clustering ===")
    X, y, images = load_faces()
    H, W = images.shape[1:]
    K = len(np.unique(y))

    Z = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(X)   # features for clustering
    Z20 = Z[:, :20]                                                        # lower-dim for GMM

    # ---------------------------------------------------------------- GRAPH 1: dendrogram
    link = linkage(Z, method="ward")
    plt.figure(figsize=(12, 5))
    dendrogram(link, truncate_mode="lastp", p=40, leaf_rotation=90, leaf_font_size=8,
               show_contracted=True, color_threshold=link[-(K - 1), 2])
    plt.axhline(link[-(K - 1), 2], color="r", ls="--", label=f"cut for {K} clusters")
    plt.title("Hierarchical clustering dendrogram (Ward linkage, truncated)")
    plt.xlabel("Cluster (number of faces in brackets)"); plt.ylabel("Distance"); plt.legend()
    save_fig("s2_01_dendrogram.png")

    # ---------------------------------------------------------------- GRAPH 2: elbow + silhouette vs k
    ks = list(range(2, 61, 2))
    inertia, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE).fit(Z)
        inertia.append(km.inertia_); sils.append(silhouette_score(Z, km.labels_))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ks, inertia, "o-"); axes[0].axvline(K, color="r", ls="--", label=f"true k={K}")
    axes[0].set_title("Elbow curve (K-means inertia)"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(ks, sils, "o-", color="g"); axes[1].axvline(K, color="r", ls="--", label=f"true k={K}")
    best_k = ks[int(np.argmax(sils))]
    axes[1].scatter([best_k], [max(sils)], s=120, c="gold", edgecolor="k", zorder=5,
                    label=f"best silhouette k={best_k}")
    axes[1].set_title("Average silhouette score vs k"); axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
    axes[1].legend(); axes[1].grid(alpha=.3)
    save_fig("s2_02_elbow_silhouette_vs_k.png")

    # ---------------------------------------------------------------- fit the four algorithms
    km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Z)
    agg = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Z)
    gmm = GaussianMixture(n_components=K, covariance_type="diag", reg_covar=1e-3, n_init=3,
                          random_state=RANDOM_STATE).fit(Z20)
    gmm_labels = gmm.predict(Z20)

    # DBSCAN: eps chosen from k-distance graph by best silhouette (unsupervised)
    min_samples = 4
    kd = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)[0][:, -1])
    best_eps, best_s, db_labels = None, -2, None
    for eps in np.percentile(kd, [40, 50, 60, 70, 80, 90]):
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        m = lab != -1
        if len(np.unique(lab[m])) >= 2:
            s = silhouette_score(Z[m], lab[m])
            if s > best_s:
                best_s, best_eps, db_labels = s, eps, lab
    if db_labels is None:
        best_eps = np.median(kd)
        db_labels = DBSCAN(eps=best_eps, min_samples=min_samples).fit_predict(Z)
    print(f"   DBSCAN eps={best_eps:.2f}, min_samples={min_samples}")

    all_labels = {"K-means": km.labels_, "Hierarchical (Ward)": agg.labels_,
                  "GMM (EM)": gmm_labels, "DBSCAN": db_labels}
    table = pd.DataFrame({n: evaluate(l, Z, y) for n, l in all_labels.items()}).T
    print("\n", table.round(3).to_string())
    table.round(4).to_csv(os.path.join(OUT_DIR, "s2_cluster_metrics.csv"))

    # ---------------------------------------------------------------- GRAPH 3: GMM BIC / AIC
    ncs = list(range(5, 61, 5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cov in ["spherical", "diag"]:
        bic, aic = [], []
        for n in ncs:
            g = GaussianMixture(n_components=n, covariance_type=cov, reg_covar=1e-3,
                                random_state=RANDOM_STATE).fit(Z20)
            bic.append(g.bic(Z20)); aic.append(g.aic(Z20))
        axes[0].plot(ncs, bic, "o-", label=cov); axes[1].plot(ncs, aic, "o-", label=cov)
    for ax, t in zip(axes, ["BIC", "AIC"]):
        ax.axvline(K, color="r", ls="--", label=f"true k={K}")
        ax.set_title(f"GMM {t} vs number of components (lower is better)")
        ax.set_xlabel("Components"); ax.set_ylabel(t); ax.legend(); ax.grid(alpha=.3)
    save_fig("s2_03_gmm_bic_aic.png")

    # ---------------------------------------------------------------- GRAPH 4: silhouette analysis (k = K)
    sil_vals = silhouette_samples(Z, km.labels_)
    plt.figure(figsize=(8, 8))
    y_low, cmap = 5, plt.cm.nipy_spectral
    for c in range(K):
        v = np.sort(sil_vals[km.labels_ == c])
        plt.fill_betweenx(np.arange(y_low, y_low + len(v)), 0, v, color=cmap(c / K), alpha=.8)
        y_low += len(v) + 5
    plt.axvline(sil_vals.mean(), color="r", ls="--", label=f"mean = {sil_vals.mean():.3f}")
    plt.xlabel("Silhouette coefficient"); plt.ylabel("Samples grouped by cluster")
    plt.title(f"Silhouette analysis for K-means (k={K})"); plt.legend(); plt.yticks([])
    save_fig("s2_04_silhouette_analysis.png")

    # ---------------------------------------------------------------- GRAPH 5: t-SNE (clusters vs identity)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=RANDOM_STATE).fit_transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(emb[:, 0], emb[:, 1], c=km.labels_, cmap="nipy_spectral", s=25)
    axes[0].set_title("t-SNE coloured by K-means cluster")
    axes[1].scatter(emb[:, 0], emb[:, 1], c=y, cmap="nipy_spectral", s=25)
    axes[1].set_title("t-SNE coloured by TRUE identity")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    save_fig("s2_05_tsne_clusters_vs_identity.png")

    # ---------------------------------------------------------------- GRAPH 6: cluster-vs-identity heatmap
    cont = np.zeros((K, K), dtype=int)
    for c, t in zip(km.labels_, y):
        cont[c, t] += 1
    order = np.argsort(np.argmax(cont, axis=1) * 1000 - cont.max(axis=1))
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cont[order], cmap="Blues", cbar_kws={"label": "faces"}, xticklabels=5, yticklabels=5)
    plt.xlabel("True person"); plt.ylabel("K-means cluster (re-ordered)")
    plt.title(f"Cluster vs identity (K-means purity = {table.loc['K-means', 'Purity']:.2f})")
    save_fig("s2_06_cluster_purity_heatmap.png")

    # ---------------------------------------------------------------- GRAPH 7: metric comparison bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (m, note) in zip(axes, [("Silhouette", "higher is better"),
                                    ("Davies-Bouldin", "lower is better"),
                                    ("ARI", "higher is better")]):
        vals = table[m].astype(float).fillna(0)
        bars = ax.bar(vals.index, vals.values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
                      edgecolor="k")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_title(f"{m} ({note})"); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.3)
    save_fig("s2_07_validity_indices_comparison.png")

    # ---------------------------------------------------------------- GRAPH 8: montage of cluster members
    sizes = np.bincount(km.labels_)
    top = np.argsort(sizes)[::-1][:6]
    n_cols = 8
    fig, axes = plt.subplots(len(top), n_cols, figsize=(1.6 * n_cols, 1.7 * len(top)))
    for r, c in enumerate(top):
        members = np.where(km.labels_ == c)[0][:n_cols]
        for j in range(n_cols):
            axes[r, j].axis("off")
            if j < len(members):
                axes[r, j].imshow(images[members[j]], cmap="gray")
                axes[r, j].set_title(f"id {y[members[j]]}", fontsize=7)
        axes[r, 0].text(-0.15, 0.5, f"Cluster {c}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("Members of the six largest K-means clusters (title = true person id)")
    save_fig("s2_08_cluster_members.png")

    print("=== Solution 2 finished ===\n")


if __name__ == "__main__":
    main()
