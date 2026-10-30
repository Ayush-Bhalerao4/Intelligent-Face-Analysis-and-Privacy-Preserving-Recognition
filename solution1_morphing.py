"""
SOLUTION 1 : Face Morphing / Synthesis
Use PCA to blend two people's faces, generate faces that never existed,
walk the face manifold.  Very visual, impressive graphs.
Lab exercises covered: 1 (data exploration/visualisation), 4 (PCA)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common import load_faces, save_fig, RANDOM_STATE


def main():
    print("=== SOLUTION 1: Face Morphing / Synthesis ===")
    X, y, images = load_faces()
    H, W = images.shape[1:]
    n_components = 80
    pca = PCA(n_components=n_components, svd_solver="full", random_state=RANDOM_STATE).fit(X)
    Z = pca.transform(X)
    print(f"   data: {X.shape[0]} images of {H}x{W} | PCA kept {n_components} components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    # ---------------------------------------------------------------- helper
    def to_img(x):
        """Clip to [0,1] and reshape to (H, W)."""
        return np.clip(x, 0, 1).reshape(H, W)

    # ---------------------------------------------------------------- GRAPH 1: morph between two faces
    idxA = np.where(y == 0)[0][0]
    idxB = np.where(y == 5)[0][0]
    zA, zB = Z[idxA], Z[idxB]
    steps = 11
    alphas = np.linspace(0, 1, steps)
    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2.5))
    for i, a in enumerate(alphas):
        blend = (1 - a) * zA + a * zB
        face = pca.inverse_transform(blend)
        axes[i].imshow(to_img(face), cmap="gray")
        axes[i].set_title(f"α={a:.1f}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle("Face morphing: linear interpolation in PCA space", fontsize=14, y=1.02)
    save_fig("s1_01_face_morph.png")

    # ---------------------------------------------------------------- GRAPH 2: random novel faces
    rng = np.random.default_rng(RANDOM_STATE)
    n_gen = 16
    stds = np.sqrt(pca.explained_variance_)
    z_random = rng.normal(0, stds, size=(n_gen, n_components))
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, z in zip(axes.ravel(), z_random):
        face = pca.inverse_transform(z)
        ax.imshow(to_img(face), cmap="gray")
        ax.axis("off")
    fig.suptitle("Randomly generated faces (sampled from PCA latent space)", fontsize=14)
    save_fig("s1_02_random_faces.png")

    # ---------------------------------------------------------------- GRAPH 3: manifold walk along top 2 PCs
    grid = 7
    pc1_range = np.linspace(-3 * stds[0], 3 * stds[0], grid)
    pc2_range = np.linspace(-3 * stds[1], 3 * stds[1], grid)
    canvas = np.zeros((grid * H, grid * W))
    for r, v2 in enumerate(pc2_range):
        for c, v1 in enumerate(pc1_range):
            z = np.zeros(n_components)
            z[0], z[1] = v1, v2
            face = pca.inverse_transform(z)
            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = to_img(face)
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap="gray")
    plt.xlabel("PC-1  →", fontsize=12); plt.ylabel("← PC-2", fontsize=12)
    plt.title("Face manifold: walking along the first two principal components", fontsize=14)
    plt.xticks([]); plt.yticks([])
    save_fig("s1_03_manifold_walk.png")

    # ---------------------------------------------------------------- GRAPH 4: multi-person blend (pixel space)
    people_ids = [0, 5, 10, 15, 20]
    mean_faces = []
    for pid in people_ids:
        mask = y == pid
        mean_faces.append(X[mask].mean(axis=0))

    n_pairs = len(people_ids)
    fig, axes = plt.subplots(n_pairs, n_pairs, figsize=(2 * n_pairs, 2 * n_pairs))
    for r in range(n_pairs):
        for c in range(n_pairs):
            blend = 0.5 * mean_faces[r] + 0.5 * mean_faces[c]
            axes[r, c].imshow(to_img(blend), cmap="gray")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"P{people_ids[c]}", fontsize=10)
            if c == 0:
                axes[r, c].set_ylabel(f"P{people_ids[r]}", fontsize=10)
    fig.suptitle("Pixel-space blending grid: 50/50 average of each pair", fontsize=14)
    save_fig("s1_04_blend_grid_pixel.png")

    # ---------------------------------------------------------------- GRAPH 5: PCA-space blend grid
    fig, axes = plt.subplots(n_pairs, n_pairs, figsize=(2 * n_pairs, 2 * n_pairs))
    z_means = [pca.transform(X[y == pid]).mean(axis=0) for pid in people_ids]
    for r in range(n_pairs):
        for c in range(n_pairs):
            z_blend = 0.5 * z_means[r] + 0.5 * z_means[c]
            face = pca.inverse_transform(z_blend)
            axes[r, c].imshow(to_img(face), cmap="gray")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"P{people_ids[c]}", fontsize=10)
            if c == 0:
                axes[r, c].set_ylabel(f"P{people_ids[r]}", fontsize=10)
    fig.suptitle("PCA-space blending grid: 50/50 mix in latent space", fontsize=14)
    save_fig("s1_05_blend_grid_pca.png")

    # ---------------------------------------------------------------- GRAPH 6: effect of each PC on mean face
    n_pcs = 8
    fig, axes = plt.subplots(n_pcs, 5, figsize=(10, 2 * n_pcs))
    offsets = [-3, -1.5, 0, 1.5, 3]
    z_mean = np.zeros(n_components)
    for r in range(n_pcs):
        for c, off in enumerate(offsets):
            z = z_mean.copy()
            z[r] = off * stds[r]
            face = pca.inverse_transform(z)
            axes[r, c].imshow(to_img(face), cmap="gray")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"{off:+.1f}σ", fontsize=10)
        axes[r, 0].set_ylabel(f"PC-{r+1}", fontsize=10)
    fig.suptitle("Effect of each principal component on the mean face", fontsize=14)
    save_fig("s1_06_pc_variations.png")

    # ---------------------------------------------------------------- GRAPH 7: reconstruction quality
    sample_ids = [0, 50, 100, 200]
    ks = [5, 10, 20, 40, 80]
    fig, axes = plt.subplots(len(sample_ids), len(ks) + 1, figsize=(2 * (len(ks) + 1), 2 * len(sample_ids)))
    for r, sid in enumerate(sample_ids):
        axes[r, 0].imshow(images[sid], cmap="gray")
        axes[r, 0].set_title("Original" if r == 0 else "", fontsize=9)
        axes[r, 0].axis("off")
        for c, k in enumerate(ks):
            z_trunc = Z[sid].copy()
            z_trunc[k:] = 0
            rec = pca.inverse_transform(z_trunc)
            axes[r, c + 1].imshow(to_img(rec), cmap="gray")
            axes[r, c + 1].set_title(f"{k} PCs" if r == 0 else "", fontsize=9)
            axes[r, c + 1].axis("off")
    fig.suptitle("Reconstruction quality vs number of principal components", fontsize=14)
    save_fig("s1_07_reconstruction.png")

    # ================================================================ NEW GRAPHS
    # ---------------------------------------------------------------- GRAPH 8: scree plot + cumulative variance
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    comps = np.arange(1, n_components + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.bar(comps, evr * 100, color="steelblue")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title("Scree plot: variance per component")
    ax2.plot(comps, cum * 100, marker="o", ms=3, color="darkorange")
    for thr, col in [(0.90, "green"), (0.95, "red")]:
        if cum[-1] >= thr:
            k = int(np.searchsorted(cum, thr)) + 1
            ax2.axhline(thr * 100, color=col, ls="--", lw=1)
            ax2.axvline(k, color=col, ls="--", lw=1)
            ax2.annotate(f"{int(thr*100)}% at {k} PCs", (k, thr * 100),
                         textcoords="offset points", xytext=(8, -14), color=col)
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_title("Cumulative explained variance")
    ax2.grid(alpha=0.3)
    fig.suptitle("How much information does each principal component carry?", fontsize=14)
    fig.tight_layout()
    save_fig("s1_08_scree_cumulative.png")

    # ---------------------------------------------------------------- GRAPH 9: 2-D face-space scatter
    plt.figure(figsize=(9, 7.5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab20", s=18, alpha=0.7)
    for pid in people_ids:
        cx, cy = Z[y == pid, :2].mean(axis=0)
        plt.scatter(cx, cy, c="black", marker="X", s=120, edgecolors="white", zorder=3)
        plt.annotate(f"P{pid}", (cx, cy), textcoords="offset points", xytext=(6, 6),
                     fontsize=11, fontweight="bold")
    plt.colorbar(sc, label="Person ID")
    plt.xlabel("PC-1"); plt.ylabel("PC-2")
    plt.title("Face space: every image projected onto the first two PCs\n(X = centroid of selected people)",
              fontsize=13)
    plt.grid(alpha=0.3)
    save_fig("s1_09_face_space_scatter.png")

    # ---------------------------------------------------------------- GRAPH 10: mean face + eigenfaces
    n_eig = 15
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
    axes = axes.ravel()
    axes[0].imshow(pca.mean_.reshape(H, W), cmap="gray")
    axes[0].set_title("Mean face", fontsize=10)
    for i in range(n_eig):
        axes[i + 1].imshow(pca.components_[i].reshape(H, W), cmap="gray")
        axes[i + 1].set_title(f"Eigenface {i+1}\n({evr[i]*100:.1f}%)", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Mean face and the top eigenfaces (principal directions of face variation)", fontsize=14)
    save_fig("s1_10_eigenfaces.png")

    print("=== Solution 1 finished ===\n")


if __name__ == "__main__":
    main()