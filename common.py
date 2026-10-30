"""Shared helpers: dataset loading, train/test split and plotting utilities.

Dataset : Olivetti Faces (400 images, 40 people, 10 images each, 64x64 grayscale)
          Downloaded automatically by scikit-learn the first time you run it
          (needs internet once; afterwards it is cached in ~/scikit_learn_data).
"""
import os
import numpy as np
import matplotlib

# Save figures to disk by default. Set environment variable SHOW=1 to also pop up windows.
if os.environ.get("SHOW", "0") != "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 100, "axes.grid": False, "font.size": 10})


def _synthetic_faces(n_people=40, per_person=10, size=64, seed=0):
    """Stand-in dataset with the same shape as Olivetti (used only for offline testing)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for p in range(n_people):
        proto = gaussian_filter(rng.random((size, size)), 6)
        proto = (proto - proto.min()) / (proto.max() - proto.min())
        for _ in range(per_person):
            var = gaussian_filter(rng.random((size, size)), 4)
            var = (var - var.mean()) / var.std()
            imgs.append(np.clip(proto + 0.08 * var, 0, 1))
            labels.append(p)
    imgs, labels = np.array(imgs, dtype=np.float32), np.array(labels)
    perm = rng.permutation(len(labels))
    return imgs.reshape(len(imgs), -1)[perm], labels[perm], imgs[perm]


def load_faces():
    """Returns X (n, h*w) in [0,1], y (n,), images (n, h, w)."""
    if os.environ.get("FACES_SYNTHETIC") == "1":
        print("[!] Using SYNTHETIC stand-in data (FACES_SYNTHETIC=1)")
        return _synthetic_faces()
    from sklearn.datasets import fetch_olivetti_faces
    data = fetch_olivetti_faces(shuffle=True, random_state=RANDOM_STATE)
    return data.data, data.target, data.images


def split_indices(y, test_size=0.25):
    """Stratified train/test split (same split is reused by all three solutions)."""
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, stratify=y,
                                      random_state=RANDOM_STATE)
    return idx_tr, idx_te


def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   saved -> outputs/{name}")
    if os.environ.get("SHOW", "0") == "1":
        plt.show()
    plt.close()
