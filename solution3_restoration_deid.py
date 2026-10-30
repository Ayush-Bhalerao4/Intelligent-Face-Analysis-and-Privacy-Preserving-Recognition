"""
SOLUTION 3 : Image restoration (de-noising) and face de-identification
Part A: Gaussian and salt-and-pepper noise removed with Gaussian filter, median filter,
        non-local means and PCA reconstruction (PSNR / SSIM evaluation).
Part B: pixelation, blur and k-same averaging; privacy-utility trade-off measured with a face recogniser trained on clean faces.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial.distance import cdist
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.restoration import denoise_nl_means
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from common import load_faces, split_indices, save_fig, OUT_DIR, RANDOM_STATE

warnings.filterwarnings("ignore")
rng = np.random.default_rng(RANDOM_STATE)

# ============================================================== noise models
def add_gaussian(imgs, sigma):
    return np.clip(imgs + rng.normal(0, sigma, imgs.shape), 0, 1)


def add_salt_pepper(imgs, amount):
    out = imgs.copy()
    r = rng.random(imgs.shape)
    out[r < amount / 2] = 0.0
    out[r > 1 - amount / 2] = 1.0
    return out


# ============================================================== de-noising methods
def f_gaussian(imgs):
    return gaussian_filter(imgs, sigma=(0, 1.0, 1.0))


def f_median(imgs):
    return median_filter(imgs, size=(1, 3, 3))


def estimate_noise_sigma(im):
    """Robust noise estimate from neighbouring-pixel differences (median absolute deviation)."""
    d = np.diff(im, axis=1).ravel()
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def f_nlm(imgs):
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        s = max(float(estimate_noise_sigma(im)), 1e-3)
        out[i] = denoise_nl_means(im, h=0.8 * s, sigma=s, patch_size=5, patch_distance=6,
                                  fast_mode=True)
    return out


def make_pca_denoiser(train_imgs, n_comp=60):
    n, h, w = train_imgs.shape
    pca = PCA(n_components=min(n_comp, n - 1), svd_solver="full").fit(train_imgs.reshape(n, -1))
    def f_pca(imgs):
        flat = imgs.reshape(len(imgs), -1)
        return np.clip(pca.inverse_transform(pca.transform(flat)), 0, 1).reshape(imgs.shape)
    return f_pca


def metrics(clean, other):
    p = np.mean([psnr(c, o, data_range=1.0) for c, o in zip(clean, other)])
    s = np.mean([ssim(c, o, data_range=1.0) for c, o in zip(clean, other)])
    return p, s


# ============================================================== de-identification methods
def pixelate(imgs, b):
    if b <= 1:
        return imgs.copy()
    n, h, w = imgs.shape
    small = imgs.reshape(n, h // b, b, w // b, b).mean(axis=(2, 4))
    return np.repeat(np.repeat(small, b, axis=1), b, axis=2)


def blur(imgs, sigma):
    if sigma <= 0:
        return imgs.copy()
    return gaussian_filter(imgs, sigma=(0, sigma, sigma))


def k_same(imgs, labels, feats, k):
    """k-Same: faces are grouped into groups of k DIFFERENT people (nearest neighbours in feature
    space) and every face in a group is replaced by the group's average face."""
    n = len(imgs)
    if k <= 1:
        return imgs.copy()
    D = cdist(feats, feats)
    unassigned, groups = list(range(n)), []
    while len(unassigned) >= k:
        seed = unassigned[0]
        cand = sorted(unassigned[1:], key=lambda j: D[seed, j])
        group, used = [seed], {labels[seed]}
        for j in cand:                                    # prefer different identities
            if labels[j] not in used:
                group.append(j); used.add(labels[j])
                if len(group) == k:
                    break
        for j in cand:                                    # fall back if not enough identities
            if len(group) == k:
                break
            if j not in group:
                group.append(j)
        for j in group:
            unassigned.remove(j)
        groups.append(group)
    for j in unassigned:                                  # leftovers join the nearest group
        g = int(np.argmin([D[j, grp].mean() for grp in groups]))
        groups[g].append(j)
    out = np.empty_like(imgs)
    for grp in groups:
        out[np.array(grp)] = imgs[np.array(grp)].mean(axis=0)
    return out


# ============================================================== main
def main():
    print("=== SOLUTION 3: Restoration and de-identification ===")
    X, y, images = load_faces()
    H, W = images.shape[1:]
    idx_tr, idx_te = split_indices(y)
    tr_imgs, te_imgs = images[idx_tr], images[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    print(f"   train={len(tr_imgs)}, test={len(te_imgs)}")

    # ------------------------------------------------------------ PART A: de-noising
    print("\n   --- Part A: de-noising ---")
    methods = {"Gaussian filter": f_gaussian, "Median filter": f_median,
               "Non-local means": f_nlm, "PCA reconstruction": make_pca_denoiser(tr_imgs)}
    noise_cfg = {
        "Gaussian noise": (add_gaussian, [0.05, 0.10, 0.15, 0.20, 0.30], "sigma"),
        "Salt & pepper noise": (add_salt_pepper, [0.02, 0.05, 0.10, 0.15, 0.20], "amount"),
    }
    rows = []
    for nname, (fn, levels, _) in noise_cfg.items():
        for lv in levels:
            noisy = fn(te_imgs, lv)
            p, s = metrics(te_imgs, noisy)
            rows.append((nname, lv, "Noisy (no filter)", p, s))
            for mname, m in methods.items():
                p, s = metrics(te_imgs, m(noisy))
                rows.append((nname, lv, mname, p, s))
            print(f"   {nname} level {lv}: done")
    df = pd.DataFrame(rows, columns=["noise", "level", "method", "PSNR", "SSIM"])
    df.round(4).to_csv(os.path.join(OUT_DIR, "s3_denoising_metrics.csv"), index=False)

    # GRAPH 1+2: before / noisy / restored grids
    for nname, (fn, levels, par) in noise_cfg.items():
        lv = levels[2]
        noisy = fn(te_imgs[:3], lv)
        restored = {m: f(noisy) for m, f in methods.items()}
        cols = ["Original", f"Noisy ({par}={lv})"] + list(methods)
        fig, axes = plt.subplots(3, len(cols), figsize=(2.2 * len(cols), 7))
        for r in range(3):
            ims = [te_imgs[r], noisy[r]] + [restored[m][r] for m in methods]
            for c, im in enumerate(ims):
                axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=1); axes[r, c].axis("off")
                t = cols[c] if r == 0 else ""
                if c >= 1:
                    t += f"\nPSNR {psnr(te_imgs[r], im, data_range=1.0):.1f} dB"
                axes[r, c].set_title(t, fontsize=8)
        fig.suptitle(f"Restoration of {nname.lower()}")
        save_fig(f"s3_0{1 if 'Gaussian' in nname else 2}_restoration_grid_"
                 f"{'gaussian' if 'Gaussian' in nname else 'saltpepper'}.png")

    # GRAPH 3: PSNR / SSIM vs noise level
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for j, (nname, (_, levels, par)) in enumerate(noise_cfg.items()):
        for i, metric in enumerate(["PSNR", "SSIM"]):
            ax = axes[i, j]
            for mname in ["Noisy (no filter)"] + list(methods):
                sub = df[(df.noise == nname) & (df.method == mname)]
                ax.plot(sub.level, sub[metric], marker="o", ls="--" if "Noisy" in mname else "-",
                        label=mname)
            ax.set_title(f"{metric} vs noise level - {nname}")
            ax.set_xlabel(f"Noise level ({par})"); ax.set_ylabel(metric + (" (dB)" if metric == "PSNR" else ""))
            ax.grid(alpha=.3); ax.legend(fontsize=8)
    save_fig("s3_03_psnr_ssim_vs_noise.png")

    # ------------------------------------------------------------ PART B: de-identification
    print("\n   --- Part B: de-identification ---")
    Xtr_flat, Xte_flat = tr_imgs.reshape(len(tr_imgs), -1), te_imgs.reshape(len(te_imgs), -1)
    recognisers = {
        "PCA + k-NN": make_pipeline(PCA(n_components=min(100, len(Xtr_flat) - 1)),
                                    KNeighborsClassifier(1)).fit(Xtr_flat, ytr),
        "PCA + SVM": make_pipeline(PCA(n_components=min(100, len(Xtr_flat) - 1)),
                                   SVC(kernel="linear")).fit(Xtr_flat, ytr),
    }
    feats_te = PCA(n_components=min(50, len(Xte_flat) - 1)).fit_transform(Xte_flat)

    def evaluate_deid(deid_imgs):
        flat = deid_imgs.reshape(len(deid_imgs), -1)
        accs = {n: float((m.predict(flat) == yte).mean() * 100) for n, m in recognisers.items()}
        s = float(np.mean([ssim(a, b, data_range=1.0) for a, b in zip(te_imgs, deid_imgs)]))
        return accs, s

    deid_cfg = {
        "Pixelation": ("block size (px)", [b for b in [1, 2, 4, 8, 16, 32] if H % b == 0],
                       lambda p: pixelate(te_imgs, p)),
        "Gaussian blur": ("sigma", [0, 1, 2, 3, 4, 6, 8], lambda p: blur(te_imgs, p)),
        "k-Same averaging": ("k", [1, 2, 3, 5, 7, 10],
                             lambda p: k_same(te_imgs, yte, feats_te, p)),
    }
    deid_rows = []
    for dname, (pname, params, fn) in deid_cfg.items():
        for p in params:
            accs, s = evaluate_deid(fn(p))
            deid_rows.append({"method": dname, "param": p, **accs, "SSIM": s})
    ddf = pd.DataFrame(deid_rows)
    ddf.round(3).to_csv(os.path.join(OUT_DIR, "s3_deidentification_metrics.csv"), index=False)
    print(ddf.round(2).to_string(index=False))

    # GRAPH 4: privacy-utility curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (dname, (pname, params, _)) in zip(axes, deid_cfg.items()):
        sub = ddf[ddf.method == dname]
        x = np.arange(len(sub))
        for rname in recognisers:
            ax.plot(x, sub[rname], marker="o", label=f"Recognition: {rname}")
        ax.set_ylabel("Recognition accuracy (%)  [privacy risk]"); ax.set_ylim(-3, 105)
        ax.set_xticks(x); ax.set_xticklabels(sub.param.astype(str)); ax.set_xlabel(pname)
        ax2 = ax.twinx()
        ax2.plot(x, sub.SSIM, "k--", marker="s", label="SSIM to original [utility]")
        ax2.set_ylabel("SSIM (image utility)"); ax2.set_ylim(0, 1.02)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
        ax.set_title(dname); ax.grid(alpha=.3)
    fig.suptitle("Privacy-utility trade-off: stronger de-identification lowers recognition (and image quality)")
    save_fig("s3_04_privacy_utility_curves.png")

    # GRAPH 5: accuracy vs SSIM scatter (all settings)
    plt.figure(figsize=(7.5, 5.5))
    for dname, mk in zip(deid_cfg, ["o", "s", "^"]):
        sub = ddf[ddf.method == dname]
        plt.plot(sub.SSIM, sub["PCA + SVM"], marker=mk, label=dname)
        for _, r in sub.iterrows():
            plt.annotate(str(r.param), (r.SSIM, r["PCA + SVM"]), fontsize=7, xytext=(3, 3),
                         textcoords="offset points")
    plt.xlabel("Image utility (SSIM with original)"); plt.ylabel("Recognition accuracy (%)  [lower = more private]")
    plt.title("Privacy vs utility (labels = method parameter)"); plt.grid(alpha=.3); plt.legend()
    save_fig("s3_05_privacy_vs_utility_scatter.png")

    # GRAPH 6: before/after strips
    n_show = 8
    strips = [("Original", te_imgs), ("Pixelated (8 px)", pixelate(te_imgs, 8 if H % 8 == 0 else 4)),
              ("Blur (sigma=4)", blur(te_imgs, 4)), ("k-Same (k=5)", k_same(te_imgs, yte, feats_te, 5))]
    fig, axes = plt.subplots(len(strips), n_show, figsize=(1.7 * n_show, 1.8 * len(strips)))
    for r, (name, ims) in enumerate(strips):
        for c in range(n_show):
            axes[r, c].imshow(ims[c], cmap="gray", vmin=0, vmax=1); axes[r, c].axis("off")
        axes[r, 0].text(-0.1, 0.5, name, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.suptitle("Face de-identification: before and after")
    save_fig("s3_06_deidentification_strips.png")

    # GRAPH 7: final summary
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    order = ["Noisy (no filter)"] + list(methods)
    avg = df.groupby(["method", "noise"])[["PSNR", "SSIM"]].mean().reset_index()
    width = 0.38
    for ax, metric in zip(axes[:2], ["PSNR", "SSIM"]):
        for i, nname in enumerate(noise_cfg):
            vals = [avg[(avg.method == m) & (avg.noise == nname)][metric].values[0] for m in order]
            bars = ax.bar(np.arange(len(order)) + (i - .5) * width, vals, width, label=nname, edgecolor="k")
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order, rotation=20)
        ax.set_title(f"Average {metric} over all noise levels"); ax.legend(); ax.grid(axis="y", alpha=.3)
    base = ddf[(ddf.method == "Pixelation") & (ddf.param == 1)]["PCA + SVM"].values[0]
    labels_ = ["Original (no de-id)", "Pixelate 8", "Blur 4", "k-Same 5"]
    vals = [base,
            ddf[(ddf.method == "Pixelation") & (ddf.param == (8 if H % 8 == 0 else 4))]["PCA + SVM"].values[0],
            ddf[(ddf.method == "Gaussian blur") & (ddf.param == 4)]["PCA + SVM"].values[0],
            ddf[(ddf.method == "k-Same averaging") & (ddf.param == 5)]["PCA + SVM"].values[0]]
    bars = axes[2].bar(labels_, vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], edgecolor="k")
    axes[2].bar_label(bars, fmt="%.1f%%"); axes[2].set_ylim(0, 105); axes[2].tick_params(axis="x", rotation=20)
    axes[2].set_title("Face recognition accuracy after de-identification"); axes[2].grid(axis="y", alpha=.3)
    save_fig("s3_07_final_summary.png")

    print("=== Solution 3 finished ===\n")


if __name__ == "__main__":
    main()
