# Intelligent Face Analysis and Privacy-Preserving Recognition
Pattern Recognition lab project: three solutions on the **Olivetti Faces** dataset
(400 grayscale images, 40 people, 64x64).

## Setup
```bash
pip install -r requirements.txt
```
Python 3.9+ recommended. The dataset is downloaded automatically by scikit-learn on the
first run (internet needed once; it is then cached in `~/scikit_learn_data`).

## Run
```bash
python run_all.py                       
# or one at a time:
python solution1_face_morphing.py
python solution2_clustering.py
python solution3_restoration_deid.py
```
All graphs (PNG) and result tables (CSV) are written to the `outputs/` folder.
To also open the plots in windows, set `SHOW=1` (e.g. `SHOW=1 python solution1_face_morphing.py`;
on Windows PowerShell: `$env:SHOW=1; python solution1_face_morphing.py`).

## Files and lab-exercise mapping
| File | What it does | Lab exercises |
|---|---|---|
| `common.py` | dataset loading, stratified 75/25 split, figure saving | - |
| `solution1_face_morphing.py` | PCA face morphing, synthesis of novel faces, face-manifold walk | 1, 4 |
| `solution2_clustering.py` | K-means, hierarchical, GMM (EM), DBSCAN + validity indices | 1, 2, 4 |
| `solution3_restoration_deid.py` | De-noising (4 methods, 2 noise types) and de-identification (3 methods) | 7, 8 |
| `run_all.py` | runs everything | - |

## Graphs produced
**Solution 1** - linear face morph between two people in PCA space; randomly generated faces sampled
from the PCA latent space; face-manifold walk along PC-1 / PC-2; pixel-space vs PCA-space 50/50 blend
grids; effect of each principal component on the mean face; reconstruction quality vs number of
components; scree plot and cumulative explained variance; 2-D face-space scatter (PC-1 vs PC-2, coloured
by person); mean face and top eigenfaces.

**Solution 2** - dendrogram; elbow and silhouette-vs-k; GMM BIC/AIC; silhouette analysis plot;
t-SNE coloured by cluster vs true identity; cluster-vs-identity heatmap; Silhouette / Davies-Bouldin /
ARI comparison; montage of cluster members.

**Solution 3** - restoration grids (Gaussian and salt-and-pepper); PSNR/SSIM vs noise level;
privacy-utility curves; recognition-accuracy vs SSIM scatter; before/after de-identification strips;
final summary bar charts.

## Notes for the report / viva
* Solution 1 is generative and has no held-out evaluation, so its PCA is fitted on all 400 images
  (80 components). The other solutions use the stratified 75/25 split from `common.py`, fitting on the
  training split only.
* Morphing works in PCA space: two faces are projected to 80-D latent vectors, interpolated linearly
  and mapped back with `inverse_transform`. Pixel-space blending (graph 4) gives ghosted double images,
  whereas latent-space blending (graph 5) gives smoother, more face-like results.
* Novel faces are drawn from a Gaussian with zero mean and per-component variance equal to the PCA
  eigenvalues, so they follow the same spread as the real data. They are not real people.
* Reconstructions are clipped to [0, 1] before display.
* Clustering is unsupervised; true labels are only used afterwards (ARI, NMI, purity).
* DBSCAN noise points (label -1) are excluded from silhouette / Davies-Bouldin.
* Non-local means assumes Gaussian noise, so it performs poorly on salt-and-pepper noise; the median filter
  is the best choice there. PCA reconstruction works well for both because it projects onto a face subspace.
* k-Same groups faces of *different* people and replaces each by the group average, so
  recognition accuracy drops to roughly 1/k - a measurable privacy guarantee, at the cost of image detail.
* Results vary slightly with the random seed (`RANDOM_STATE` in `common.py`).