"""
Confirmatory Test Suite Data Acquisition Script (scripts/fetch_test_data.py).

Acquires, standardizes, caches, and hashes all 27 pre-registered Confirmatory TEST datasets:
- 15 Oncology Microarray Benchmarks (Schliep Lab CompCancer archive)
- 5 Large-K Benchmarks with K >= 15 (OpenML / 20 Newsgroups)
- 3 Low-D Physical/Biomedical Controls (OpenML)
- 4 Single-Cell RNA-seq Count Cohorts (10x Genomics PBMC, GEO Drop-seq, Baron Loom)

Strictly enforced by Rule R2:
Must be executed only after git freeze tag freeze-v1.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nullconsensus.data.guard import verify_test_access_allowed  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache" / "test"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = REPO_ROOT / "data" / "test_manifest.json"


def compute_sha256(filepath: Path) -> str:
    """Computes SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_souto_file(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parses a Souto et al. tab-delimited microarray matrix."""
    lines = text.strip().splitlines()
    sample_line = [s.strip().strip('"') for s in lines[0].split("\t") if s.strip()]
    raw_class_line = [c.strip().strip('"') for c in lines[1].split("\t") if c.strip()]

    if len(raw_class_line) == len(sample_line) + 1:
        class_line = raw_class_line[1:]
    else:
        class_line = raw_class_line[: len(sample_line)]

    n_samples = len(sample_line)
    gene_rows = []
    for line in lines[2:]:
        parts = line.split("\t")
        try:
            float(parts[0].strip().strip('"'))
            start_idx = 0
        except ValueError:
            start_idx = 1
        vals = []
        for p in parts[start_idx : start_idx + n_samples]:
            try:
                vals.append(float(p.strip().strip('"')))
            except ValueError:
                vals.append(np.nan)
        if len(vals) == n_samples:
            gene_rows.append(vals)

    gene_mat = np.array(gene_rows, dtype=np.float32)  # (D, N)
    X = gene_mat.T  # (N, D)

    # Impute missing values with column mean if any (e.g. cDNA spot dropouts)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    y = LabelEncoder().fit_transform(class_line)
    X_std = StandardScaler().fit_transform(X).astype(np.float32)
    return X_std, y.astype(np.int64)


def fetch_cancer_microarrays() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fetches and processes the 15 cancer microarray cohorts from Schliep Lab."""
    base_url = "https://schlieplab.org/Static/Supplements/CompCancer/"
    cancer_specs = [
        ("armstrong2002", "Affymetrix/armstrong-2002-v2/armstrong-2002-v2_database.txt"),
        ("bhattacharjee2001", "Affymetrix/bhattacharjee-2001/bhattacharjee-2001_database.txt"),
        ("chowdary2006", "Affymetrix/chowdary-2006/chowdary-2006_database.txt"),
        ("dyrskjot2003", "Affymetrix/dyrskjot-2003/dyrskjot-2003_database.txt"),
        ("gordon2002", "Affymetrix/gordon-2002/gordon-2002_database.txt"),
        ("pomeroy2002", "Affymetrix/pomeroy-2002-v2/pomeroy-2002-v2_database.txt"),
        ("ramaswamy2001", "Affymetrix/ramaswamy-2001/ramaswamy-2001_database.txt"),
        ("shipp2002", "Affymetrix/shipp-2002-v1/shipp-2002-v1_database.txt"),
        ("singh2002", "Affymetrix/singh-2002/singh-2002_database.txt"),
        ("su2001", "Affymetrix/su-2001/su-2001_database.txt"),
        ("west2001", "Affymetrix/west-2001/west-2001_database.txt"),
        ("yeoh2002", "Affymetrix/yeoh-2002-v2/yeoh-2002-v2_database.txt"),
        ("alizadeh2000", "CDNA/alizadeh-2000-v1/alizadeh-2000-v1_database.txt"),
        ("nutt2003", "Affymetrix/nutt-2003-v1/nutt-2003-v1_database.txt"),
        ("lapointe2004", "CDNA/lapointe-2004-v1/lapointe-2004-v1_database.txt"),
    ]
    out = {}
    for ds_id, path in cancer_specs:
        verify_test_access_allowed(ds_id)
        url = base_url + path
        print(f"Fetching cancer microarray: {ds_id} ({url})...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode(errors="ignore")
        X, y = parse_souto_file(text)
        out[ds_id] = (X, y)
        print(f"  -> {ds_id}: shape={X.shape}, classes={len(np.unique(y))}", flush=True)
    return out


def fetch_large_k_benchmarks() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fetches the 5 large-K benchmarks."""
    out = {}

    # 1. COIL-20
    verify_test_access_allowed("coil20")
    print("Fetching coil20...", flush=True)
    ds = sklearn.datasets.fetch_openml(data_id=40979, as_frame=False, parser="auto")
    X_c = StandardScaler().fit_transform(np.asarray(ds.data, dtype=np.float32))
    y_c = LabelEncoder().fit_transform(ds.target).astype(np.int64)
    out["coil20"] = (X_c, y_c)
    print(f"  -> coil20: shape={X_c.shape}, classes={len(np.unique(y_c))}", flush=True)

    # 2. UMIST-Faces
    verify_test_access_allowed("umist_faces")
    print("Fetching umist_faces...", flush=True)
    ds = sklearn.datasets.fetch_openml(data_id=41084, as_frame=False, parser="auto")
    X_u = StandardScaler().fit_transform(np.asarray(ds.data, dtype=np.float32)[:, :644])
    y_u = LabelEncoder().fit_transform(ds.target).astype(np.int64)
    out["umist_faces"] = (X_u, y_u)
    print(f"  -> umist_faces: shape={X_u.shape}, classes={len(np.unique(y_u))}", flush=True)

    # 3. Texture-16
    verify_test_access_allowed("texture16")
    print("Fetching texture16...", flush=True)
    ds = sklearn.datasets.fetch_openml(data_id=40498, as_frame=False, parser="auto")
    X_t = StandardScaler().fit_transform(np.asarray(ds.data, dtype=np.float32))
    y_t = LabelEncoder().fit_transform(ds.target).astype(np.int64)
    out["texture16"] = (X_t, y_t)
    print(f"  -> texture16: shape={X_t.shape}, classes={len(np.unique(y_t))}", flush=True)

    # 4. Letter Recognition
    verify_test_access_allowed("letter_recognition")
    print("Fetching letter_recognition...", flush=True)
    ds = sklearn.datasets.fetch_openml(data_id=6, as_frame=False, parser="auto")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ds.data), size=2000, replace=False)
    X_l = StandardScaler().fit_transform(np.asarray(ds.data[idx], dtype=np.float32))
    y_l = LabelEncoder().fit_transform(ds.target[idx]).astype(np.int64)
    out["letter_recognition"] = (X_l, y_l)
    print(f"  -> letter_recognition: shape={X_l.shape}, classes={len(np.unique(y_l))}", flush=True)

    # 5. News-20 Fresh Subset (15 categories)
    verify_test_access_allowed("news20_subset")
    print("Fetching news20_subset...", flush=True)
    categories = [
        "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos",
        "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt",
        "sci.electronics", "sci.med", "sci.space"
    ]
    news = sklearn.datasets.fetch_20newsgroups(subset="test", categories=categories, remove=("headers", "footers", "quotes"))
    vec = TfidfVectorizer(max_features=2000, stop_words="english")
    X_n = vec.fit_transform(news.data).toarray().astype(np.float32)[:1500]
    y_n = news.target[:1500].astype(np.int64)
    out["news20_subset"] = (X_n, y_n)
    print(f"  -> news20_subset: shape={X_n.shape}, classes={len(np.unique(y_n))}", flush=True)

    return out


def fetch_low_d_controls() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fetches the 3 low-D controls."""
    out = {}

    # 1. Parkinsons
    verify_test_access_allowed("parkinsons")
    print("Fetching parkinsons...", flush=True)
    ds = sklearn.datasets.fetch_openml(data_id=1488, as_frame=False, parser="auto")
    X_p = StandardScaler().fit_transform(np.asarray(ds.data, dtype=np.float32))
    y_p = LabelEncoder().fit_transform(ds.target).astype(np.int64)
    out["parkinsons"] = (X_p, y_p)
    print(f"  -> parkinsons: shape={X_p.shape}, classes={len(np.unique(y_p))}", flush=True)

    # 2. WDBC
    verify_test_access_allowed("wdbc")
    print("Fetching wdbc...", flush=True)
    ds = sklearn.datasets.fetch_openml(data_id=1510, as_frame=False, parser="auto")
    X_w = StandardScaler().fit_transform(np.asarray(ds.data, dtype=np.float32))
    y_w = LabelEncoder().fit_transform(ds.target).astype(np.int64)
    out["wdbc"] = (X_w, y_w)
    print(f"  -> wdbc: shape={X_w.shape}, classes={len(np.unique(y_w))}", flush=True)

    # 3. Ionosphere
    verify_test_access_allowed("ionosphere")
    print("Fetching ionosphere...", flush=True)
    ds = sklearn.datasets.fetch_openml(data_id=59, as_frame=False, parser="auto")
    X_i = StandardScaler().fit_transform(np.asarray(ds.data, dtype=np.float32))
    y_i = LabelEncoder().fit_transform(ds.target).astype(np.int64)
    out["ionosphere"] = (X_i, y_i)
    print(f"  -> ionosphere: shape={X_i.shape}, classes={len(np.unique(y_i))}", flush=True)

    return out


def fetch_single_cell_cohorts() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fetches the 4 scRNA-seq count cohorts."""
    import h5py
    import scanpy as sc

    out = {}

    # PBMC cohorts
    verify_test_access_allowed("zheng_b_cells")
    verify_test_access_allowed("zheng_t_cells")
    print("Preparing Zheng PBMC single-cell cohorts...", flush=True)
    adata = sc.datasets.pbmc3k_processed()
    X_counts = adata.raw.X.toarray() if adata.raw is not None else adata.X.toarray()
    labels = np.array(adata.obs["louvain"].astype(str))

    # B-cells cohort
    mask_b = np.isin(labels, ["B cells", "CD14+ Monocytes", "NK cells"])
    X_b = X_counts[mask_b][:1000].astype(np.float32)
    # log1p library size scaling
    lib_b = np.sum(X_b, axis=1, keepdims=True)
    lib_b[lib_b == 0] = 1.0
    X_b = np.log1p(X_b * 1e4 / lib_b).astype(np.float32)
    y_b = LabelEncoder().fit_transform(labels[mask_b][:1000]).astype(np.int64)
    out["zheng_b_cells"] = (X_b, y_b)
    print(f"  -> zheng_b_cells: shape={X_b.shape}, classes={len(np.unique(y_b))}", flush=True)

    # T-cells cohort
    mask_t = np.isin(labels, ["CD4 T cells", "CD8 T cells", "NK cells", "FCGR3A+ Monocytes"])
    X_t = X_counts[mask_t][:1000].astype(np.float32)
    lib_t = np.sum(X_t, axis=1, keepdims=True)
    lib_t[lib_t == 0] = 1.0
    X_t = np.log1p(X_t * 1e4 / lib_t).astype(np.float32)
    y_t = LabelEncoder().fit_transform(labels[mask_t][:1000]).astype(np.int64)
    out["zheng_t_cells"] = (X_t, y_t)
    print(f"  -> zheng_t_cells: shape={X_t.shape}, classes={len(np.unique(y_t))}", flush=True)

    # Baron Pancreas cohort
    verify_test_access_allowed("baron_pancreas")
    print("Preparing Baron pancreas cohort...", flush=True)
    baron_path = REPO_ROOT / "data" / "cache" / "baron.loom"
    if not baron_path.exists():
        baron_url = "https://raw.githubusercontent.com/YosefLab/scVI-data/master/baron.loom"
        print(f"Downloading baron.loom from {baron_url}...", flush=True)
        urllib.request.urlretrieve(baron_url, baron_path)

    with h5py.File(baron_path, "r") as f:
        # matrix is (genes, cells)
        raw_mat = np.array(f["matrix"][:2000, :1200], dtype=np.float32).T  # (1200, 2000)
        raw_clust = np.array(f["col_attrs"]["ClusterID"][:1200])

    # Top 8 classes
    u_c, c_counts = np.unique(raw_clust, return_counts=True)
    top_8_c = u_c[np.argsort(-c_counts)[:8]]
    mask_baron = np.isin(raw_clust, top_8_c)
    X_bp = raw_mat[mask_baron]
    lib_bp = np.sum(X_bp, axis=1, keepdims=True)
    lib_bp[lib_bp == 0] = 1.0
    X_bp = np.log1p(X_bp * 1e4 / lib_bp).astype(np.float32)
    y_bp = LabelEncoder().fit_transform(raw_clust[mask_baron]).astype(np.int64)
    out["baron_pancreas"] = (X_bp, y_bp)
    print(f"  -> baron_pancreas: shape={X_bp.shape}, classes={len(np.unique(y_bp))}", flush=True)

    # Klein Embryonic Stem Cells
    verify_test_access_allowed("klein_embryonic")
    print("Preparing Klein embryonic stem cells cohort...", flush=True)
    # 4 differentiation time points: d0, d2, d4, d7
    klein_cache = REPO_ROOT / "data" / "cache" / "klein"
    klein_cache.mkdir(parents=True, exist_ok=True)
    timepoints = [
        ("d0", "GSM1599494_ES_d0_main.csv.bz2"),
        ("d2", "GSM1599497_ES_d2_LIFminus.csv.bz2"),
        ("d4", "GSM1599498_ES_d4_LIFminus.csv.bz2"),
        ("d7", "GSM1599499_ES_d7_LIFminus.csv.bz2"),
    ]
    geo_base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65525/suppl/"
    xs, ys = [], []
    for label_idx, (t_name, fname) in enumerate(timepoints):
        fpath = klein_cache / fname
        if not fpath.exists():
            print(f"Downloading {fname} from GEO...", flush=True)
            urllib.request.urlretrieve(geo_base + fname, fpath)
        import bz2
        with bz2.open(fpath, "rt") as f:
            # First row is sample names, subsequent rows are gene, val1, val2...
            # Each file has ~250 cells
            rows = []
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(",")
                vals = [float(p) for p in parts[1:251]]  # take up to 250 cells
                rows.append(vals)
                if len(rows) >= 2000:
                    break
        mat = np.array(rows, dtype=np.float32).T  # (250, 2000)
        xs.append(mat)
        ys.append(np.full(len(mat), label_idx, dtype=np.int64))

    X_k = np.vstack(xs)
    y_k = np.concatenate(ys)
    lib_k = np.sum(X_k, axis=1, keepdims=True)
    lib_k[lib_k == 0] = 1.0
    X_k = np.log1p(X_k * 1e4 / lib_k).astype(np.float32)
    out["klein_embryonic"] = (X_k, y_k)
    print(f"  -> klein_embryonic: shape={X_k.shape}, classes={len(np.unique(y_k))}", flush=True)

    return out


def main():
    print("=" * 70)
    print("PHASE B6: CONFIRMATORY TEST SUITE ACQUISITION & STANDARDIZATION")
    print(f"Target Cache: {CACHE_DIR}")
    print("=" * 70, flush=True)

    # 1. Fetch all groups
    datasets = {}
    print("\n[Group 1/4] Fetching 15 Oncology Microarray Benchmarks...")
    datasets.update(fetch_cancer_microarrays())

    print("\n[Group 2/4] Fetching 5 Large-K Benchmarks...")
    datasets.update(fetch_large_k_benchmarks())

    print("\n[Group 3/4] Fetching 3 Low-D Controls...")
    datasets.update(fetch_low_d_controls())

    print("\n[Group 4/4] Fetching 4 scRNA-seq Count Cohorts...")
    datasets.update(fetch_single_cell_cohorts())

    print("\n" + "=" * 70)
    print(f"Total datasets fetched: {len(datasets)} / 27")
    print("Saving .npz files and computing SHA256 checksums...")
    print("=" * 70)

    hashes = {}
    for ds_id, (X, y) in sorted(datasets.items()):
        npz_file = CACHE_DIR / f"{ds_id}.npz"
        np.savez_compressed(npz_file, X=X, y=y)
        sha = compute_sha256(npz_file)
        hashes[ds_id] = sha
        print(f"SAVED: {ds_id:20s} -> shape={X.shape}, K={len(np.unique(y)):2d}, SHA256={sha[:16]}...")

    # Update test_manifest.json with real SHA256
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    for entry in manifest_data["datasets"]:
        ds_id = entry["id"]
        if ds_id in hashes:
            entry["sha256"] = hashes[ds_id]
            entry["n_samples"] = int(datasets[ds_id][0].shape[0])
            entry["n_features"] = int(datasets[ds_id][0].shape[1])
            entry["n_clusters"] = int(len(np.unique(datasets[ds_id][1])))
            entry["d_over_n"] = round(float(entry["n_features"] / entry["n_samples"]), 2)
            entry["class_balance"] = [int(c) for c in np.bincount(datasets[ds_id][1])]

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print("\nSuccessfully updated data/test_manifest.json with confirmed SHA256 hashes!")


if __name__ == "__main__":
    main()
