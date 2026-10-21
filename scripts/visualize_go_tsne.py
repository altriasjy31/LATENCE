#!/usr/bin/env python

import os
import random

import click as ck
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@ck.command()
@ck.option("--embedding-file", "-i", required=True, type=ck.Path(exists=True),
           help="Input pickle file containing GO embeddings.")
@ck.option("--out-image", "-o", default="go_tsne.png",
           help="Output image path, e.g. go_tsne.png or go_tsne.pdf.")
@ck.option("--out-coords", default=None,
           help="Optional output CSV file for t-SNE coordinates.")
@ck.option("--id-column", default="auto",
           help="Column containing GO IDs/classes. Use auto to infer.")
@ck.option("--vector-column", default="auto",
           help="Embedding column. Use auto to prefer center if available, otherwise embeddings.")
@ck.option("--strip-radius/--no-strip-radius", default=False,
           help="If true, remove the last dimension from embeddings. Useful for ELEmbeddings if radius is stored as the last dimension.")
@ck.option("--max-points", default=0, type=int,
           help="Randomly sample at most this many points. 0 means use all.")
@ck.option("--only-go/--all-classes", default=True,
           help="Only visualize entries whose ID contains GO_.")
@ck.option("--perplexity", default=30.0, type=float,
           help="t-SNE perplexity. Must be smaller than number of points.")
@ck.option("--learning-rate", default="auto",
           help="t-SNE learning rate. Use auto or a float.")
@ck.option("--iterations", default=1000, type=int,
           help="t-SNE optimization iterations.")
@ck.option("--metric", default="euclidean",
           help="Distance metric for t-SNE, e.g. euclidean or cosine.")
@ck.option("--standardize/--no-standardize", default=True,
           help="Standardize embeddings before t-SNE.")
@ck.option("--point-size", default=5.0, type=float,
           help="Scatter point size.")
@ck.option("--alpha", default=0.65, type=float,
           help="Scatter point alpha.")
@ck.option("--figsize", default="8,7",
           help="Figure size as width,height.")
@ck.option("--dpi", default=300, type=int,
           help="Output image DPI.")
@ck.option("--label-top", default=0, type=int,
           help="Label first N points after sampling. 0 means no labels.")
@ck.option("--seed", default=100, type=int,
           help="Random seed.")
def main(
    embedding_file,
    out_image,
    out_coords,
    id_column,
    vector_column,
    strip_radius,
    max_points,
    only_go,
    perplexity,
    learning_rate,
    iterations,
    metric,
    standardize,
    point_size,
    alpha,
    figsize,
    dpi,
    label_top,
    seed,
):
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_pickle(embedding_file)

    print("Loaded:", embedding_file)
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    id_col = infer_id_column(df, id_column)
    vec_col = infer_vector_column(df, vector_column)

    print("ID column:", id_col)
    print("Vector column:", vec_col)

    # Keep GO terms only if requested.
    if only_go:
        mask = df[id_col].astype(str).str.contains("GO_")
        df = df.loc[mask].copy()
        print("Rows after GO filter:", len(df))

    if len(df) == 0:
        raise ValueError("No entries left after filtering.")

    # Optional sampling.
    if max_points and max_points > 0 and len(df) > max_points:
        df = df.sample(n=max_points, random_state=seed).copy()
        print("Rows after sampling:", len(df))

    ids = df[id_col].astype(str).tolist()

    X = np.vstack([
        to_vector(x, strip_radius=strip_radius)
        for x in df[vec_col].values
    ]).astype(np.float32)

    print("Embedding matrix shape:", X.shape)

    # Remove rows with NaN or inf.
    finite_mask = np.isfinite(X).all(axis=1)
    if not finite_mask.all():
        print("Warning: removing non-finite vectors:", np.sum(~finite_mask))
        X = X[finite_mask]
        ids = [x for x, keep in zip(ids, finite_mask) if keep]

    n = X.shape[0]

    if n < 3:
        raise ValueError("Need at least 3 points for t-SNE.")

    if perplexity >= n:
        new_perplexity = max(2.0, min(30.0, (n - 1) / 3.0))
        print(f"Perplexity {perplexity} is too large for n={n}; using {new_perplexity}")
        perplexity = new_perplexity

    if standardize:
        X = StandardScaler().fit_transform(X)

    lr = parse_learning_rate(learning_rate)

    print("Running t-SNE...")
    print("n_points:", n)
    print("perplexity:", perplexity)
    print("learning_rate:", lr)
    print("iterations:", iterations)
    print("metric:", metric)

    coords = run_tsne(
        X,
        perplexity=perplexity,
        learning_rate=lr,
        iterations=iterations,
        metric=metric,
        seed=seed,
    )

    plot_tsne(
        coords=coords,
        ids=ids,
        out_image=out_image,
        point_size=point_size,
        alpha=alpha,
        figsize=figsize,
        dpi=dpi,
        label_top=label_top,
        title=f"GO embeddings t-SNE\n{os.path.basename(embedding_file)}",
    )

    if out_coords:
        out_df = pd.DataFrame({
            "id": ids,
            "tsne_1": coords[:, 0],
            "tsne_2": coords[:, 1],
        })
        out_df.to_csv(out_coords, index=False)
        print("Saved coordinates:", out_coords)

    print("Saved image:", out_image)


def infer_id_column(df, id_column):
    if id_column != "auto":
        if id_column not in df.columns:
            raise ValueError(f"ID column not found: {id_column}")
        return id_column

    candidates = [
        "classes",
        "class",
        "term",
        "terms",
        "id",
        "go_id",
        "relations",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not infer ID column. Available columns: "
        + ", ".join(df.columns)
    )


def infer_vector_column(df, vector_column):
    if vector_column != "auto":
        if vector_column not in df.columns:
            raise ValueError(f"Vector column not found: {vector_column}")
        return vector_column

    # For ELEmbeddings, prefer center if present.
    if "center" in df.columns:
        return "center"

    if "embeddings" in df.columns:
        return "embeddings"

    if "embedding" in df.columns:
        return "embedding"

    raise ValueError(
        "Could not infer vector column. Available columns: "
        + ", ".join(df.columns)
    )


def to_vector(x, strip_radius=False):
    """
    Convert a cell value to a 1D numpy vector.

    Supports:
    - numpy arrays
    - Python lists
    - tuples of arrays
    - strings containing list-like values, if necessary
    """
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, list):
        arr = np.asarray(x)
    elif isinstance(x, tuple):
        parts = [np.asarray(v).reshape(-1) for v in x]
        arr = np.concatenate(parts, axis=0)
    else:
        # Fallback for unusual serialized values.
        arr = np.asarray(x)

    arr = arr.astype(np.float32).reshape(-1)

    if strip_radius:
        if arr.shape[0] <= 1:
            raise ValueError("Cannot strip radius from vector with length <= 1.")
        arr = arr[:-1]

    return arr


def parse_learning_rate(x):
    if isinstance(x, str):
        if x.lower() == "auto":
            return "auto"
        return float(x)
    return x


def run_tsne(X, perplexity, learning_rate, iterations, metric, seed):
    """
    Compatibility wrapper for scikit-learn versions.

    Some sklearn versions use max_iter; older versions used n_iter.
    """
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=iterations,
            init="pca",
            metric=metric,
            random_state=seed,
            verbose=1,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=iterations,
            init="pca",
            metric=metric,
            random_state=seed,
            verbose=1,
        )

    return tsne.fit_transform(X)


def plot_tsne(
    coords,
    ids,
    out_image,
    point_size=5.0,
    alpha=0.65,
    figsize="8,7",
    dpi=300,
    label_top=0,
    title=None,
):
    width, height = parse_figsize(figsize)

    plt.figure(figsize=(width, height))

    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    if label_top and label_top > 0:
        n_label = min(label_top, len(ids))
        for i in range(n_label):
            plt.text(
                coords[i, 0],
                coords[i, 1],
                ids[i],
                fontsize=6,
                alpha=0.85,
            )

    if title:
        plt.title(title)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_image) or ".", exist_ok=True)
    plt.savefig(out_image, dpi=dpi)
    plt.close()


def parse_figsize(s):
    if isinstance(s, str):
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError("figsize should be formatted as width,height, e.g. 8,7")
        return float(parts[0]), float(parts[1])
    return s


if __name__ == "__main__":
    main()