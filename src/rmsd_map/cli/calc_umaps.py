#!/usr/bin/env python3

import math
import os

import datashader as ds
from datashader.mpl_ext import dsshow
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from umap import UMAP
import multiprocessing as mp
from functools import partial

import argparse
from pathlib import Path
# reading data matrix

def make_umap_df(neighbors, distances, densmap=False, seed=42):
    reducer = UMAP(metric='precomputed',
                   n_neighbors=neighbors,
                   densmap=densmap,
                   random_state=seed,
                   n_jobs=1,
                   init='random' if len(distances) == 3 else 'spectral',
                   min_dist = 0.0001
                   )
    reducer.fit(distances)
    df = pl.DataFrame(reducer.embedding_, schema=["X", "Y"])
    df = df.with_columns(pl.lit(neighbors).alias("N"))
    return df

def read_distances_npz(npz_file):
    archive = np.load(npz_file, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected an NPZ archive with fields 'names' and 'distances'")
    fields = {}
    with archive:
        for field in ("names", "distances"):
            try:
                fields[field] = archive[field]
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid or missing NPZ field '{field}': {exc}") from exc

    names = fields["names"]
    distances = fields["distances"]
    if distances.ndim != 2 or not len(distances) or distances.shape[0] != distances.shape[1]:
        raise ValueError("NPZ field 'distances' must be a nonempty square matrix")
    if names.shape != (len(distances),):
        raise ValueError("NPZ field 'names' must contain one ID per row of 'distances'")
    if distances.dtype.kind not in "fiu" or not np.isfinite(distances).all():
        raise ValueError("NPZ field 'distances' must contain finite real numbers")
    if (distances < 0).any() or np.any(distances.diagonal() != 0):
        raise ValueError("NPZ field 'distances' must be nonnegative with a zero diagonal")
    if not np.array_equal(distances, distances.T):
        raise ValueError("NPZ field 'distances' must be symmetric")
    return names, distances


def generate_n_neighbors_list(max_neighbors: int = 500,
                              min_neighbors: int = 4):
    # Both requested bounds are inclusive. Keep the same grid across runs.
    candidates = [*range(2, 10), *range(10, 100, 10), *range(100, max_neighbors + 1, 50)]
    return [value for value in candidates if min_neighbors <= value <= max_neighbors]

def plot_umap_pages(df, output_file, grid_W = 2, grid_H = 3,
                    fig_W = 600, fig_H = 600,
                    page_W = 8.27, page_H = 11.69):
    slots_per_page = grid_W * grid_H
    N_NEIGHBORS_LIST = np.sort(np.unique(df["N"]))
    N_NEIGHBORS = len(N_NEIGHBORS_LIST)
    n_pages = math.ceil(N_NEIGHBORS / slots_per_page)

    with PdfPages(output_file) as pdf:
        for p in range(n_pages):
            fig, axes = plt.subplots(nrows=grid_H, ncols=grid_W,
                                 figsize=(page_W, page_H),
                                 squeeze=False)
            start = p * slots_per_page
            stop  = min(start + slots_per_page, N_NEIGHBORS)
            for i in range(slots_per_page):
                r, c = divmod(i, grid_W)
                ax = axes[r][c]
                idx = start + i
                if idx < stop:
                    NN = N_NEIGHBORS_LIST[idx]
                    coords = df.filter(N = NN).select(["X","Y"]).to_pandas()
                    ax.set_title(f'n_neighbors={NN}')
                    _ = dsshow(coords, ds.Point("X", "Y"), ax = ax,
                               plot_width = fig_W, plot_height = fig_H, aspect = 'auto')
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
       description='Calculate UMAP embeddings for multiple n_neighbors values',
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', type=Path,
                        help='Input NPZ file with distance matrix')
    parser.add_argument('-o', '--output-prefix', type=str, default='umap_results',
                        help='Output file prefix for CSV and PDF files')
    parser.add_argument('-d', '--densmap', action='store_true',
                        help='Use densmap variant of UMAP'),
    parser.add_argument('-s', '--seed', type = int, default=42,
                        help='Random seed for reproducible UMAP runs')
    group_neighbors = parser.add_mutually_exclusive_group()
    group_neighbors.add_argument('--neighbors', type=lambda s: [int(x) for x in s.split(',')],
                    help='Comma-separated list of n_neighbors values: "5,10,15,20,50,100"')
    group_neighbors.add_argument('--auto-neighbors', action='store_true',
                        help='Use automatic n_neighbors generation (default)')
    parser.add_argument('--min-neighbors', type=int, default=4,
                        help='Minimum n_neighbors value (for auto mode)')
    parser.add_argument('--max-neighbors', type=int, default=500,
                        help='Maximum n_neighbors value (for auto mode)')

    args = parser.parse_args()

    try:
        names, distances = read_distances_npz(args.input_file)
    except (OSError, ValueError) as exc:
        parser.error(f"{args.input_file}: {exc}")
    count = len(names)
    if count < 3:
        parser.error("UMAP needs at least three fragments")

    if args.neighbors is not None:
        neighbors_list = args.neighbors
        if any(not 2 <= value < count for value in neighbors_list):
            parser.error(f"--neighbors values must be between 2 and {count - 1}")
        if len(neighbors_list) != len(set(neighbors_list)):
            parser.error("--neighbors must not contain duplicate values")
    else:
        if not 2 <= args.min_neighbors <= args.max_neighbors:
            parser.error("Expected 2 <= --min-neighbors <= --max-neighbors")
        neighbors_list = generate_n_neighbors_list(
            min(args.max_neighbors, count // 2), args.min_neighbors)
        if not neighbors_list:
            parser.error("No automatic n_neighbors values for this dataset and bounds; "
                         "supply an explicit list with --neighbors")

    print(f"Computing UMAP for n_neighbors={neighbors_list}")
    runner = partial(make_umap_df, distances=distances, densmap=args.densmap, seed=args.seed)
    with mp.Pool(processes=min(len(neighbors_list), os.cpu_count() or 1)) as pool:
        dfs = pool.map(runner, neighbors_list)
    ids = pl.Series("ID", names)
    data = pl.concat([df.with_columns(ids) for df in dfs], how="vertical")
    data.write_csv(f'{args.output_prefix}.csv')
    plot_umap_pages(data, f'{args.output_prefix}.pdf')

if __name__ == '__main__':
    main()
