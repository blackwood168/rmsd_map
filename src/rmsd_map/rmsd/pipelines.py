#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from rmsd_map.rmsd.kabsch_jax import calculate_pairs, kabsch_full_proper, kabsch_full_improp, kabsch_rmsd_improp, Reporter

def _stack_fragment_coords(fragments):
    if len(fragments) == 0:
        raise ValueError("No fragments to compare")
    first = fragments[0]
    for fragment in fragments:
        if len(fragment.elements) != len(first.elements):
            raise ValueError(
                f"Fragment {fragment.id} has {len(fragment.elements)} atoms, "
                f"expected {len(first.elements)} as in {first.id}"
            )
        if not np.array_equal(fragment.elements, first.elements):
            raise ValueError(f"Fragment {fragment.id} has a different element order from {first.id}")
        if not len(fragment.elements) or np.shape(fragment.coords) != (len(fragment.elements), 3):
            raise ValueError(f"Fragment {fragment.id} must have one XYZ row per atom and at least one atom")
    return np.stack([fragment.coords for fragment in fragments])


def distance_matrix_stream(coords,
                           perm,
                           batch_size,
                           report_spacing,
                           allow_improper = True):
    if batch_size <= 0 or report_spacing <= 0:
        raise ValueError("Batch size and report interval must be positive")
    if allow_improper:
        core = kabsch_rmsd_improp
    else:
        core = kabsch_full_proper
    AB = jax.device_put(coords) # AB: (M, N, 3)
    perm_d  = jax.device_put(perm)
    AB = AB - AB.mean(axis=1, keepdims=True)
    M , N, _ = AB.shape
    T = M * (M-1) // 2
    pairs = triu_gen_batched(M, 1, batch_size)
    logger = Reporter(T, report_spacing)
    for jax_dict in calculate_pairs(AB, perm_d, pairs, core, logger):
        np_dict = {k:np.asarray(v) for k,v in jax_dict.items()}
        yield np_dict

def distance_matrix_fragments(fragments, batch_size, report_spacing, allow_improper=True):
    if batch_size <= 0 or report_spacing <= 0:
        raise ValueError("Batch size and report interval must be positive")
    coords_np = _stack_fragment_coords(fragments)
    names = np.array([fragment.id for fragment in fragments])
    mat = np.zeros((len(fragments), len(fragments)), dtype=float)
    if len(fragments) == 1:
        return mat, pl.DataFrame(schema={"ID_A": pl.String, "ID_B": pl.String, "Dist": pl.Float32})

    perms = fragments[0].perms()
    dfs = []
    for batch in distance_matrix_stream(coords_np, perms, batch_size, report_spacing, allow_improper):
       i = batch["i"]
       j = batch["j"]
       rmsd  = batch["RMSD"]
       mat[i, j] = rmsd
       mat[j, i] = rmsd
       df =  pl.DataFrame(
           {
               'ID_A': names[i],
               'ID_B': names[j],
               'Dist': rmsd
           })
       dfs.append(df)
    df = pl.concat(dfs, how="vertical")
    return (mat, df)

def align_stream(coords,
                 perm,
                 center = 0,
                 batch_size = 4096,
                 rmsd_kernel = kabsch_full_improp):
    AB = jax.device_put(coords) # AB: (M, N, 3)
    perm_d  = jax.device_put(perm)
    AB = AB - AB.mean(axis=1, keepdims=True)
    M , _, _ = AB.shape
    pairs = align_gen_batched(M, center, batch_size)
    for jax_dict in calculate_pairs(AB, perm_d, pairs, rmsd_kernel):
        np_dict = {k:np.asarray(v) for k,v in jax_dict.items()}
        yield np_dict

def align_fragments(fragments, n_center=0, batch_size=4096):
    coords_np = _stack_fragment_coords(fragments)
    perms = fragments[n_center].perms()
    fragments_aligned = []
    for batch in align_stream(coords_np, perms, center=n_center, batch_size=batch_size):
        for i, transformed in zip(batch["i"], batch["Transformed"]):
            fragments_aligned.append(fragments[i].replace(coords=transformed))
    return np.asarray(fragments_aligned, dtype=object)


def partial_align_fragments(fragments, subset_idx, n_center=0, batch_size=4096):
    coords_np = _stack_fragment_coords(fragments)
    subset = coords_np[:, subset_idx, :]
    if subset.ndim != 3 or subset.shape[1] == 0:
        raise ValueError("Select a nonempty, one-dimensional subset of atoms")
    perms = np.arange(subset.shape[1])[None, :]
    mean_reduced = subset.mean(axis=1)
    fragments_aligned = np.empty(len(fragments), dtype=object)
    for batch in align_stream(subset, perms, center=n_center,
                              batch_size=batch_size, rmsd_kernel=kabsch_full_proper):
        for i, rot in zip(batch["i"], batch["Rotation"]):
            coords_new = (coords_np[i] - mean_reduced[i]) @ rot
            fragments_aligned[i] = fragments[i].replace(coords=coords_new)
    return fragments_aligned

def triu_gen_batched(M, k, batch_size):
    rows = []
    cols = []
    for i in range(M):
        for j in range(i + k, M):
            rows.append(i)
            cols.append(j)

            if len(rows) == batch_size:
                yield (jnp.array(rows, dtype=jnp.int32), jnp.array(cols, dtype = jnp.int32))
                rows.clear()
                cols.clear()
    if rows:
        yield (jnp.array(rows, dtype=jnp.int32), jnp.array(cols, dtype=jnp.int32))

def align_gen_batched(M, center, batch_size):
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        yield (jnp.arange(start, stop, dtype=jnp.int32),
               jnp.full(stop - start, center, dtype=jnp.int32))


def _align_chain_step(current, previous, perms):
    return kabsch_full_improp(current[None], previous[None], perms)["Transformed"][0]


def chain_fragments_naive(fragments):
    """Fit each frame to its predecessor in the coordinate system of the first frame."""
    coords = jax.device_put(_stack_fragment_coords(fragments))
    centers = coords.mean(axis=1, keepdims=True)
    centered = coords - centers
    perms = jax.device_put(fragments[0].perms())
    previous = centered[0]
    aligned = [fragments[0]]
    for i in range(1, len(fragments)):
        previous = _align_chain_step(centered[i], previous, perms)
        aligned.append(fragments[i].replace(coords=np.asarray(previous + centers[0])))
    return np.asarray(aligned, dtype=object)


def chain_fragments(fragments):
    """Fit sequential frames in the coordinate system of the first frame using JAX scan."""
    coords = jax.device_put(_stack_fragment_coords(fragments))
    if len(fragments) == 1:
        return np.asarray(fragments, dtype=object)
    centers = coords.mean(axis=1, keepdims=True)
    centered = coords - centers
    perms = jax.device_put(fragments[0].perms())

    def next_frame(previous, current):
        aligned = _align_chain_step(current, previous, perms)
        return aligned, aligned

    _, aligned = jax.lax.scan(next_frame, centered[0], centered[1:])
    aligned = np.asarray(aligned + centers[0])
    return np.asarray([fragments[0], *(fragment.replace(coords=xyz)
                       for fragment, xyz in zip(fragments[1:], aligned))], dtype=object)
