import numpy as np
import pytest

from rmsd_map.cli import calc_distances
from rmsd_map.mol_io.fragment import Fragment
from rmsd_map.rmsd.pipelines import (
    align_fragments,
    align_gen_batched,
    chain_fragments,
    chain_fragments_naive,
    distance_matrix_fragments,
    partial_align_fragments,
)


@pytest.mark.parametrize("proper_flag", [[], ["--proper_only"]])
def test_distances_cli_keeps_repeated_refcodes(tmp_path, monkeypatch, proper_flag):
    cor = tmp_path / "water.cor"
    cor.write_text("""WATERX01 **FRAG** 1
O1 0 0 0
H1 0.95 0.45 0
H2 -0.65 0.67 0
WATERX01 **FRAG** 2
O1 0 0 0
H1 -0.65 0.67 0
H2 0.95 0.45 0
""")
    prefix = tmp_path / "distances"
    monkeypatch.setattr("sys.argv", ["rmsd-map-distances", str(cor), "-o",
                                    str(prefix), *proper_flag])
    calc_distances.main()
    with np.load(prefix.with_suffix(".npz")) as data:
        assert data["names"].tolist() == ["WATERX01", "WATERX01"]
        assert data["distances"].shape == (2, 2)
        assert data["distances"][0, 1] < 1e-3
    header, pair = prefix.with_suffix(".csv").read_text().splitlines()
    assert header == "ID_A,ID_B,Dist"
    assert pair.split(",")[:2] == ["WATERX01", "WATERX01"]


def test_empty_and_single_fragment(simple_fragment):
    with pytest.raises(ValueError, match="No fragments"):
        distance_matrix_fragments([], 2, 1)
    matrix, df = distance_matrix_fragments([simple_fragment], 2, 1)
    np.testing.assert_array_equal(matrix, [[0.0]])
    assert df.shape == (0, 3)
    assert df.columns == ["ID_A", "ID_B", "Dist"]


def test_fragment_compatibility(simple_fragment):
    other = simple_fragment.replace(id="DIFFERENT", elements=np.array(["H", "H", "O"]))
    with pytest.raises(ValueError, match="DIFFERENT has a different element order"):
        distance_matrix_fragments([simple_fragment, other], 2, 1)
    fewer = other.replace(elements=other.elements[:2], coords=other.coords[:2])
    with pytest.raises(ValueError, match="DIFFERENT has 2 atoms, expected 3"):
        distance_matrix_fragments([simple_fragment, fewer], 2, 1)


def test_alignment_batch_size():
    batches = list(align_gen_batched(16, 2, 5))
    assert [len(i) for i, _ in batches] == [5, 5, 5, 1]
    np.testing.assert_array_equal(np.concatenate([i for i, _ in batches]), np.arange(16))
    np.testing.assert_array_equal(np.concatenate([j for _, j in batches]), np.full(16, 2))
    with pytest.raises(ValueError, match="positive"):
        list(align_gen_batched(16, 2, 0))


@pytest.fixture
def fragment_series():
    coords = np.array([[0, 0, 0], [1.3, 0, 0], [-0.45, 1.3, 0],
                       [-0.3, -0.4, 1.5], [0, 0, -1.7]], dtype=np.float32)
    elements = np.array(["C", "F", "O", "N", "Cl"])
    rng = np.random.default_rng(123)
    fragments = []
    for i in range(16):
        angle = i * 0.2
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        xyz = (coords + rng.normal(0, 0.02, coords.shape)) @ rotation + [40*i, -30*i, 10*i]
        fragments.append(Fragment(str(i), elements, xyz.astype(np.float32)))
    return fragments


@pytest.mark.parametrize("align,args", [
    (align_fragments, ()),
    (partial_align_fragments, ([0, 1, 2, 3],)),
])
def test_alignment_matches_one_batch(align, args, fragment_series):
    actual = align(fragment_series, *args, n_center=3, batch_size=5)
    expected = align(fragment_series, *args, n_center=3, batch_size=32)
    assert [f.id for f in actual] == [f.id for f in fragment_series]
    np.testing.assert_allclose(np.stack([f.coords for f in actual]),
                               np.stack([f.coords for f in expected]), atol=1e-5)
    if args:
        reference = fragment_series[3].coords
        center = reference[args[0]].mean(axis=0)
        assert np.linalg.norm(reference.mean(axis=0) - center) > 0.1
        np.testing.assert_allclose(actual[3].coords, reference - center, atol=1e-5)


def test_chains_preserve_frame_geometry_and_match(fragment_series):
    frames = fragment_series[:5]
    actual = chain_fragments(frames)
    expected = chain_fragments_naive(frames)
    np.testing.assert_array_equal(actual[0].coords, frames[0].coords)
    for original, result, baseline in zip(frames, actual, expected):
        assert original.id == result.id == baseline.id
        np.testing.assert_allclose(result.coords, baseline.coords, atol=1e-5)
        a, b = original.coords, result.coords
        np.testing.assert_allclose(np.linalg.norm(a[:, None] - a, axis=-1),
                                   np.linalg.norm(b[:, None] - b, axis=-1), atol=1e-5)
    for prev, current in zip(actual, actual[1:]):
        observed = np.sqrt(np.mean(np.sum((prev.coords - current.coords)**2, axis=-1)))
        matrix, _ = distance_matrix_fragments([prev, current], 2, 1)
        np.testing.assert_allclose(observed, matrix[0, 1], atol=1e-4)
