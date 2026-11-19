import numpy as np
import polars as pl
import pytest

from rmsd_map.cli import calc_umaps


def test_automatic_neighbors_use_inclusive_bounds():
    generate = calc_umaps.generate_n_neighbors_list
    assert generate(10, 9) == [9, 10]
    assert generate(100, 90) == [90, 100]
    assert generate(150, 11) == [20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
    assert generate(149, 101) == []


@pytest.mark.parametrize("fields, message", [
    ({"distances": np.eye(2)}, "'names'"),
    ({"names": ["A", "B"]}, "'distances'"),
    ({"names": ["A"], "distances": [[0, 1], [1, 0]]}, "'names'"),
    ({"names": ["A", "B"], "distances": [[0, 1]]}, "square"),
    ({"names": ["A", "B"], "distances": [[0, np.nan], [np.nan, 0]]}, "finite"),
    ({"names": ["A", "B"], "distances": [[0, 1], [2, 0]]}, "symmetric"),
    ({"names": ["A", "B"], "distances": [[1, 1], [1, 0]]}, "zero diagonal"),
])
def test_invalid_distance_archive(tmp_path, fields, message):
    archive = tmp_path / "input.npz"
    np.savez(archive, **fields)
    with pytest.raises(ValueError, match=message):
        calc_umaps.read_distances_npz(archive)


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "input.npz"
    np.savez(path, names=["DAMWOH", "ECAKIG01", "DAMWOH", "ECAKIG"],
             distances=np.ones((4, 4)) - np.eye(4))
    return path


@pytest.mark.parametrize("options,message", [
    ([], "No automatic n_neighbors values"),
    (["--neighbors", "1"], "between 2 and 3"),
    (["--neighbors", "4"], "between 2 and 3"),
    (["--neighbors", "2,2"], "duplicate values"),
    (["--min-neighbors", "7", "--max-neighbors", "6"], "Expected 2 <="),
])
def test_cli_rejects_bad_neighbors(archive, options, message, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["rmsd-map-umaps", str(archive), *options])
    with pytest.raises(SystemExit) as exc:
        calc_umaps.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("densmap", [False, True])
def test_cli_passes_densmap_and_preserves_row_ids(tmp_path, archive, monkeypatch, densmap):
    calls = []

    def make_map(neighbors, distances, densmap, seed):
        calls.append(densmap)
        assert distances.shape == (4, 4)
        return pl.DataFrame({"X": range(4), "Y": range(4), "N": [neighbors] * 4})

    class LocalPool:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, worker, neighbors):
            return list(map(worker, neighbors))

    monkeypatch.setattr(calc_umaps, "make_umap_df", make_map)
    monkeypatch.setattr(calc_umaps.mp, "Pool", lambda processes: LocalPool())
    monkeypatch.setattr(calc_umaps, "plot_umap_pages", lambda df, output: None)
    prefix = tmp_path / "maps"
    args = ["rmsd-map-umaps", str(archive), "--neighbors", "3,2", "-o", str(prefix)]
    monkeypatch.setattr("sys.argv", [*args, "--densmap"] if densmap else args)
    calc_umaps.main()
    assert calls == [densmap, densmap]
    data = pl.read_csv(prefix.with_suffix(".csv"))
    assert data["N"].to_list() == [3] * 4 + [2] * 4
    assert data["ID"].to_list() == ["DAMWOH", "ECAKIG01", "DAMWOH", "ECAKIG"] * 2
