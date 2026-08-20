"""Graph ETL tests against synthetic CSVs.

Covers the things that silently produce a plausible-but-wrong brain: sign
assignment, the histamine override, index mapping, and edge dropping. Spec 5
names the order of suspicion when M2 fails — GLUT sign, W_SYN, id_map
off-by-one, pre/post swap — and everything on that list except W_SYN (which is
deliberately not in the graph) is pinned here.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flydoom import config  # noqa: E402
from flydoom.graph import ConnectomeGraph  # noqa: E402

np = pytest.importorskip("numpy")

# root_ids: 3 photoreceptors, 3 ordinary neurons, 1 outside the universe
PHOTO = [1001, 1002, 1003]
PLAIN = [2001, 2002, 2003]
ORPHAN = 9999


def write_fixture(d: Path, edges: list[tuple[int, int, int, str]]) -> Path:
    d.mkdir(parents=True, exist_ok=True)

    with open(d / "classification.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root_id", "super_class", "class", "sub_class", "side"])
        for r in PHOTO:
            w.writerow([r, "optic", "optic_lobe_intrinsic", "", "left"])
        for r in PLAIN:
            w.writerow([r, "central", "", "", "right"])

    with open(d / "consolidated_cell_types.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root_id", "primary_type", "additional_type(s)"])
        for r in PHOTO:
            w.writerow([r, "R1-6", ""])
        for r in PLAIN:
            w.writerow([r, "SomeInterneuron", ""])

    with open(d / "connections_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"])
        for pre, post, n, nt in edges:
            w.writerow([pre, post, "ME_L", n, nt])
    return d


def build(d: Path, **kw) -> ConnectomeGraph:
    return ConnectomeGraph.build(raw_dir=d, connections_file="connections_test.csv", **kw)


# -- signs -----------------------------------------------------------------


def test_glut_is_inhibitory_in_the_fly(tmp_path):
    """The spec asks for this assertion by name. Vertebrate intuition says
    glutamate excites; in flies it opens chloride channels and inhibits."""
    assert config.SIGN["GLUT"] == -1
    assert config.SIGN["GABA"] == -1
    assert config.SIGN["ACH"] == +1


def test_each_transmitter_gets_its_sign(tmp_path):
    d = write_fixture(tmp_path, [
        (2001, 2002, 10, "ACH"),
        (2002, 2003, 10, "GABA"),
        (2003, 2001, 10, "GLUT"),
        (2001, 2003, 10, "DA"),
    ])
    g = build(d)
    by_nt = dict(zip(
        ["ACH", "GABA", "GLUT", "DA"],
        [10.0, -10.0, -10.0, 10.0],
    ))
    # edges come back in join order, so match on value multiset
    assert sorted(g.signed_syn.tolist()) == sorted(by_nt.values())


def test_signed_weight_is_sign_times_syn_count(tmp_path):
    d = write_fixture(tmp_path, [(2001, 2002, 37, "GABA")])
    g = build(d)
    assert g.signed_syn.tolist() == [-37.0]


def test_w_syn_is_not_baked_into_the_graph(tmp_path):
    """The graph must hold measured quantities only. W_SYN is applied by the
    simulator, so a gain sweep needs no rebuild."""
    d = write_fixture(tmp_path, [(2001, 2002, 8, "ACH")])
    g = build(d)
    assert g.signed_syn.tolist() == [8.0]
    assert abs(g.signed_syn[0]) == 8.0 != config.W_SYN * 8


# -- the histamine override ------------------------------------------------


def test_photoreceptor_edges_are_forced_inhibitory(tmp_path):
    """R1-6 is predicted ACH but actually releases histamine, which inhibits.
    Without this override 83% of photoreceptor output is signed backwards."""
    d = write_fixture(tmp_path, [
        (1001, 2001, 20, "ACH"),   # photoreceptor, mispredicted excitatory
        (1002, 2002, 15, "GLUT"),  # photoreceptor, already inhibitory
        (2001, 2002, 30, "ACH"),   # ordinary neuron, must stay excitatory
    ])
    g = build(d)
    vals = sorted(g.signed_syn.tolist())
    assert vals == [-20.0, -15.0, 30.0]
    assert g.report.n_photoreceptor_edges_flipped == 1  # only the ACH one changed
    assert g.report.n_photoreceptor_neurons == 3


def test_override_can_be_disabled_to_show_the_damage(tmp_path):
    d = write_fixture(tmp_path, [(1001, 2001, 20, "ACH")])
    g = build(d, apply_histamine_override=False)
    assert g.signed_syn.tolist() == [20.0]  # wrong sign, on purpose


# -- index mapping ---------------------------------------------------------


def test_index_space_comes_from_classification(tmp_path):
    d = write_fixture(tmp_path, [(2001, 2002, 10, "ACH")])
    g = build(d)
    assert g.n_neurons == len(PHOTO) + len(PLAIN)
    assert sorted(g.root_ids.tolist()) == sorted(PHOTO + PLAIN)


def test_edges_touching_unmapped_neurons_are_dropped(tmp_path):
    d = write_fixture(tmp_path, [
        (2001, 2002, 10, "ACH"),
        (2001, ORPHAN, 10, "ACH"),
        (ORPHAN, 2002, 10, "ACH"),
    ])
    g = build(d)
    assert g.n_edges == 1
    assert g.report.n_edges_dropped_unmapped == 2


def test_pre_and_post_are_not_swapped(tmp_path):
    """Item 4 on the spec's order-of-suspicion list."""
    d = write_fixture(tmp_path, [(2001, 2003, 10, "ACH")])
    g = build(d)
    assert g.root_ids[g.pre_idx[0]] == 2001
    assert g.root_ids[g.post_idx[0]] == 2003


def test_index_of_round_trips(tmp_path):
    d = write_fixture(tmp_path, [(2001, 2002, 10, "ACH")])
    g = build(d)
    idx = g.index_of([2003, 1001])
    assert g.root_ids[idx].tolist() == [2003, 1001]


def test_index_of_rejects_unknown_root_id(tmp_path):
    d = write_fixture(tmp_path, [(2001, 2002, 10, "ACH")])
    g = build(d)
    with pytest.raises(KeyError, match="not in the index space"):
        g.index_of([ORPHAN])


# -- thresholding and dtypes ----------------------------------------------


def test_threshold_is_applied(tmp_path):
    d = write_fixture(tmp_path, [
        (2001, 2002, 4, "ACH"),   # below
        (2002, 2003, 5, "ACH"),   # at
        (2003, 2001, 99, "ACH"),  # above
    ])
    g = build(d, threshold=5)
    assert g.n_edges == 2


def test_dtypes_are_compact(tmp_path):
    d = write_fixture(tmp_path, [(2001, 2002, 10, "ACH")])
    g = build(d)
    assert g.pre_idx.dtype == np.int32
    assert g.post_idx.dtype == np.int32
    assert g.signed_syn.dtype == np.float32


def test_unknown_transmitter_is_rejected_loudly(tmp_path):
    d = write_fixture(tmp_path, [(2001, 2002, 10, "SOMETHING_NEW")])
    with pytest.raises(ValueError, match="unexpected nt_type"):
        build(d)


def test_null_transmitter_becomes_unk_not_a_crash(tmp_path):
    d = write_fixture(tmp_path, [(2001, 2002, 10, "")])
    g = build(d)
    assert g.report.n_unknown_nt == 1
    assert g.signed_syn.tolist() == [10.0]  # UNK defaults excitatory


# -- persistence -----------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    d = write_fixture(tmp_path / "raw", [(2001, 2002, 12, "GABA")])
    g = build(d)
    g.save(tmp_path / "processed")
    back = ConnectomeGraph.load(tmp_path / "processed")
    assert back.n_neurons == g.n_neurons
    assert back.signed_syn.tolist() == g.signed_syn.tolist()
    assert back.root_ids.tolist() == g.root_ids.tolist()
