"""Resolver machinery tests against a synthetic annotation table.

These do NOT validate anything about the real connectome — they validate that
the search ladder, grading and near-miss reporting behave. Real-data checks
live in experiments/m0_resolve.py, which needs the CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flydoom.cells import AnnotationTable, MatchKind, Status  # noqa: E402
from flydoom.registry import Handle, by_name  # noqa: E402


def rows(cell_type: str, n: int, side: str = "both", **kw):
    """n neurons of one type, split across hemispheres unless side is given."""
    out = []
    for i in range(n):
        s = side if side != "both" else ("left" if i % 2 == 0 else "right")
        out.append(
            {
                "root_id": abs(hash((cell_type, i))) % 10**15,
                "primary_type": cell_type,
                "super_class": kw.get("super_class", "central"),
                "class": kw.get("klass", ""),
                "sub_class": kw.get("sub_class", ""),
                "label": kw.get("label", ""),
                "side": s,
            }
        )
    return out


@pytest.fixture
def table() -> AnnotationTable:
    data = [
        # exact-match targets
        *rows("DNa02", 2, super_class="descending"),
        *rows("DNp01", 2, super_class="descending"),
        *rows("MDN", 4, super_class="descending"),
        *rows("MN9", 2, super_class="motor"),
        *rows("LC4", 60, super_class="visual_projection"),
        *rows("LPLC2", 80, super_class="visual_projection"),
        # prefix-match targets: "KC" must gather every subtype
        *rows("KCg-m", 2000),
        *rows("KCab-c", 1200),
        *rows("KCg-d", 300),
        *rows("MBON01", 40),
        *rows("MBON12", 30),
        # short-pattern guard: "V" must not vacuum up VA2/VA1v/etc
        *rows("ORN_V", 40, super_class="sensory", klass="olfactory"),
        *rows("ORN_VA2", 30, super_class="sensory", klass="olfactory"),
        *rows("ORN_DA2", 30, super_class="sensory", klass="olfactory"),
        # fuzzy-match target: only reachable via the "moonwalker" alias
        *rows("unnamed_type_44", 6, sub_class="moonwalker-adjacent"),
        # unilateral: one hemisphere only
        *rows("LC11", 60, side="right", super_class="visual_projection"),
        # count anomaly: far more than the handle expects
        *rows("BPN", 400),
        # near-miss decoys for a handle that should MISS
        *rows("DNg12", 2, super_class="descending"),
        *rows("DNg14", 2, super_class="descending"),
    ]
    return AnnotationTable(pl.DataFrame(data), sources_used=["synthetic"])


# -- the search ladder ------------------------------------------------------


def test_exact_match_wins(table):
    res = table.resolve(by_name("DNa02"))
    assert res.status is Status.OK
    assert res.match_kind is MatchKind.EXACT
    assert res.count == 2
    assert res.matched_types == ["DNa02"]


def test_prefix_gathers_subtypes(table):
    """KC must collect KCg-m, KCab-c and KCg-d as one population."""
    res = table.resolve(by_name("KC"))
    assert res.match_kind is MatchKind.PREFIX
    assert res.count == 3500
    assert set(res.matched_types) == {"KCg-m", "KCab-c", "KCg-d"}


def test_exact_beats_prefix_for_subtype(table):
    """KCg-d resolves to just itself, not to everything starting with KC."""
    res = table.resolve(by_name("KCg-d"))
    assert res.match_kind is MatchKind.EXACT
    assert res.count == 300


def test_fuzzy_reaches_via_alias(table):
    res = table.resolve(by_name("MDN"))
    assert res.match_kind is MatchKind.EXACT  # exact "MDN" exists, alias unused
    assert res.count == 4


def test_fuzzy_alias_when_no_exact():
    """A handle whose only route is a substring alias still resolves, weakly."""
    df = pl.DataFrame(rows("unnamed_44", 6, sub_class="moonwalker-adjacent"))
    t = AnnotationTable(df, sources_used=["synthetic"])
    h = Handle("MDN", "descending", patterns=("MDN",), aliases=("moonwalker",),
               expect_min=2, expect_max=8)
    res = t.resolve(h)
    assert res.match_kind is MatchKind.FUZZY
    assert res.status is Status.WEAK_MATCH
    assert res.matched_column == "sub_class"


# -- the short-pattern guard ------------------------------------------------


def test_short_pattern_does_not_vacuum(table):
    """ORN_V's pattern list includes bare 'V'. It must not swallow VA2/DA2.

    This is the guard that stops the CO2 glomerulus from matching half the
    olfactory system by substring.
    """
    res = table.resolve(by_name("ORN_V"))
    assert res.matched_types == ["ORN_V"]
    assert res.count == 40


def test_exact_only_reports_missing_rather_than_a_superset():
    """The real vacuum risk: ORN_V is a proper PREFIX of ORN_VA2.

    With the CO2 glomerulus absent, an unguarded prefix fallback would happily
    return ORN_VA2 + ORN_VA1v and call it CO2. exact_only must make it MISS
    instead -- a wrong population that looks right is worse than no population.
    """
    df = pl.DataFrame([*rows("ORN_VA2", 30), *rows("ORN_VA1v", 20)])
    t = AnnotationTable(df, sources_used=["synthetic"])

    guarded = t.resolve(by_name("ORN_V"))
    assert guarded.status is Status.MISSING
    assert guarded.count == 0
    assert any("VA" in c for c in guarded.candidates)

    # ...and prove the guard is what saved us, by removing it.
    unguarded = t.resolve(
        Handle("ORN_V", "olfactory", patterns=("ORN_V", "V"),
               expect_min=10, expect_max=300, exact_only=False)
    )
    assert unguarded.count == 50
    assert set(unguarded.matched_types) == {"ORN_VA2", "ORN_VA1v"}


# -- grading ----------------------------------------------------------------


def test_unilateral_is_flagged(table):
    res = table.resolve(by_name("LC11"))
    assert res.status is Status.UNILATERAL
    assert set(res.side_counts) == {"right"}


def test_count_anomaly_is_flagged(table):
    res = table.resolve(by_name("BPN"))
    assert res.status is Status.COUNT_OFF
    assert res.count == 400  # handle expects <= 20


def test_missing_reports_near_misses(table):
    res = table.resolve(by_name("DNg13"))
    assert res.status is Status.MISSING
    assert res.count == 0
    assert any("DNg1" in c for c in res.candidates)


def test_missing_required_is_fatal(table):
    """A required handle that misses must abort the run; optional must not."""
    missing_required = Handle("NOPE", "descending", patterns=("NOPE",),
                              required=True)
    missing_optional = Handle("NOPE2", "descending", patterns=("NOPE2",),
                              required=False)
    assert table.resolve(missing_required).is_fatal
    assert not table.resolve(missing_optional).is_fatal


# -- side filtering ---------------------------------------------------------


def test_side_filter(table):
    both = table.resolve(by_name("LC4"))
    left = table.resolve(by_name("LC4"), side="left")
    assert left.count == both.count // 2 == 30


# -- loader contract --------------------------------------------------------


def test_missing_csv_gives_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        AnnotationTable.load(tmp_path)
    assert "codex.flywire.ai" in str(exc.value)
    assert "v783" in str(exc.value)


def test_no_type_columns_is_an_error():
    df = pl.DataFrame([{"root_id": 1, "irrelevant": "x"}])
    with pytest.raises(ValueError, match="no usable type columns"):
        AnnotationTable(df, sources_used=["synthetic"])


# -- free-text labels -------------------------------------------------------


def test_free_text_label_hit_is_always_weak():
    """BPN exists only as a community label in v783. A hit there must never be
    graded OK -- free text is a hint, not a typing."""
    df = pl.DataFrame(
        rows("unnamed_9", 6, label="Type 1 BPN (Bolt Protocerebral Neuron)")
    )
    t = AnnotationTable(df, sources_used=["synthetic"])
    res = t.resolve(by_name("BPN"))
    assert res.match_kind is MatchKind.FUZZY
    assert res.matched_column == "label"
    assert res.status is Status.WEAK_MATCH
    assert "free-text" in res.matched_types[0]


def test_labels_excluded_from_near_miss_vocabulary():
    """Near-miss suggestions must not offer whole sentences."""
    df = pl.DataFrame(rows("DNg12", 2, label="a very long community annotation"))
    t = AnnotationTable(df, sources_used=["synthetic"])
    assert all(len(v) < 40 for v in t.vocab)


def test_sibling_patterns_are_unioned():
    """pC1 lists pC1a..pC1e. All five must come back, not just the first.

    Before this fix the resolver returned 2 neurons instead of 10 -- a silent
    80% undercount that graded as a clean match.
    """
    df = pl.DataFrame([*rows("pC1a", 2), *rows("pC1b", 2), *rows("pC1c", 2),
                       *rows("pC1d", 2), *rows("pC1e", 2)])
    t = AnnotationTable(df, sources_used=["synthetic"])
    res = t.resolve(by_name("pC1"))
    assert res.count == 10
    assert set(res.matched_types) == {"pC1a", "pC1b", "pC1c", "pC1d", "pC1e"}


def test_union_does_not_cross_columns():
    """MN9's patterns are alternatives in different columns, not siblings.

    Column priority must still pick one column's worth, so an alternative
    spelling never inflates a population that already resolved.
    """
    df = pl.DataFrame([*rows("MN9", 2),
                       *rows("other", 24, sub_class="proboscis_motor_neuron")])
    t = AnnotationTable(df, sources_used=["synthetic"])
    res = t.resolve(by_name("MN9"))
    assert res.matched_column == "primary_type"
    assert res.count == 2


def test_orn_handles_do_not_collide_with_distal_medulla():
    """DM2 (antennal lobe glomerulus) vs Dm2 (distal medulla) differ only in
    case. An ORN handle must not swallow the medulla population -- this pulled
    1,275 optic lobe neurons into ORN_DM2 before the patterns were tightened.
    """
    df = pl.DataFrame([*rows("ORN_DM2", 54), *rows("Dm2", 1200)])
    t = AnnotationTable(df, sources_used=["synthetic"])
    res = t.resolve(by_name("ORN_DM2"))
    assert res.count == 54
    assert res.matched_types == ["ORN_DM2"]
