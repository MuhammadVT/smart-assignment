"""
Unit tests for the weighted scoring layer (shared/scoring.py).
"""

from __future__ import annotations

from smart_assignment.shared.constraints import build_context
from smart_assignment.shared.scoring import capacity_buffer, score_candidate


def test_score_is_normalized(sample_customer, open_route, config):
    ctx = build_context(sample_customer, open_route)
    breakdown, total = score_candidate(sample_customer, open_route, ctx, config)
    assert 0.0 <= total <= 1.0
    assert {f.name for f in breakdown} == {
        "geographic_clustering",
        "capacity_buffer",
        "window_match",
    }


def test_capacity_buffer_flat_within_safe_zone(sample_customer, open_route, config):
    # Below the safety margin (default 15pp under the 90% ceiling, i.e. <=75%
    # utilized), the score is flat at 1.0 -- an almost-empty route and a
    # fairly busy-but-still-safe route score identically. This is the bias
    # fix: extra headroom beyond "safe" no longer buys extra score.
    open_route.committed_stops[0].case_volume = 10
    ctx_empty = build_context(sample_customer, open_route)
    f_empty = capacity_buffer(sample_customer, open_route, ctx_empty, config)

    open_route.committed_stops[0].case_volume = 600
    ctx_busier = build_context(sample_customer, open_route)
    f_busier = capacity_buffer(sample_customer, open_route, ctx_busier, config)

    assert ctx_busier.utilization_after > ctx_empty.utilization_after  # sanity: busier is busier
    assert f_empty.value == 1.0
    assert f_busier.value == 1.0


def test_capacity_buffer_decays_between_safe_line_and_ceiling(sample_customer, open_route, config):
    # 700 committed + 90 new = 79% utilized, inside the 75-90% decay band.
    open_route.committed_stops[0].case_volume = 700
    ctx = build_context(sample_customer, open_route)
    f = capacity_buffer(sample_customer, open_route, ctx, config)
    assert 0.0 < f.value < 1.0


def test_capacity_buffer_reaches_zero_at_the_ceiling(sample_customer, open_route, config):
    # 810 committed + 90 new = exactly 90% utilized -- right at the hard ceiling.
    open_route.committed_stops[0].case_volume = 810
    ctx = build_context(sample_customer, open_route)
    f = capacity_buffer(sample_customer, open_route, ctx, config)
    assert f.value == 0.0


def test_factor_weights_respect_config(sample_customer, open_route, config):
    ctx = build_context(sample_customer, open_route)
    breakdown, _ = score_candidate(sample_customer, open_route, ctx, config)
    weights = {f.name: f.weight for f in breakdown}
    # Priority order from spec: clustering > capacity buffer > window match.
    assert weights["geographic_clustering"] > weights["capacity_buffer"]
    assert weights["capacity_buffer"] > weights["window_match"]
