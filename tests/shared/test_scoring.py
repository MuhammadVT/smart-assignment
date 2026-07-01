"""
Unit tests for the weighted scoring layer (shared/scoring.py).
"""

from __future__ import annotations

from smart_assignment.shared.constraints import build_context
from smart_assignment.shared.scoring import score_candidate


def test_score_is_normalized(sample_customer, open_route, config):
    ctx = build_context(sample_customer, open_route)
    breakdown, total = score_candidate(sample_customer, open_route, ctx, config)
    assert 0.0 <= total <= 1.0
    assert {f.name for f in breakdown} == {
        "geographic_clustering",
        "capacity_buffer",
        "window_match",
    }


def test_more_capacity_headroom_scores_higher(sample_customer, open_route, config):
    ctx = build_context(sample_customer, open_route)
    _, base = score_candidate(sample_customer, open_route, ctx, config)

    # Same route but much more committed volume -> less headroom -> lower score.
    open_route.committed_stops[0].case_volume = 800
    ctx2 = build_context(sample_customer, open_route)
    _, tighter = score_candidate(sample_customer, open_route, ctx2, config)

    assert tighter < base


def test_factor_weights_respect_config(sample_customer, open_route, config):
    ctx = build_context(sample_customer, open_route)
    breakdown, _ = score_candidate(sample_customer, open_route, ctx, config)
    weights = {f.name: f.weight for f in breakdown}
    # Priority order from spec: clustering > capacity buffer > window match.
    assert weights["geographic_clustering"] > weights["capacity_buffer"]
    assert weights["capacity_buffer"] > weights["window_match"]
