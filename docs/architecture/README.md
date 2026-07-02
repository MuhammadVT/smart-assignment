# Architecture diagrams

Place one diagram per workflow here, named to match its workflow folder
(e.g. `slot_recommendation.png`).

For `slot_recommendation`, the current architecture is:

```
START
  -> intake_node                (code — intake + geocode + Top-N nearest routes)
  -> constraint_and_score_node  (code — HARD constraints, then weighted scoring)
  -> route_on_feasibility       (code, conditional)
       NO_OPTIONS  -> escalate_no_feasible_slot     (human input)
       HAS_OPTIONS -> build_recommendation_node     (code — rank + total score)
                        -> total_score_gate           (code, conditional)
                             LOW_SCORE  -> escalate_low_score (human input)
                             HIGH_SCORE -> format_output      (code)
```

Reasoning (the natural-language trace on the final recommendation) is produced
by a pluggable reasoner — LLM by default, with a deterministic fallback.

No image file is included in this package — generate one (e.g. via the
ADK Web UI's graph view, or any diagramming tool) and drop it here as
`slot_recommendation.png` once available.
