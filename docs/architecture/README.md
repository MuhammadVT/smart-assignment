# Architecture diagrams

Place one diagram per workflow here, named to match its workflow folder
(e.g. `slot_recommendation.png`).

For `slot_recommendation`, the current architecture is:

```
START
  -> geocode_and_cluster_customer        (code)
  -> fetch_candidate_slots_node          (code — calls route capacity system)
  -> filter_feasible_slots_node          (code — HARD constraints only)
  -> route_on_feasibility                (code, conditional)
       NO_OPTIONS  -> escalate_no_feasible_slot   (human input)
       HAS_OPTIONS -> recommend_slot_agent        (LLM — ranks feasible options)
                        -> confidence_gate          (code, conditional)
                             LOW_CONFIDENCE  -> escalate_low_confidence (human input)
                             HIGH_CONFIDENCE -> format_output            (code)
```

No image file is included in this package — generate one (e.g. via the
ADK Web UI's graph view, or any diagramming tool) and drop it here as
`slot_recommendation.png` once available.
