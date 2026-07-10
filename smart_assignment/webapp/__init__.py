"""
Live web app for the Smart Assignment workflow.

A small FastAPI application that serves a chat interface and visualizes the
agent's five-step workflow on *live* input — the same step-by-step animation
the published GitHub Pages **Simulator** shows, but driven by the real pipeline
rather than pre-computed sample runs.

The heavy lifting is reused, not re-implemented:

* the workflow itself is ``smart_assignment.pipeline.run_slot_recommendation``
  (the same function the offline demo and the page generator call), and
* the visualization payload is
  ``smart_assignment.reporting.page.build_workflow_payload`` — so the live UI can
  never drift from the published examples.

Phase 1 (this module) runs fully offline with the deterministic reasoner — no
API key required. See ``app.py`` for how to run it.
"""
