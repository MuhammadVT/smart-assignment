"""
Container for individual agentic workflows. Each subpackage owns its own
internal architecture (graph, sequential pipeline, multi-agent hierarchy,
etc) and exposes a `root_agent`. This package itself makes no assumption
about which orchestration pattern any workflow uses.
"""
