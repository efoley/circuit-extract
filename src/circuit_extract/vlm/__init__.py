"""Approach 1: zero-shot VLM extraction.

Multi-step prompting pipeline that asks a multimodal model to:

1. Identify components (refdes, type, value, optional bbox).
2. For each component, enumerate its pins.
3. Trace nets and assign every pin to one.

Each step is its own prompt so that the model never has to do everything in
one shot, and so we can swap providers / inspect intermediate output.
"""

from circuit_extract.vlm.pipeline import VLMExtractionPipeline, extract_netlist

__all__ = ["VLMExtractionPipeline", "extract_netlist"]
