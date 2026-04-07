"""Approach 2 (planned): component detection with YOLO.

This module will host:
- YOLO model loading / inference wrappers
- Dataset adapters for public schematic datasets (CGHD, FCDB, ...)
- Crop extraction utilities that feed each detected component into a VLM
  for fine-grained value/pin parsing.

Nothing implemented yet — start with the VLM zero-shot pipeline first.
"""
