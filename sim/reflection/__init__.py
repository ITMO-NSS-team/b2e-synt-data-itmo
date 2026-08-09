"""Nightly reflection and the long-term memory it writes.

Memory is a registry artefact (``kind="memory"``), not a skill and not a prompt
edit — ``sim/skills.py``'s human approval gate is untouched, and there is no
second place an agent's behaviour can change without a fingerprinted config
version. Everything before the single LLM call in the reflection procedure is
deterministic code; the model proposes lesson text but never owns an evidence
counter.
"""
from __future__ import annotations
