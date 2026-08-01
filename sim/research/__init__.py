"""C4 — read/write API for researchers.

A thin layer over Phoenix and the run store, deliberately not a second tracing
system. Spans live in Phoenix; this service queries them, joins them to the run
store, scores them against the oracle, and exports.
"""
