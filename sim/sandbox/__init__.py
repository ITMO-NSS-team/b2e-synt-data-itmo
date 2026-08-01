"""Skill execution sandbox — the two-container spool split from G1.

See ``docs/skill-execution-threat-model.md`` §5. Layout:

``runner``   executes one skill in a hardened child process, JSON in / JSON out
``worker``   claims jobs from the spool, enforces the wall clock, never networked
``gateway``  submits jobs and reads results; the only side with network

The split exists so the executing process can run with ``network_mode: none``
while still being reachable, without giving any service a Docker socket.
"""
