"""Difficulty knobs recorded per instance, for grouping/plotting results.

Difficulty is set at *generation* time (see `instances.gen_block_alpha`):

- ``alpha``  — near-optimum ruggedness. Small alpha packs many near-optimal
  competitors just above OPT (hard to single out the winner); large alpha isolates
  the optimum. It is a generator *input*, not something we fit here.
- ``stakes_S`` — sets RAND/OPT, i.e. how much an average mistake costs (how much the
  instance actually tests).

Natural families (``uniform``, ``two_gaussian``, ...) carry neither knob and are
recorded as NaN. Everything else we need for plots — OPT, RAND, the optimality margin
``mu``, and the competitive / regret ratios — already lives in ``metrics.py``.
"""

from __future__ import annotations

from typing import Any

from private_matching.instances import Instance


def instance_difficulty(inst: Instance) -> dict[str, Any]:
    """Difficulty columns to attach to each result row (from generation params)."""
    return {
        "alpha": inst.params.get("alpha", float("nan")),
        "stakes_S": inst.params.get("stakes_S", float("nan")),
    }
