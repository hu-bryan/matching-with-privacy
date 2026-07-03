"""Local perturbation mechanism (planar Laplace + Hungarian)."""

from __future__ import annotations

import numpy as np

from private_matching.instances import Instance
from private_matching.matching import match_hungarian, planar_laplace
from private_matching.mechanisms.base import Mechanism, register_mechanism


@register_mechanism("local")
class LocalMechanism(Mechanism):
    """Perturb every private point with planar Laplace, then Hungarian on noisy costs."""

    def match(
        self, inst: Instance, epsilon: float, rng: np.random.Generator
    ) -> np.ndarray:
        R_tilde = planar_laplace(inst.R, epsilon, rng)
        _, sigma = match_hungarian(inst.Q, R_tilde)
        return sigma
