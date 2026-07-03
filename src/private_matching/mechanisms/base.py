"""Mechanism ABC and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from private_matching.instances import Instance

MECHANISMS: dict[str, type["Mechanism"]] = {}


def register_mechanism(name: str):
    """Decorator to register a mechanism class."""

    def decorator(cls: type[Mechanism]):
        MECHANISMS[name] = cls
        cls.name = name
        return cls

    return decorator


class Mechanism(ABC):
    """Randomized matching mechanism: returns a permutation sigma."""

    name: ClassVar[str]

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams

    @abstractmethod
    def match(
        self, inst: Instance, epsilon: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Return sigma with sigma[i] = matched private index."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.hyperparams})"
