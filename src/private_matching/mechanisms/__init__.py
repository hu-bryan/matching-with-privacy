"""Mechanism registry — import all mechanisms to populate MECHANISMS."""

from private_matching.mechanisms.auction import AuctionMechanism
from private_matching.mechanisms.base import MECHANISMS, Mechanism, register_mechanism
from private_matching.mechanisms.dual_sinkhorn import DualSinkhornMechanism
from private_matching.mechanisms.local import LocalMechanism

__all__ = [
    "MECHANISMS",
    "Mechanism",
    "register_mechanism",
    "LocalMechanism",
    "AuctionMechanism",
    "DualSinkhornMechanism",
]
