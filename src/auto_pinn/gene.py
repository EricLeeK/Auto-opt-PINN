"""Gene representation for neural architecture search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class LayerType(str, Enum):
    DNN = "DNN"
    KAN = "KAN"
    ATTENTION = "Attention"


@dataclass
class LayerGene:
    """Description of an individual layer in the hybrid PINN."""

    layer_type: LayerType
    params: Dict[str, int]

    def copy(self) -> "LayerGene":
        return LayerGene(layer_type=self.layer_type, params=dict(self.params))


Gene = List[LayerGene]
