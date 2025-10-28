"""Gene representation for neural architecture search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


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

GeneSignature = Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...]


def gene_signature(gene: Gene) -> GeneSignature:
    """Hashable signature capturing layer types and sorted integer params."""

    signature_layers = []
    for layer in gene:
        sorted_params = tuple(sorted((key, int(value)) for key, value in layer.params.items()))
        signature_layers.append((layer.layer_type.value, sorted_params))
    return tuple(signature_layers)
