"""
An arbitrary processor for test only.
"""


import logging
from typing import Any

import torch
import torch.nn.utils.prune as prune

from plato.config import Config
from plato.processors import model
from plato.utils import unary_encoding

print("A processor for test only is running.")

class Processor(model.Processor):
    """An arbitrary processor for test only"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any) -> Any:

        output = super().process(data)

        logging.info(
            "[Client #%d] Arbitrary processor for test only.",
            self.client_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:

        return layer
