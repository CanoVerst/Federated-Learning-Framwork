"""
A processor that encrypts model weights tensors.
"""

from turtle import st
import logging
from typing import Any

import torch
import tenseal as ts

from plato.processors import model
from plato.utils import homo_enc

class Processor(model.Processor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context = homo_enc.get_ckks_context()

        para_nums = {}
        extract_model = self.trainer.model.cpu().state_dict()
        for key in extract_model.keys():
            para_nums[key] = torch.numel(extract_model[key])
        self.para_nums = para_nums

    def process(self, data: Any) -> Any:
        logging.info(
            "[Client #%d] Applying a processor that encrypts the model.",
            self.client_id)
        encrypted_weights = homo_enc.encrypt_weights(data, 
                                    serialize = True,
                                    context = self.context,
                                    para_nums = self.para_nums,
                                    encrypt_ratio = 0.05)
        return encrypted_weights

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        return layer