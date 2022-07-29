"""
A processor that encrypts model weights tensors.
"""

from turtle import st
import logging
import time
from typing import Any

import torch
import tenseal as ts
from plato.config import Config

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

        self.encrypt_mask = None

    def process(self, data: Any) -> Any:
        logging.info(
            "[Client #%d] Applying a processor that encrypts the model.",
            self.client_id)
        start_time = time.time()
        encrypted_weights = homo_enc.encrypt_weights(data, 
                                    serialize = True,
                                    context = self.context,
                                    para_nums = self.para_nums,
                                    encrypt_ratio = 0.05,
                                    enc_mask = self.encrypt_mask)

        logging.info(f"Encryption Time: {time.time() - start_time}")

        homo_enc.update_est(Config(), self.client_id, encrypted_weights)
        return encrypted_weights

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        return layer