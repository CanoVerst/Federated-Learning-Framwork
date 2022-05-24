"""
A processor that encrypts model weights tensors.
"""

from turtle import st
import numpy as np
import logging
from typing import Any
from rsa import encrypt
from timeit import default_timer as timer

import torch
import torch.nn.utils.prune as prune
import tenseal as ts

from plato.config import Config
from plato.processors import model
from plato.utils import unary_encoding, homo_enc

class Processor(model.Processor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = None
        self.context = homo_enc.get_ckks_context()

    def process(self, data: Any) -> Any:
        self.model = self.trainer.model
        output = self.model.cpu().state_dict()

        logging.info(
            "[Client #%d] Applying a processor that encrypts the model.",
            self.client_id)

        key_list = output.keys()
        elements_to_add = {}

        for tensor_name in key_list:
            shape = tensor_name + ".shape"
            tensor_to_encrypt = output[tensor_name]
            elements_to_add[shape] = tensor_to_encrypt.size()

            encrypted_tensor = torch.flatten(tensor_to_encrypt)
            print(encrypted_tensor.size())

            start_enc = timer()
            encrypted_tensor = ts.ckks_tensor(self.context, encrypted_tensor)
            after_enc = timer()
            print(after_enc-start_enc)

            start_ser = timer()
            encrypted_tensor = encrypted_tensor.serialize()
            after_ser = timer()
            print(after_ser-start_ser)

            output[tensor_name] = encrypted_tensor

        for key in elements_to_add.keys():
            output[key] = elements_to_add[key]

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        return layer