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

        self.model = None
        self.context = homo_enc.get_ckks_context()

    def process(self, data: Any) -> Any:
        self.model = self.trainer.model
        output = self.model.cpu().state_dict()
        logging.info(
            "[Client #%d] Applying a processor that encrypts the model.",
            self.client_id)

        key_list = output.keys()
        para_num = 0
        for tensor_name in key_list:
            para_num = para_num + torch.numel(output[tensor_name])
            #print(para_num)
        print(para_num)

        for tensor_name in key_list:
            tensor_to_encrypt = output[tensor_name]

            encrypted_tensor = torch.flatten(tensor_to_encrypt)
            encrypted_tensor = ts.ckks_vector(self.context, encrypted_tensor)
            encrypted_tensor = encrypted_tensor.serialize()

            output[tensor_name] = encrypted_tensor

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        return layer