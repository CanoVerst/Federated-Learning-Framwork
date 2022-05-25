"""
A processor that decrypts model weights tensors.
"""

import logging
from termios import EXTA
from typing import Any
import pickle
import tenseal as ts
import torch

from plato.processors import model
from plato.utils import homo_enc

class Processor(model.Processor):
    """
    A processor that decrypts model tensors
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context = homo_enc.get_ckks_context()

    def process(self, data: Any) -> Any:
        extract_model = self.trainer.model.cpu().state_dict()
        output = data
        new_key_list = output.keys()
        elements_to_delete = []

        for key in new_key_list:
            if ".shape" not in key:
                vector_to_build = output[key]
                shape = key + ".shape"
                elements_to_delete.append(shape)

                rebuilt_vector = ts.lazy_ckks_vector_from(vector_to_build)
                rebuilt_vector.link_context(self.context)
                rebuilt_tensor = torch.tensor(rebuilt_vector.decrypt())
                rebuilt_tensor = rebuilt_tensor.reshape(output[shape])
                output[key] = rebuilt_tensor

        for key in elements_to_delete:
            output.pop(key)
        
        return output
        

    def _process_layer(self, layer: Any) -> Any:
        """
        No operation
        """
