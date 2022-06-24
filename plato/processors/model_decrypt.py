"""
A processor that decrypts model weights tensors.
"""

from termios import EXTA
from typing import Any
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
        weight_shapes = {}
        para_nums = {}
        extract_model = self.trainer.model.cpu().state_dict()
        
        for key in extract_model.keys():
            weight_shapes[key] = extract_model[key].size()
            para_nums[key] = torch.numel(extract_model[key])
        self.weight_shapes = weight_shapes
        self.para_nums = para_nums

    def process(self, data: Any) -> Any:
        deserialized_weights = homo_enc.deserialize_weights(data, self.context)
        if self.client_id:
            output = homo_enc.decrypt_weights(deserialized_weights, self.weight_shapes, self.para_nums)
        else:
            output = deserialized_weights
        return output

    def _process_layer(self, layer: Any) -> Any:
        """
        No operation
        """
