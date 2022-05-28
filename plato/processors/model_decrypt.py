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
        extract_model = self.trainer.model.cpu().state_dict()
        for key in extract_model.keys():
            weight_shapes[key] = extract_model[key].size()
        self.weight_shapes = weight_shapes

    def process(self, data: Any) -> Any:
        output = data
        key_list = output.keys()

        for key in key_list:
            vector_to_build = output[key]
            rebuilt_tensor_shape = self.weight_shapes[key]

            rebuilt_vector = ts.lazy_ckks_vector_from(vector_to_build)
            rebuilt_vector.link_context(self.context)
            if self.client_id:
                rebuilt_tensor = torch.tensor(rebuilt_vector.decrypt())
                rebuilt_tensor = rebuilt_tensor.reshape(rebuilt_tensor_shape)
                output[key] = rebuilt_tensor
            else:
                output[key] = rebuilt_vector
        return output

    def _process_layer(self, layer: Any) -> Any:
        """
        No operation
        """
