"""
Implement a Processor for checking the payload received by the server.
"""

import logging
from typing import Any
import pickle
import tenseal as ts
import torch

from plato.processors import model
from plato.utils import homo_enc


class Processor(model.Processor):
    """
    Implement a Processor checking the payload received by the server
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context = homo_enc.get_ckks_context()

    def process(self, data: Any) -> Any:
        """
        Only conv1.weight tensor is operated for now.
        """
        output = data

        """
        print(type(data))
        rebuilt_vector = ts.lazy_ckks_tensor_from(output['conv1.weight'])
        rebuilt_vector.link_context(self.context)
        rebuilt_tensor = torch.tensor(rebuilt_vector.decrypt().tolist())
        #print(rebuilt_tensor)
        rebuilt_tensor=rebuilt_tensor.reshape(output['conv1.weight.shape'])
        #print(rebuilt_tensor)
        output['conv1.weight'] = rebuilt_tensor
        output.pop('conv1.weight.shape')
        #print(data[str(list(data)[0])])
        return output
        """
        print(output['conv1.weight'])

        return output


    def _process_layer(self, layer: Any) -> Any:
        """
        No operation
        """