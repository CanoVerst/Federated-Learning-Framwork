"""
An arbitrary processor for test only.
"""

import numpy as np
import logging
from typing import Any

import torch
import torch.nn.utils.prune as prune
import tenseal as ts

from plato.config import Config
from plato.processors import model
from plato.utils import unary_encoding


class Processor(model.Processor):
    """
    An processor that flatten, encrypt, and serialize the weight tensors of the model.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = None

    def process(self, data: Any) -> Any:

        self.model = self.trainer.model

        output = self.model.cpu().state_dict()
        #output = super().process(data)

        logging.info(
            "[Client #%d] Applying an arbitrary processor for test only.",
            self.client_id)
        
        
        ckks_text = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        ckks_text.global_scale = pow(2,40)
        ckks_text.generate_galois_keys()

        print("Start printing info of the model")
        print("Printing the keys")
        print(output.keys())
        print("Printing the first key")
        print(str(list(output)[0]))
        #Flatten the frist value of the output
        #output['conv1.weight'] = torch.flatten(output['conv1.weight'])
        #print(output['conv1.weight'])
        #print(torch.flatten(output['conv1.weight']))
        """
        for i in output.keys():
            if 'weight' in i:
                
                #Flatten the tensor
                output[i] = torch.flatten(output[i])
                #Encrypt the flattened tensor
                output[i] = ts.ckks_tensor(ckks_text,output[i])
                output[i] = output[i].serialize()
                #print(output[i])
        """
        
        """
        For now, only the first weight tensor is modified.
        """
        output['conv1.weight'] = torch.flatten(output['conv1.weight'])
        output['conv1.weight'] = ts.ckks_tensor(ckks_text,output['conv1.weight'])
        output['conv1.weight'] = output['conv1.weight'].serialize()
        
        #print("Printing the shape of the first value")
        #print(str(output['conv1.bias'].shape))
        #print(str(output))
        print("Stop printing info of the model")
        
        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        
        """print(type(layer))
        print(layer)"""
        return layer
