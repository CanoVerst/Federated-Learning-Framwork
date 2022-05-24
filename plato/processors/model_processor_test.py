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
from plato.utils import unary_encoding, homo_enc


class Processor(model.Processor):
    """
    An processor that flatten, encrypt, and serialize the weight tensors of the model.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = None
        self.context = homo_enc.get_ckks_context()

    def process(self, data: Any) -> Any:

        self.model = self.trainer.model

        output = self.model.cpu().state_dict()
        #output = super().process(data)

        logging.info(
            "[Client #%d] Applying a processor that flattens, encrypts, and serializes tensors of the model.",
            self.client_id)
        
        print("Start printing info of the model")
        print("Printing the keys")
        print(output.keys())
        print(type(output['conv1.weight']))
        print(type(output['conv1.bias']))
        #print(output['conv3.weight'])
        print(output['conv1.weight'].size())
        print(output['conv2.weight'].size())
        print(output['conv3.weight'].size())
        print(output['fc4.weight'].size())
        print(output['fc5.weight'].size())



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
        For now, only conv1.weight tensor is modified.
        
        #print(output['conv1.weight'])
        #print(output['conv1.weight'].size())
        output['conv1.weight.shape'] = output['conv1.weight'].size()
        output['conv1.weight'] = torch.flatten(output['conv1.weight'])
        output['conv1.weight'] = ts.ckks_tensor(self.context,output['conv1.weight'])
        output['conv1.weight'] = output['conv1.weight'].serialize()
        """
        
        #print("Printing the shape of the first value")
        #print(str(output['conv1.bias'].shape))
        #print(str(output))
        print("Stop printing info of the model")
        
        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        
        """print(type(layer))
        print(layer)"""
        return layer
