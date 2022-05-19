"""
Implement a Processor for checking the payload received by the server.
"""

import logging
from typing import Any
import pickle

from plato.processors import model

class Processor(model.Processor):
    """
    Implement a Processor checking the payload received by the server
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any) -> Any:
        output = data
        print(type(data))
        #print(data[str(list(data)[0])])
        return output


    def _process_layer(self, layer: Any) -> Any:
        """
        No operation
        """