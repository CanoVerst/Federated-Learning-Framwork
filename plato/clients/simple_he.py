"""
A basic federated learning client with homomorphic encryption support
"""

import logging
import time
from dataclasses import dataclass

from plato.algorithms import registry as algorithms_registry
from plato.clients import simple
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry
from plato.utils import fonts

class Client(simple.Client):
    """A basic federated learning client who sends simple weight updates."""

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model = model, datasource = datasource, algorithm = algorithm, trainer = trainer)
        