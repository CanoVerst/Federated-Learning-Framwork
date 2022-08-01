"""
Implementation of the fedrep clients.

"""

import logging

from plato.clients import pers_simple
from plato.config import Config


class Client(pers_simple.Client):
    """A personalized federated learning client with self-supervised support
        for the contrastive adaptation method."""

    def load_personalized_model(self):
        """ Initial the personalized model with the global model. """

        model_name = Config().trainer.model_name
        global_model_name = Config().trainer.global_model_name
        personalized_model_name = Config().trainer.personalized_model_name
        logging.info(
            "[Client #%d]. copy [%s] (after completion) with the downloaded global model [%s] as its personalized model [%s].",
            self.client_id, model_name, global_model_name,
            personalized_model_name)
        # should directly copy from the downloaded model
        self.trainer.personalized_model.load_state_dict(
            self.trainer.model.state_dict())
