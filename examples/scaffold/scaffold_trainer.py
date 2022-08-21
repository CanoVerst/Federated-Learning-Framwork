"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from collections import OrderedDict

import logging
import pickle
import torch

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """The federated learning trainer for the SCAFFOLD client."""

    def __init__(self, model=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model)

        self.server_control_variate = None
        self.client_control_variate = None
        self.global_model_weights = None
        self.client_control_variate_path = None
        self.extra_payload_path = None
        self.param_groups = None

    def get_optimizer(self, model):
        """Gets the parameter groups from the optimizer"""
        optimizer = super().get_optimizer(model)
        self.param_groups = optimizer.param_groups
        return optimizer

    def train_run_start(self, config):
        """Initializes the client control variate to 0 if the client
        is participating for the first time.
        """
        if self.client_control_variate is None:
            self.client_control_variate = {}
            for variate in self.server_control_variate:
                self.client_control_variate[variate] = torch.zeros(
                    self.server_control_variate[variate].shape
                )

    def train_step_end(self, config, batch=None, loss=None):
        """Modifies the weights based on the server and client control variates."""
        for group in self.param_groups:
            learning_rate = -group["lr"]
            counter = 0
            for name in self.server_control_variate:
                if "weight" in name or "bias" in name:
                    server_control_variate = self.server_control_variate[name].to(
                        self.device
                    )
                    param = group["params"][counter]
                    if self.client_control_variate:
                        param.data.add_(
                            torch.sub(
                                server_control_variate,
                                self.client_control_variate[name].to(self.device),
                            ),
                            alpha=learning_rate,
                        )
                    else:
                        param.data.add_(server_control_variate, alpha=learning_rate)
                    counter += 1

    def train_run_end(self, config):
        """Compute deltas of this client's control variate"""

        new_client_control_variate = OrderedDict()
        control_variate_deltas = OrderedDict()
        if self.client_control_variate:
            for name, previous_weight in self.global_model_weights.items():
                new_client_control_variate[name] = torch.sub(
                    self.client_control_variate[name], self.server_control_variate[name]
                )
                new_client_control_variate[name].add_(
                    torch.sub(previous_weight, self.model.state_dict()[name]),
                    alpha=1 / Config().trainer.epochs,
                )

                control_variate_deltas[name] = torch.sub(
                    new_client_control_variate[name], self.client_control_variate[name]
                )
        else:
            for name, previous_weight in self.global_model_weights.items():
                new_client_control_variate[name] = -self.server_control_variate[name]
                new_client_control_variate[name].add_(
                    torch.sub(previous_weight, self.model.state_dict()[name]),
                    alpha=1 / Config().trainer.epochs,
                )

                control_variate_deltas[name] = new_client_control_variate[name]

        # Update client control variate
        self.client_control_variate = new_client_control_variate

        # Save client control variate
        logging.info(
            "[Client #%d] Saving the control variate to %s.",
            self.client_id,
            self.client_control_variate_path,
        )
        with open(self.client_control_variate_path, "wb") as path:
            pickle.dump(self.client_control_variate, path)

        logging.info(
            "[Client #%d] Control variate saved to %s.",
            self.client_id,
            self.client_control_variate_path,
        )
