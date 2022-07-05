"""
A simple federated learning server using federated averaging.
"""

import asyncio
from typing import OrderedDict
import torch

from plato.servers import fedavg
from plato.utils import homo_enc


class Server(fedavg.Server):
    """Federated learning server using federated averaging."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.encrypted_model = None
        self.weight_shapes = {}
        self.para_nums = {}
        self.ckks_context = homo_enc.get_ckks_context()

    def init_trainer(self):
        """Setting up the global model to be trained via federated learning."""
        super().init_trainer()
        
        extract_model = self.trainer.model.cpu().state_dict()
        for key in extract_model.keys():
            self.weight_shapes[key] = extract_model[key].size()
            self.para_nums[key] = torch.numel(extract_model[key])
        
        self.encrypted_model = homo_enc.encrypt_weights(extract_model, True,
                                                        self.ckks_context, self.para_nums,
                                                        encrypt_ratio = 0)
            

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        self.encrypted_model = await self.federated_averaging_he(updates)

        # Decrypt model weights for test accuracy
        decrypted_weights = homo_enc.decrypt_weights(self.encrypted_model, 
                                                     self.weight_shapes, self.para_nums)
        self.algorithm.load_weights(decrypted_weights)

    async def federated_averaging_he(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = [payload for (__, __, payload, __) in updates]

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.size())
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            __, report, __, __ = updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    def customize_server_payload(self, payload):
        """ Customize the server payload before sending to the client. """        
        return self.encrypted_model
