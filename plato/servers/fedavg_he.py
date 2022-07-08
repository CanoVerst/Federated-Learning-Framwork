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

        self.final_mask = None
        self.last_selected_clients = []


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
            

    def mask_consensus(self, updates):
        proposals = [mask for (__, __, mask, __) in updates]
        mask_size = len(proposals[0])
        interleaved_indices = torch.zeros((sum([len(x) for x in proposals])))
        for i in range(len(proposals)):
            interleaved_indices[i::len(proposals)] = proposals[i]
        
        _, indices = interleaved_indices.unique(sorted=False, return_inverse = True)
        
        self.final_mask = interleaved_indices[indices.unique()[:mask_size]].clone().detach()
        self.final_mask = self.final_mask.int().long()

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        if self.current_round % 2 != 0:
            self.mask_consensus(updates)
        else:
            self.encrypted_model = await self.federated_averaging_he(updates)

            # Decrypt model weights for test accuracy
            decrypted_weights = homo_enc.decrypt_weights(self.encrypted_model, 
                                                         self.weight_shapes, self.para_nums)
            self.algorithm.load_weights(decrypted_weights)

            self.encrypted_model["encrypted_weights"] = self.encrypted_model["encrypted_weights"].serialize()

    async def federated_averaging_he(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = [homo_enc.deserialize_weights(payload, self.ckks_context)  
                               for (__, __, payload, __) in updates]

        unencrypted_weights = [x["unencrypted_weights"] for x in weights_received]
        encrypted_weights = [x["encrypted_weights"] for x in weights_received]

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates])

        # Perform weighted averaging
        unencrypted_avg_update = self.trainer.zeros(unencrypted_weights[0].size())
        encrypted_avg_update = self.trainer.zeros(encrypted_weights[0].size())

        for i, (unenc_w, enc_w) in enumerate(zip(unencrypted_weights, encrypted_weights)):
            __, report, __, __ = updates[i]
            num_samples = report.num_samples

            unencrypted_avg_update += unenc_w * (num_samples / self.total_samples)
            encrypted_avg_update += enc_w * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        avg_result = OrderedDict()
        avg_result["unencrypted_weights"] = unencrypted_avg_update
        avg_result["encrypted_weights"] = encrypted_avg_update
        avg_result["encrypt_indices"] = self.final_mask
        return avg_result

    def customize_server_payload(self, payload):
        """ Customize the server payload before sending to the client. """
        if self.current_round % 2 != 0:
            return self.encrypted_model
        else:
            return self.final_mask

    def choose_clients(self, clients_pool, clients_count):
        if self.current_round % 2 != 0:
            self.last_selected_clients = super().choose_clients(clients_pool, clients_count)
            return self.last_selected_clients
        else:
            return self.last_selected_clients
    
    async def process_reports(self):
        if self.current_round % 2 != 0:
            await super().process_reports()
        else:
            await self.aggregate_weights(self.updates)

        