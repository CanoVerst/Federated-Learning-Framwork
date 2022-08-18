"""
The training loop that takes place on clients of FedSCR.
"""
from collections import OrderedDict

import copy
import pickle
import os
import logging
import torch
import numpy as np

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer used by the client."""

    def __init__(self, model=None):
        """Initialize the trainer with the provided model."""
        super().__init__(model=model)

        # The threshold for determining whether or not an update is significant
        self.update_threshold = (
            Config().clients.update_threshold
            if hasattr(Config().clients, "update_threshold")
            else 0.3
        )

        # The overall weight updates applied to the model in a single round
        self.total_grad = OrderedDict()

        # The accumulated gradients for a client throughout the FL session
        self.acc_grads = []

        # Should the clients use the adaptive algorithm?
        self.use_adaptive = bool(
            hasattr(Config().clients, "adaptive") and Config().clients.adaptive
        )
        self.avg_update = None
        self.div_from_global = None
        self.orig_weights = None

        # Get the names of the convolutional layers
        self.conv_layer_names = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                tensor = module.weight.data
                conv_layer_name = [
                    name
                    for name in self.model.state_dict()
                    if torch.equal(tensor, self.model.state_dict()[name])
                ]
                self.conv_layer_names.append(conv_layer_name[0])

    def prune_update(self):
        """Prune the weight update by setting some parameters in update to 0."""
        self.load_acc_grads()

        conv_updates = OrderedDict()
        i = 0
        for orig_module, trained_module in zip(
            self.orig_weights.modules(), self.model.modules()
        ):
            if isinstance(
                trained_module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
            ):
                orig_tensor = orig_module.weight.data.cpu().numpy()
                trained_tensor = trained_module.weight.data.cpu().numpy()
                delta = trained_tensor - orig_tensor + self.acc_grads[i]
                orig_delta = copy.deepcopy(delta)

                aggregated_channels = self.aggregate_channels(delta)
                aggregated_filters = self.aggregate_filters(delta)

                delta = self.prune_channels(aggregated_channels, delta)
                delta = self.prune_filters(aggregated_filters, delta)

                self.acc_grads[i] = orig_delta - delta
                conv_updates[self.conv_layer_names[i]] = delta
                i += 1

        for orig_key, trained_key in zip(
            self.orig_weights.state_dict(), self.model.state_dict()
        ):
            if not orig_key in conv_updates:
                orig_tensor = self.orig_weights.state_dict()[orig_key]
                trained_tensor = self.model.state_dict()[trained_key]
                delta = trained_tensor - orig_tensor
                self.total_grad[orig_key] = delta
            else:
                self.total_grad[orig_key] = torch.from_numpy(conv_updates[orig_key])

        self.save_acc_grads()

    def aggregate_channels(self, delta):
        """Aggregate the sum of a certain channel from all filters."""
        num_channels = delta.shape[1]
        num_filters = delta.shape[0]
        aggregated_channels = [None] * num_channels

        for channel_index in range(num_channels):
            tensor_sum = 0
            for filters in range(num_filters):
                tensor_sum += np.abs(delta[filters, channel_index, :, :])
            aggregated_channels[channel_index] = tensor_sum

        for index, __ in enumerate(aggregated_channels):
            aggregated_channels[index] = np.sum(aggregated_channels[index])

        return aggregated_channels

    def aggregate_filters(self, delta):
        """Aggregate the sum of all channels from a single filter."""
        num_channels = delta.shape[1]
        num_filters = delta.shape[0]
        aggregated_filters = [None] * num_filters

        for filter_index in range(num_filters):
            tensor_sum = 0
            for channel in range(num_channels):
                tensor_sum += np.abs(delta[filter_index, channel, :, :])
            aggregated_filters[filter_index] = tensor_sum

        for index, __ in enumerate(aggregated_filters):
            aggregated_filters[index] = np.sum(aggregated_filters[index])

        return aggregated_filters

    def prune_channels(self, aggregated_channels, delta):
        """Prune the channels in update that lie below the FedSCR threshold."""
        for i, norm in enumerate(aggregated_channels):
            if norm < self.update_threshold:
                delta[:, i, :, :] = 0

        return delta

    def prune_filters(self, aggregated_filters, delta):
        """Prune the filters in update that lie below the FedSCR threshold."""
        for i, norm in enumerate(aggregated_filters):
            if norm < self.update_threshold:
                delta[i, :, :, :] = 0

        return delta

    def save_acc_grads(self):
        """Save the accumulated client gradients for the next communication round."""
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]

        acc_grad_path = (
            f"{checkpoint_path}/{model_name}_client{self.client_id}_grad.pth"
        )
        with open(acc_grad_path, "wb") as payload_file:
            pickle.dump(self.acc_grads, payload_file)

    def load_acc_grads(self):
        """Load the accumulated gradients from a previous communication round."""
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        grad_path = f"{checkpoint_path}/{model_name}_client{self.client_id}_grad.pth"
        if os.path.exists(grad_path):
            with open(grad_path, "rb") as payload_file:
                self.acc_grads = pickle.load(payload_file)
        else:
            count = 0
            for module in self.model.modules():
                if isinstance(
                    module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
                ):
                    count += 1
            self.acc_grads = [0] * count

    def compute_pruned_amount(self):
        """Compute the pruned percentage of the entire model."""
        nonzero = 0
        total = 0
        for key in sorted(self.total_grad.keys()):
            tensor = self.total_grad[key].cpu()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params

        return 100 * (total - nonzero) / total

    def train_run_start(self, config):
        """Method called at the start of training run."""
        self.total_grad = OrderedDict()
        self.orig_weights = copy.deepcopy(self.model)
        self.orig_weights.to(self.device)

    def train_run_end(self, config):
        """Method called at the end of training run."""
        # Get the overall weight updates and prune them
        logging.info("[Client #%d] Pruning weight updates.", self.client_id)
        self.prune_update()
        logging.info(
            "[Client #%d] SCR ratio (pruned amount): %.2f%%",
            self.client_id,
            self.compute_pruned_amount(),
        )

        # Add weight divergence and average update to client report
        if self.use_adaptive is True:
            # Calculate weight divergence between local and global model
            self.div_from_global = self.compute_weight_divergence()

            # Calculate average local weight updates
            self.avg_update = self.local_update_significance()

            logging.info(
                "[Client #%d] Average local weight updates: %.2f",
                self.client_id,
                self.avg_update,
            )
            logging.info(
                "[Client #%d] Weight divergence: %.2f",
                self.client_id,
                self.div_from_global,
            )

            self.run_history.update_metric("div_from_global", self.div_from_global)
            self.run_history.update_metric("avg_update", self.avg_update)

        self.model.load_state_dict(self.total_grad, strict=True)

    def compute_weight_divergence(self):
        """Calculate the divergence of the locally trained model from the global model."""
        div_from_global = 0
        for orig_layer, trained_layer in zip(
            self.orig_weights.state_dict().values(), self.model.state_dict().values()
        ):
            orig_tensor = orig_layer.cpu()
            trained_tensor = trained_layer.cpu()
            div_from_global += (
                torch.abs(torch.sum(trained_tensor - orig_tensor))
                / torch.abs(torch.sum(orig_tensor))
            ).numpy()

        return div_from_global

    def local_update_significance(self):
        """Calculate the average weight update."""
        delta = 0
        total = 0

        for layer in self.total_grad.values():
            tensor = layer.cpu()
            delta += torch.sum(tensor).numpy()

        for layer in self.model.state_dict().values():
            tensor = layer.cpu()
            total += torch.sum(tensor).numpy()

        return np.abs(delta / total)
