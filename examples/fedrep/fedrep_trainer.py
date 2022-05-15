"""
Implement the trainer for Fedrep method.

"""
import os
import time
import logging

import torch
import numpy as np

from opacus import GradSampleModule
from opacus.privacy_engine import PrivacyEngine
from opacus.validators import ModuleValidator

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__(model)

        self.model_representation_weights_key = []
        self.model_head_weights_key = []

    def set_global_local_weights_key(self, global_keys):
        """ Setting the global local weights key. """
        # the representation keys are obtained from
        #   the server response
        self.model_representation_weights_key = global_keys
        # the left weights are regarded as the head in default
        full_model_weights_key = self.model.state_dict().keys()
        self.model_head_weights_key = [
            name for name in full_model_weights_key if name not in global_keys
        ]
        logging.info(("representation_weights: {}").format(
            self.model_representation_weights_key))
        logging.info(("head_weights: {}").format(self.model_head_weights_key))

    def train_model(self, config, trainset, sampler, cut_layer=None):
        """The main training loop of FedRep in a federated learning workload. 
        
            The local training stage contains two parts:
                - head optimization:
                Makes τ local gradient-based updates to solve for its optimal head given 
                the current global representation communicated by the server.
                - representation optimization:
                Takes one local gradient-based update with respect to the current representation,
        """
        batch_size = config['batch_size']
        log_interval = 10
        tic = time.perf_counter()

        logging.info("[Client #%d] Loading the dataset.", self.client_id)
        _train_loader = getattr(self, "train_loader", None)

        if callable(_train_loader):
            train_loader = self.train_loader(batch_size, trainset, sampler,
                                             cut_layer)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       shuffle=False,
                                                       batch_size=batch_size,
                                                       sampler=sampler)

        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        # load the total local update epochs
        epochs = config['epochs']
        # load the local update epochs for head optimization
        head_epochs = config['head_epochs']

        representation_epochs = epochs - head_epochs
        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer
        get_optimizer = getattr(self, "get_optimizer",
                                optimizers.get_optimizer)
        optimizer = get_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if hasattr(config, 'lr_schedule'):
            lr_schedule = optimizers.get_lr_schedule(optimizer,
                                                     iterations_per_epoch,
                                                     train_loader)
        else:
            lr_schedule = None

        if 'differential_privacy' in config and config['differential_privacy']:
            privacy_engine = PrivacyEngine(accountant='rdp', secure_mode=False)

            self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=config['dp_epsilon']
                if 'dp_epsilon' in config else 10.0,
                target_delta=config['dp_delta']
                if 'dp_delta' in config else 1e-5,
                epochs=epochs,
                max_grad_norm=config['dp_max_grad_norm']
                if 'max_grad_norm' in config else 1.0,
            )

        for epoch in range(1, epochs + 1):

            if epoch <= head_epochs:
                for name, param in self.model.named_parameters():
                    if name in self.model_representation_weights_key:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            if epoch > head_epochs:
                for name, param in self.model.named_parameters():
                    if name in self.model_representation_weights_key:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # Use a default training loop
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                if 'differential_privacy' in config and config[
                        'differential_privacy']:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad()

                if cut_layer is None:
                    outputs = self.model(examples)
                else:
                    outputs = self.model.forward_from(examples, cut_layer)

                loss = loss_criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(), epoch, epochs, batch_id,
                            len(train_loader), loss.data.item())
                    else:
                        logging.info(
                            "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            self.client_id, epoch, epochs, batch_id,
                            len(train_loader), loss.data.item())

            if lr_schedule is not None:
                lr_schedule.step()

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

            # Simulate client's speed
            if self.client_id != 0 and hasattr(
                    Config().clients,
                    "speed_simulation") and Config().clients.speed_simulation:
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if hasattr(Config().server,
                       'request_update') and Config().server.request_update:
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)
