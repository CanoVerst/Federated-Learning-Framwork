import asyncio
import logging
import math
import os
import pickle
import random
import time
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from plato.config import Config
from plato.trainers import basic
from torchvision import transforms

from defense.GradDefense.dataloader import get_root_set_loader
from defense.GradDefense.sensitivity import compute_sens
from defense.outpost.perturb import compute_risk
from utils.utils import cross_entropy_for_onehot, label_to_onehot

criterion = cross_entropy_for_onehot
tt = transforms.ToPILImage()


class Trainer(basic.Trainer):
    """ The federated learning trainer for the gradient leakage attack. """

    def train_loop(self, config, trainset, sampler, cut_layer):
        """ The default training loop when a custom training loop is not supplied. """
        partition_size = Config().data.partition_size
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

        epochs = config['epochs']

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)
        self.model.train()
        last_model = deepcopy(self.model)

        target_grad = None
        total_local_steps = epochs * math.ceil(partition_size / batch_size)

        for epoch in range(1, epochs + 1):
            # Use a default training loop
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                examples.requires_grad = True
                self.model.zero_grad()

                # Store data in the first epoch (later epochs will still have the same partitioned data)
                if epoch == 1:
                    try:
                        full_examples = torch.cat((examples, full_examples),
                                                  dim=0)
                        full_labels = torch.cat((labels, full_labels), dim=0)
                    except:
                        full_examples = examples
                        full_labels = labels

                # plt.imshow(tt(examples[0].cpu()))
                # plt.title("Ground truth image")

                # Compute gradients in the current step
                if hasattr(Config().algorithm, 'defense') and Config().algorithm.defense == 'GradDefense' and \
                        hasattr(Config().algorithm, 'clip') and Config().algorithm.clip is True:
                    list_grad = []
                    for index in range(len(examples)):
                        outputs, _ = self.model(
                            torch.unsqueeze(examples[index], dim=0))

                        loss = loss_criterion(
                            outputs, torch.unsqueeze(labels[index], dim=0))
                        grad = torch.autograd.grad(
                            loss,
                            self.model.parameters(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
                        list_grad.append(
                            list((_.detach().clone() for _ in grad)))
                else:
                    outputs, feature_fc1_graph = self.model(examples)

                    # Save the ground truth and gradients
                    loss = loss_criterion(outputs, labels)
                    grad = torch.autograd.grad(
                        loss,
                        self.model.parameters(),
                        retain_graph=True,
                        create_graph=True,
                        only_inputs=True)
                    list_grad = list((_.detach().clone() for _ in grad))

                    # cast grad back to tuple type
                    grad = tuple(list_grad)

                # Update model weights with gradients and learning rate
                for (param, grad_part) in zip(self.model.parameters(), grad):
                    param.data = param.data - Config().trainer.learning_rate * grad_part

                # Sum up the gradients for each local update
                try:
                    target_grad = [
                        sum(x) for x in zip(
                            list((_.detach().clone()
                                  for _ in grad)), target_grad)
                    ]
                except:
                    target_grad = list((_.detach().clone() for _ in grad))

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

            full_onehot_labels = label_to_onehot(
                full_labels, num_classes=Config().trainer.num_classes)

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

        # Apply defense on model update (delta)
        if hasattr(Config().algorithm, 'defense') and Config().algorithm.defense != 'no':
            delta = []
            for (param, last_param) in zip(self.model.parameters(), last_model.parameters()):
                delta.append((param - last_param).detach().clone())
            if Config().algorithm.defense == 'GradDefense':
                root_set_loader = get_root_set_loader(trainset)
                sensitivity = compute_sens(model=self.model,
                                           rootset_loader=root_set_loader,
                                           device=Config().device())
                if hasattr(Config().algorithm,
                            'clip') and Config().algorithm.clip is True:
                    from defense.GradDefense.clip import noise
                else:
                    from defense.GradDefense.perturb import noise
                delta = noise(
                    dy_dx=[delta],
                    sensitivity=sensitivity,
                    slices_num=Config().algorithm.slices_num,
                    perturb_slices_num=Config().algorithm.
                    perturb_slices_num,
                    noise_intensity=Config().algorithm.scale)

            elif Config().algorithm.defense == 'GC':
                for i in range(len(delta)):
                    grad_tensor = delta[i].cpu().numpy()
                    flattened_weights = np.abs(grad_tensor.flatten())
                    # Generate the pruning threshold according to 'prune by percentage'
                    thresh = np.percentile(
                        flattened_weights, Config().algorithm.prune_pct)
                    grad_tensor = np.where(
                        abs(grad_tensor) < thresh, 0, grad_tensor)
                    delta[i] = torch.Tensor(
                        grad_tensor).to(self.device)

            elif Config().algorithm.defense == 'DP':
                for i in range(len(delta)):
                    grad_tensor = delta[i].cpu().numpy()
                    noise = np.random.laplace(
                        0, 1e-1, size=grad_tensor.shape)
                    grad_tensor = grad_tensor + noise
                    delta[i] = torch.Tensor(
                        grad_tensor).to(self.device)
            
            for (param, last_param, d) in zip(self.model.parameters(), last_model.parameters(), delta):
                param.data = last_param.data + d


        if hasattr(Config().algorithm,
                   'share_gradients') and Config().algorithm.share_gradients:
            try:
                target_grad = [x / total_local_steps for x in target_grad]
            except:
                target_grad = None

        full_examples = full_examples.detach()
        file_path = f"{Config().params['model_path']}/{self.client_id}.pickle"
        with open(file_path, 'wb') as handle:
            pickle.dump([full_examples, full_onehot_labels, target_grad],
                        handle)

    async def server_test(self, testset, sampler=None, **kwargs):
        """Testing the model on the server using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        **kwargs (optional): Additional keyword arguments.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        self.model.to(self.device)
        self.model.eval()

        custom_test = getattr(self, "test_model", None)

        if callable(custom_test):
            return self.test_model(config, testset)

        if sampler is None:
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=config['batch_size'], shuffle=False)
        else:
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=config['batch_size'],
                shuffle=False,
                sampler=sampler)

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                try:
                    outputs, _ = self.model(examples)
                except:
                    outputs = self.model(examples)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Yield to other tasks in the server
                await asyncio.sleep(0)

        return correct / total