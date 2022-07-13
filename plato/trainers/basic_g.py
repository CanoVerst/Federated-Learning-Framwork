"""The YOLOV5 model for PyTorch."""
import logging
import os
from pathlib import Path
from typing import OrderedDict

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.cuda import amp
import yaml
from tqdm import tqdm

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """The trainer with gradient computation when local training finished.."""
    def __init__(self, model=None):
        super().__init__(model = model)
        self.gradient = OrderedDict()

    def save_gradient(self, filename=None, location=None):
        """Saving the model to a file."""
        model_path = Config(
        ).params['model_path'] if location is None else location
        model_name = Config().trainer.model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            model_path = f'{model_path}/{filename}'
        else:
            model_path = f'{model_path}/{model_name}.pth'

        torch.save(self.gradient, model_path)

    def load_gradient(self, filename=None, location=None):
        """Load gradients from a file."""
        model_path = Config(
        ).params['model_path'] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f'{model_path}/{filename}'
        else:
            model_path = f'{model_path}/{model_name}.pth'

        self.gradient = torch.load(model_path)

    def train_model(self, config, trainset, sampler, cut_layer=None):
        '''Compute gradients after training'''
        self.train_loop(config, trainset, sampler, cut_layer)

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       shuffle=False,
                                                       batch_size=Config().trainer.batch_size,
                                                       sampler=sampler)
        # Set the existing gradients to zeros
        [x.grad.zero_() for x in list(self.model.parameters())]
        self.model.to(self.device)
        for idx,(examples, labels) in enumerate(train_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            outputs = self.model(examples)
            loss_criterion = torch.nn.CrossEntropyLoss()
            loss = loss_criterion(outputs, labels) 
            loss = loss * (len(labels) / len(sampler))  
            loss.backward()

        param_dict = dict(list(self.model.named_parameters()))
        state_dict = self.model.state_dict()
        for name in state_dict.keys():
            if name in param_dict:
                self.gradient[name] = param_dict[name].grad
            else:
                self.gradient[name] = torch.zeros(state_dict[name].shape)

        model_type = config['model_name']
        filename = f"{model_type}_gradient_{self.client_id}_{config['run_id']}.pth"
        self.save_gradient(filename)


    def load_model(self, filename=None, location=None):
        super().load_model(filename, location)
        model_type = Config().trainer.model_name
        run_id = Config().params["run_id"]
        filename = f"{model_type}_gradient_{self.client_id}_{run_id}.pth"
        self.load_gradient(filename)