"""The YOLOV5 model for PyTorch."""
import logging
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

    def train_model(self, config, trainset, sampler, cut_layer=None):
        '''Compute gradients after training'''
        self.train_loop(config, trainset, sampler, cut_layer)

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       shuffle=False,
                                                       batch_size=len(sampler),
                                                       sampler=sampler)
        examples, labels = next(iter(train_loader))
        examples, labels = examples.to(self.device), labels.to(self.device)
        self.model.to(self.device)
        outputs = self.model(examples)
        loss_criterion = torch.nn.CrossEntropyLoss()
        loss = loss_criterion(outputs, labels)
        loss.backward()
        for name, param in list(self.model.named_parameters()):
            self.gradient[name] = param.grad
        
        print(123)
