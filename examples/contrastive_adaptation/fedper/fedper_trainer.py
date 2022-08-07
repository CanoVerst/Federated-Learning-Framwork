"""
A personalized federated learning trainer using FedPer.

"""

import logging
import warnings

warnings.simplefilter('ignore')

import torch
from tqdm import tqdm
from plato.config import Config
from plato.trainers import pers_basic
from plato.utils import optimizers

from plato.utils.checkpoint_operator import perform_client_checkpoint_saving


class Trainer(pers_basic.Trainer):
    """A personalized federated learning trainer using the FedRep algorithm."""

    def perform_evaluation_op(self, to_eval_data_loader, defined_model):

        # Define the test phase of the eval stage
        acc_meter = optimizers.AverageMeter(name='Accuracy')
        defined_model.eval()
        defined_model.to(self.device)
        correct = 0

        encoded_samples = list()
        loaded_labels = list()

        acc_meter.reset()
        for _, (examples, labels) in enumerate(to_eval_data_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            with torch.no_grad():
                # preds = self.personalized_model(examples).argmax(dim=1)

                features = defined_model.encoder(examples)
                preds = defined_model.clf_fc(features).argmax(dim=1)

                correct = (preds == labels).sum().item()
                acc_meter.update(correct / preds.shape[0], labels.size(0))

                encoded_samples.append(features)
                loaded_labels.append(labels)

        accuracy = acc_meter.avg

        outputs = {
            "accuracy": accuracy,
            "encoded_samples": encoded_samples,
            "loaded_labels": loaded_labels
        }

        return outputs

    def on_start_pers_train(
        self,
        defined_model,
        model_name,
        data_loader,
        current_round,
        epoch,
        global_epoch,
        config,
        optimizer,
        lr_schedule,
        **kwargs,
    ):
        """ The customize behavior before performing one epoch of personalized training.
            By default, we need to save the encoded data, the accuracy, and the model when possible.
        """
        eval_outputs, _ = super().on_start_pers_train(defined_model,
                                                      model_name, data_loader,
                                                      current_round, epoch,
                                                      global_epoch, config,
                                                      optimizer, lr_schedule)
        self.checkpoint_encoded_samples(
            encoded_samples=eval_outputs['encoded_samples'],
            encoded_labels=eval_outputs['loaded_labels'],
            current_round=current_round,
            epoch=epoch,
            run_id=None,
            encoded_type="testEncoded")

        return eval_outputs, _

    def on_end_pers_train_epoch(
        self,
        defined_model,
        model_name,
        data_loader,
        current_round,
        epoch,
        global_epoch,
        config,
        optimizer,
        lr_schedule,
        epoch_loss_meter,
        **kwargs,
    ):
        eval_outputs = super().on_end_pers_train_epoch(
            defined_model, model_name, data_loader, current_round, epoch,
            global_epoch, config, optimizer, lr_schedule, epoch_loss_meter)
        if eval_outputs:
            self.checkpoint_encoded_samples(
                encoded_samples=eval_outputs['encoded_samples'],
                encoded_labels=eval_outputs['loaded_labels'],
                current_round=current_round,
                epoch=epoch,
                run_id=None,
                encoded_type="testEncoded")

        return eval_outputs
