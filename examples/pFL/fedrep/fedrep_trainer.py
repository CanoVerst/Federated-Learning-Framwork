"""
A personalized federated learning trainer using FedRep.

"""

import warnings

warnings.simplefilter('ignore')

import torch
from tqdm import tqdm

from plato.trainers import pers_basic, tracking


class Trainer(pers_basic.Trainer):
    """A personalized federated learning trainer using the FedRep algorithm."""

    def freeze_model(self, model, param_prefix=None):
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                param.requires_grad = False

    def active_model(self, model, param_prefix=None):
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                param.requires_grad = True

    def train_one_epoch(self, config, optimizer, loss_criterion,
                        train_data_loader, epoch_loss_meter):
        self.model.train()
        epochs = config['epochs']
        epoch = self.current_epoch

        # load the local update epochs for head optimization
        head_epochs = config[
            'head_epochs'] if 'head_epochs' in config else epochs - 1

        # As presented in Section 3 of the FedRep paper, the head is optimized
        # for (epochs - 1) while freezing the representation.
        if epoch <= head_epochs:
            self.freeze_model(self.model, param_prefix="encoder")
            self.active_model(self.model, param_prefix="clf_fc")

        # The representation will then be optimized for only one epoch
        if epoch > head_epochs:
            self.freeze_model(self.model, param_prefix="clf_fc")
            self.active_model(self.model, param_prefix="encoder")

        epoch_loss_meter.reset()
        # Use a default training loop
        for batch_id, (examples, labels) in enumerate(train_data_loader):
            examples, labels = self.train_step_start(config,
                                                     batch_samples=examples,
                                                     batch_labels=labels)

            # Reset and clear previous data
            optimizer.zero_grad()

            # Forward the model and compute the loss
            outputs = self.model(examples)
            loss = loss_criterion(outputs, labels)

            # Perform the backpropagation
            if "create_graph" in config:
                loss.backward(create_graph=config["create_graph"])
            else:
                loss.backward()

            optimizer.step()

            # Update the loss data in the logging container
            epoch_loss_meter.update(loss, labels.size(0))

            self.train_step_end(config, batch=batch_id, loss=loss)
            self.callback_handler.call_event("on_train_step_end",
                                             self,
                                             config,
                                             batch=batch_id,
                                             loss=loss)

        if hasattr(optimizer, "params_state_update"):
            optimizer.params_state_update()

    def perform_evaluation_op(self, to_eval_data_loader, defined_model):

        # Define the test phase of the eval stage
        acc_meter = tracking.AverageMeter(name='Accuracy')
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

                correct = (preds == labels).sum()
                acc_meter.update(correct / preds.shape[0], labels.size(0))

                encoded_samples.append(features)
                loaded_labels.append(labels)

        accuracy = acc_meter.average

        test_outputs = {
            "accuracy": accuracy,
            "encoded_samples": encoded_samples,
            "loaded_labels": loaded_labels
        }

        return test_outputs

    def pers_train_one_epoch(
        self,
        config,
        pers_optimizer,
        lr_schedule,
        loss_criterion,
        train_loader,
        epoch_loss_meter,
    ):
        """ Performing one epoch of learning for the personalization. """

        epoch_loss_meter.reset()
        self.personalized_model.train()
        self.personalized_model.to(self.device)

        pers_epochs = config["pers_epochs"]
        epoch = self.current_epoch

        local_progress = tqdm(train_loader,
                              desc=f'Epoch {epoch}/{pers_epochs+1}',
                              disable=True)

        self.freeze_model(self.personalized_model, param_prefix="encoder")
        self.active_model(self.personalized_model, param_prefix="clf_fc")

        for _, (examples, labels) in enumerate(local_progress):
            examples, labels = examples.to(self.device), labels.to(self.device)
            # Clear the previous gradient
            pers_optimizer.zero_grad()

            # Perfrom the training and compute the loss
            preds = self.personalized_model(examples)
            loss = loss_criterion(preds, labels)

            # Perfrom the optimization
            loss.backward()
            pers_optimizer.step()

            # Update the epoch loss container
            epoch_loss_meter.update(loss, labels.size(0))

            local_progress.set_postfix({
                'lr': lr_schedule,
                "loss": epoch_loss_meter.val,
                'loss_avg': epoch_loss_meter.average
            })

        return epoch_loss_meter

    def pers_train_run_start(
        self,
        config,
        **kwargs,
    ):
        """ The customize behavior before performing one epoch of personalized training.
            By default, we need to save the encoded data, the accuracy, and the model when possible.
        """
        current_round = config['current_round']
        eval_outputs, _ = super().pers_train_run_start(config, **kwargs)

        self.checkpoint_encoded_samples(
            encoded_samples=eval_outputs['encoded_samples'],
            encoded_labels=eval_outputs['loaded_labels'],
            current_round=current_round,
            epoch=self.current_epoch,
            run_id=None,
            encoded_type="testEncoded")

        return eval_outputs, _

    def pers_train_epoch_end(
        self,
        config,
        **kwargs,
    ):
        current_round = config['current_round']
        eval_outputs = super().pers_train_epoch_end(config, **kwargs)

        if eval_outputs:
            self.checkpoint_encoded_samples(
                encoded_samples=eval_outputs['encoded_samples'],
                encoded_labels=eval_outputs['loaded_labels'],
                current_round=current_round,
                epoch=self.current_epoch,
                run_id=None,
                encoded_type="testEncoded")

        return eval_outputs
