import random

import numpy as np
from plato.utils.export_target_model import export_target_model, get_shapes_dict
from plato.attacker.data_sampler import AttackDataLoader
from plato.attacker.membership_inference import train_attack_model, _attack
from plato.utils.homo_enc import check_accuracy
from plato.config import Config
import plato.models.registry as model_registry

from plato.utils.csv_processor import initialize_csv, write_csv

class AttackAgent:
    def __init__(self, config) -> None:
        self.model_name = config.trainer.model_name
        self.dataset = config.data.datasource
        self.encryption_ratio = config.clients.encrypt_ratio
        self.random_mask = False
        self.base_dir = f"checkpoints/{self.dataset}_{self.model_name}_{self.encryption_ratio}"
        if config.clients.random_mask:
            self.base_dir += "_random"

        self.num_clients = config.clients.total_clients
        
        self.num_classes = config.data.num_classes
        self.random_seed = config.data.random_seed
        self.partition_size = config.data.partition_size

        self.model_instance = model_registry.get()
        self.model_shape_dict = get_shapes_dict(self.model_instance)
        self.models = {}
        self.models["est"] = {}
        self.models["plain"] = {}
        self.load_models()

        self.accuracies = {}
        self.accuracies["est"] = {}
        self.accuracies["plain"] = {}

        self.dataloader = AttackDataLoader(self.num_clients, self.partition_size, self.random_seed)

        # self.eval_accuracy()

        self.load_weights(self.models["latest"])
        self.attack_model = train_attack_model(shadow_model = self.model_instance, 
                                                shadow_client_loaders = self.dataloader.get_shadow_loader(),
                                                shadow_test_loader = self.dataloader.get_test_loader(),
                                                N_class = self.num_classes)
    
    def load_weights(self, model_weights):
        self.model_instance.load_state_dict(model_weights)

    def eval_accuracy(self):
        # Evaluate Latest model
        self.load_weights(self.models["latest"])
        acc = check_accuracy(self.dataloader.get_test_loader(), self.model_instance)
        self.accuracies["latest"] = acc

        # Evaluate init model
        self.load_weights(self.models["init"])
        acc = check_accuracy(self.dataloader.get_test_loader(), self.model_instance)
        self.accuracies["init"] = acc

        # Evaluate est and plain model
        for i in range(1, self.num_clients + 1):
            self.load_weights(self.models["est"][i])
            est_acc = check_accuracy(self.dataloader.get_test_loader(), self.model_instance)

            self.load_weights(self.models["plain"][i])
            plain_acc = check_accuracy(self.dataloader.get_test_loader(), self.model_instance)

            self.accuracies["est"][i] = est_acc
            self.accuracies["plain"][i] = plain_acc

    def get_model(self, filename):
        try:
            model = export_target_model(self.model_shape_dict, filename)
            return model
        except:
            return None

    def load_models(self):
        # load init model
        init_file = f"{self.base_dir}/init.pth"
        self.models["init"] = self.get_model(init_file)

        # load lastest model
        latest_file = f"{self.base_dir}/latest.pth"
        self.models["latest"] = self.get_model(latest_file)

        # load estimated and plain model
        for i in range(1, self.num_clients + 1):
            est_file = f"{self.base_dir}/{self.model_name}_est_{i}.pth"
            plain_file = f"{self.base_dir}/{self.model_name}_plain_{i}.pth"
            self.models["est"][i] = self.get_model(est_file)
            self.models["plain"][i] = self.get_model(plain_file)
    
    def attack(self):
        result_file = f"{self.base_dir}/result.csv"
        header = ["Model", "train_acc", "test_acc", "attack_acc", "attack_precision", "attack_recall"]
        initialize_csv(result_csv_file= result_file, recorded_items=header, result_path=self.base_dir)
        
        test_loader = self.dataloader.get_test_loader()
        
        init_result = []
        plain_result = []
        est_result = []

        for i in random.sample(range(1, self.num_clients + 1), 2):
            plain_model = self.models["plain"][i]
            est_model = self.models["est"][i]
            if plain_model is None or est_model is None:
                continue
            client_loader = self.dataloader.get_train_loader(i)

            self.load_weights(self.models["init"])
            init_test_acc = check_accuracy(self.dataloader.get_test_loader(), self.model_instance)
            init_train_acc = check_accuracy(client_loader, self.model_instance)
            acc_init, pre_init, rec_init = _attack(self.model_instance, self.attack_model, 
                                    client_loader, test_loader, self.num_classes)
            newrow = [f"init_{i}",init_train_acc, init_test_acc, acc_init, pre_init, rec_init]
            init_result.append(newrow[1:])
            write_csv(result_file, newrow)

            self.load_weights(plain_model)
            plain_test_acc = check_accuracy(self.dataloader.get_test_loader(), self.model_instance)
            plain_train_acc = check_accuracy(client_loader, self.model_instance)
            acc_plain, pre_plain, rec_plain = _attack(self.model_instance, self.attack_model, 
                                  client_loader, test_loader, self.num_classes)
            newrow = [f"plain_{i}", plain_train_acc, plain_test_acc, acc_plain, pre_plain, rec_plain]
            plain_result.append(newrow[1:])
            write_csv(result_file, newrow)

            self.load_weights(est_model)
            est_test_acc = check_accuracy(self.dataloader.get_test_loader(), self.model_instance)
            est_train_acc = check_accuracy(client_loader, self.model_instance)
            acc_est, pre_est, rec_est = _attack(self.model_instance, self.attack_model, 
                                        client_loader, test_loader, self.num_classes)
            newrow = [f"est_{i}", est_train_acc, est_test_acc, acc_est, pre_est, rec_est]
            est_result.append(newrow[1:])
            write_csv(result_file, newrow)
            
            print(f"client_id: {i}\t pre_plain: {pre_plain} \t rec_plain: {rec_plain} \t pre_est: {pre_est} \t rec_est: {rec_est}")
            print(f"Plain: train acc:{plain_train_acc} \t test acc{plain_test_acc} \t Est: train acc:{est_train_acc} \t test acc{est_test_acc}")

        init_result = np.array(init_result).mean(axis=0)
        newrow = ["init_avg"] + init_result.tolist()
        write_csv(result_file, newrow)

        plain_result = np.array(plain_result).mean(axis=0)
        newrow = ["plain_avg"] + plain_result.tolist()
        write_csv(result_file, newrow)

        est_result = np.array(est_result).mean(axis=0)
        newrow = ["est_avg"] + est_result.tolist()
        write_csv(result_file, newrow)
        

