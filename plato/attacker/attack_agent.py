from plato.utils.export_target_model import export_target_model
from plato.attacker.data_sampler import AttackDataLoader
from plato.attacker.membership_inference import train_attack_model, _attack

class AttackAgent:
    def __init__(self) -> None:
        self.base_dir = "checkpoints/CIFAR100_resnet_18_0.2"
        self.model_name = "resnet_18"
        self.dataset = "CIFAR100"
        self.num_clients = 100
        self.num_classes = 100
        self.random_seed = 1
        self.partition_size = 500

        self.models = {}
        self.models["est"] = {}
        self.models["plain"] = {}
        self.load_models()

        self.dataloader = AttackDataLoader(self.dataset, self.num_clients, self.partition_size, self.random_seed)
        self.attack_model = train_attack_model(shadow_model = self.models["latest"], 
                                                shadow_client_loaders = self.dataloader.get_shadow_loader(),
                                                shadow_test_loader = self.dataloader.get_test_loader(),
                                                N_class = self.num_classes)

    def get_model(self, filename):
        try:
            model = export_target_model(self.model_name, self.num_classes, filename)
            return model
        except:
            return None

    def load_models(self):
        # load init model
        init_file = f"{self.base_dir}/init.pth"
        self.models["init"] = self.get_model(init_file)

        # load lastest model
        latest_file = f"{self.base_dir}/init.pth"
        self.models["latest"] = self.get_model(latest_file)

        # load estimated and plain model
        for i in range(1, self.num_clients + 1):
            est_file = f"{self.base_dir}/{self.model_name}_est_{i}.pth"
            plain_file = f"{self.base_dir}/{self.model_name}_plain_{i}.pth"
            self.models["est"][i] = self.get_model(est_file)
            self.models["plain"][i] = self.get_model(plain_file)
    
    def attack(self):
        
        test_loader = self.dataloader.get_test_loader()
        for i in range(1, self.num_clients + 1):
            plain_model = self.models["plain"][i]
            est_model = self.models["est"][i]
            if plain_model is None or est_model is None:
                continue
            client_loader = self.dataloader.get_train_loader(i)
            pre_plain, rec_plain = _attack(plain_model, self.attack_model, 
                                  client_loader, test_loader, self.num_classes)

            pre_est, rec_est = _attack(est_model, self.attack_model, 
                                        client_loader, test_loader, self.num_classes)
            
            print(f"client_id: {i}\t pre_plain: {pre_plain} \t rec_plain: {rec_plain} \t pre_est: {pre_est} \t rec_est: {rec_est}")
            
