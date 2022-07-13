import argparse
from plato.attacker.attack_agent import AttackAgent


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, required=True,
    #                 help="Name of the target model")
    # parser.add_argument("--run_id", type=str, required=True,
    #                 help="run_id of target models")
    # parser.parse_args()
    # shadow_model = None
    # target_model = None
    # client_data_loaders = None
    # test_data_loaders = None

    # attack_model = None
    agent = AttackAgent()
    agent.attack()