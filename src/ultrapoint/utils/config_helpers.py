import yaml


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config_path: str, config: dict):
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=4)
