import yaml


def load_config(config_path: str):
    with open(config_path, "r") as f:
        yaml_contents = yaml.safe_load(f)

    return yaml_contents


def save_config(config_path: str, config: dict):
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
