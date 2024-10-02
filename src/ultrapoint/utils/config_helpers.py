import yaml
import collections


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config_path: str, config: dict):
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
